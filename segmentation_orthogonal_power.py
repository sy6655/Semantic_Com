"""Cityscapes Q/Rayleigh inference, DDRM evaluation and allocation plots."""

import argparse
import json
import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from datasets.data_processing import CityscapesSemanticDataset
from inference import StaggeredReverseDiffusion, _freeze
from inference_oracle_gain import save_image, save_power_allocation
from posterior_sampling import add_restoration_arguments, build_restoration_sampler
from segmentation_transport import (
    load_segmentation_transmitter, prepare_segmentation_transmissions,
    restore_segmentation_transmissions,
)
from train_power_allocator import VAE_MODEL_ID, VAE_MODEL_REVISION, encode_latents


def parse_arguments(mode, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, required=mode == "evaluation")
    parser.add_argument("--max_timestep", type=int, default=999,
                        help="Allocation noise ceiling; does not change the diffusion schedule")
    parser.add_argument("--min_gain", type=float, default=0.0)
    parser.add_argument("--receiver_mode", choices=("oracle",), default="oracle")
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--diffusion_steps", type=int, default=25)
    add_restoration_arguments(parser)
    parser.set_defaults(restoration_method="ddrm")
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--guidance_max_rms", type=float, default=1.0)
    parser.add_argument("--posterior_eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legacy_awgn", action="store_true",
                        help="Dispatch to the original CNN/AWGN implementation")
    if mode == "evaluation":
        parser.add_argument("--batch", "--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_samples", type=int, default=None)
        parser.add_argument("--log_interval", type=int, default=20)
        parser.add_argument("--output_path", default=None)
        parser.add_argument("--strategy", choices=("all", "allocated", "uniform", "uncompressed"), default="all")
    else:
        parser.add_argument("--sample_index", type=int, default=0)
        parser.add_argument("--output_dir", default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.max_timestep <= 999:
        parser.error("--max_timestep must be between 0 and 999")
    if not 1 <= args.diffusion_steps <= 1000:
        parser.error("--diffusion_steps must be between 1 and 1000")
    if not torch.isfinite(torch.tensor(args.min_gain)) or args.min_gain < 0:
        parser.error("--min_gain must be finite and non-negative")
    if args.snr_db is not None and not torch.isfinite(torch.tensor(args.snr_db)):
        parser.error("--snr_db must be finite")
    if mode == "evaluation":
        if args.batch <= 0 or args.num_workers < 0 or args.log_interval <= 0:
            parser.error("batch/log_interval must be positive and num_workers non-negative")
        if args.max_samples is not None and args.max_samples <= 0:
            parser.error("--max_samples must be positive")
    elif args.sample_index < 0:
        parser.error("--sample_index must be non-negative")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or run on a GPU host")
    return args


def _load_vae(args, config, device):
    kwargs = {"local_files_only": args.local_files_only, "torch_dtype": torch.float32}
    if args.vae_path is None:
        kwargs.update(subfolder="vae", revision=config.get("vae_model_revision", VAE_MODEL_REVISION))
    vae = _freeze(AutoencoderKL.from_pretrained(
        args.vae_path or config.get("vae_model_id", VAE_MODEL_ID), **kwargs
    ).to(device))
    # encode_latents uses the SD1.5 scaling factor; disallow an incompatible VAE.
    if int(vae.config.latent_channels) != 4 or abs(float(vae.config.scaling_factor) - 0.18215) > 1e-6:
        raise ValueError("Cityscapes transport requires a four-channel SD1.5 VAE with scale 0.18215")
    return vae


def _build_sampler(args, config, device):
    reverse_model = StaggeredReverseDiffusion(
        args.diffusion_model or config["vae_model_id"], device,
        torch.float16 if args.amp and device.type == "cuda" else torch.float32,
        args.local_files_only,
        revision=config.get("vae_model_revision") if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    return build_restoration_sampler(reverse_model, args, receiver_mode="oracle")


def _run_tag(args):
    tag = f"{args.restoration_method}_rayleigh_snr_{args.snr_db:g}_steps_{args.diffusion_steps}"
    if args.restoration_method == "ddrm":
        tag += f"_eta_{args.ddrm_eta:g}_etab_{args.ddrm_eta_b:g}"
    else:
        tag += f"_guidance_{args.guidance_scale:g}_rms_{args.guidance_max_rms:g}_eta_{args.posterior_eta:g}"
    return f"{tag}_maxt_{args.max_timestep}_mingain_{args.min_gain:g}_seed_{args.seed}"


def _metadata(args, config, layout, checkpoint_path):
    return {
        "checkpoint": str(checkpoint_path), "dataset": "Cityscapes", **layout.config(),
        "channel_model": config["channel_model"], "receiver_mode": "oracle", "pilots_per_group": 0,
        "h_distribution": "CN(0,1), one complex h per image shared by all blocks",
        "noise_real_component_variance": "N0=10**(-snr_db/10)", "mean_real_component_power": 1.0,
        "snr_db": args.snr_db, "seed": args.seed, "max_timestep_power_floor": args.max_timestep,
        "min_gain": args.min_gain, "restoration_method": args.restoration_method,
        "diffusion_model": args.diffusion_model or config["vae_model_id"],
        "diffusion_steps": args.diffusion_steps,
        "restoration_parameters": ({"eta": args.ddrm_eta, "eta_b": args.ddrm_eta_b,
                                    "initialization_policy": "clip"}
                                   if args.restoration_method == "ddrm" else
                                   {"guidance_scale": args.guidance_scale,
                                    "guidance_max_rms": args.guidance_max_rms, "eta": args.posterior_eta}),
        "same_images_and_vae_latents_for_strategies": True,
        "same_h_and_channel_noise_for_strategies": True,
        "same_diffusion_noise_for_strategies": True,
        "uniform_definition": "spatial latent, no Q, constant gain under the same frame energy budget",
        "allocated_definition": "learned Q and power allocator with perfect CSI",
        "uncompressed_definition": "VAE encode/decode without channel noise",
    }


def _write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _transmit(args, latent, allocator, channel, config, generator, strategies):
    return prepare_segmentation_transmissions(
        latent, torch.full((latent.shape[0],), args.snr_db, device=latent.device),
        allocator, channel, generator, float(config.get("temperature", 1.0)),
        max_timestep=args.max_timestep, min_gain=args.min_gain, strategies=strategies,
    )


def run_single(args, allocation_only=False):
    from visualize_segmentation_power import _colorize_ground_truth, _decode_latent, _save_labeled_row

    device = torch.device(args.device)
    allocator, channel, layout, checkpoint, checkpoint_path = load_segmentation_transmitter(args.checkpoint, device)
    config = checkpoint["config"]
    if args.snr_db is None:
        args.snr_db = float(sum(config["snr_db"]) / len(config["snr_db"]))
    dataset = CityscapesSemanticDataset(args.data_path or config["data_path"], "val")
    if args.sample_index >= len(dataset):
        raise IndexError(f"--sample_index must be less than {len(dataset)}")
    output_dir = Path(args.output_dir or checkpoint_path.parent / (
        "allocation" if allocation_only else "inference"
    ) / f"sample_{args.sample_index:06d}_{_run_tag(args)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = _load_vae(args, config, device)
    image, labels = dataset[args.sample_index]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    diffusion_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    result = _metadata(args, config, layout, checkpoint_path)
    result.update(sample_index=args.sample_index, output_dir=str(output_dir),
                  restoration_applied=not allocation_only, epoch=int(checkpoint["epoch"]) + 1)
    with torch.no_grad():
        latent = encode_latents(image.unsqueeze(0).to(device), vae, generator)
        layout.check_latent(latent)
        transmission = _transmit(args, latent, allocator, channel, config, channel_generator,
                                 ("uniform", "allocated"))
        restored, diagnostics = {}, {}
        if not allocation_only:
            sampler = _build_sampler(args, config, device)
            restored, diagnostics = restore_segmentation_transmissions(
                transmission, sampler, diffusion_generator,
            )
        save_image(image.unsqueeze(0), output_dir / "input.png")
        save_image(_colorize_ground_truth(labels).unsqueeze(0).float() / 255,
                   output_dir / "ground_truth.png")
        clean = _decode_latent(vae, latent)
        save_image(clean, output_dir / "vae_reconstruction.png")
        panels, titles = [clean[0]], ["VAE reconstruction"]
        strategies = {}
        tensor_state = {"latent": latent.cpu(), "h": transmission["h"].cpu(),
                        "N0": transmission["N0"].cpu(),
                        "Q": transmission["allocation_info"]["orthogonal_matrix"].cpu()}
        for name, received in transmission["strategies"].items():
            direct = _decode_latent(vae, received["direct_latent"])
            save_image(direct, output_dir / f"{name}_received_direct.png")
            strategies[name] = {key: received[key][0].cpu().tolist() for key in (
                "group_power", "energy_allocation_ratio", "noise_variance", "precision")}
            strategies[name].update(uses_Q=received["uses_Q"],
                                    frame_energy=received["frame_energy"].item(),
                                    diffusion=diagnostics.get(name))
            tensor_state[name] = {key: received[key].cpu() for key in (
                "observation", "precision", "group_power", "direct_latent")}
            if name in restored:
                denoised = _decode_latent(vae, restored[name])
                save_image(denoised, output_dir / f"{name}_{args.restoration_method}.png")
                _save_labeled_row([clean[0], direct[0], denoised[0]],
                                  ["VAE reconstruction", f"{name} received", f"{name} {args.restoration_method}"],
                                  output_dir / f"{name}_denoising_comparison.png")
                panels.append(denoised[0])
                titles.append(f"{name} {args.restoration_method}")
                tensor_state[name]["restored_latent"] = restored[name].cpu()
        save_power_allocation(strategies, output_dir / "power_allocation.png", layout.rows, layout.cols)
        if restored:
            _save_labeled_row(panels, titles, output_dir / "denoised_comparison.png")
        result.update(strategies=strategies, h_real=transmission["h"].real.item(),
                      h_imag=transmission["h"].imag.item(), h_abs=transmission["h"].abs().item(),
                      gain_floor_infeasible_images=int(transmission["allocation_info"]["gain_floor_fallback"].sum()))
        torch.save(tensor_state, output_dir / "latents.pt")
    _write_json(output_dir / "result.json", result)
    print(f"Saved: {output_dir}", flush=True)
    return result


def evaluate(args):
    from eval_segmentation_power_allocator import update_confusion
    from models.models import construct_model
    from train_segmentation_power_allocator import NUM_CLASSES, segmentation_metrics

    device = torch.device(args.device)
    allocator, channel, layout, checkpoint, checkpoint_path = load_segmentation_transmitter(args.checkpoint, device)
    config = checkpoint["config"]
    selected = ("uniform", "allocated", "uncompressed") if args.strategy == "all" else (args.strategy,)
    transmitted = tuple(name for name in selected if name != "uncompressed")
    dataset = CityscapesSemanticDataset(args.data_path or config["data_path"], "val")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    limit = min(len(dataset), args.max_samples or len(dataset))
    if not limit:
        raise ValueError("Cityscapes validation dataset is empty")
    output_path = Path(args.output_path or checkpoint_path.parent / "evaluation" /
                       f"mask2former_{_run_tag(args)}_{args.strategy}_n{limit}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("segmentation_evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(output_path.with_suffix(".log"))):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    logger.info("Q=%dx%d patch=%dx%d grid=%dx%d samples=%d output=%s", layout.num_groups,
                layout.num_groups, layout.patch_height, layout.patch_width, layout.rows, layout.cols,
                limit, output_path)
    vae = _load_vae(args, config, device)
    evaluator = _freeze(construct_model("Mask2Former").to(device))
    sampler = _build_sampler(args, config, device) if transmitted else None
    vae_generator = torch.Generator(device=device).manual_seed(args.seed)
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    diffusion_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    matrices = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64) for name in selected}
    totals = {name: {"latent_snr_db": 0., "real_power": 0., "clipped_images": 0, "clipped_blocks": 0}
              for name in transmitted}
    num_samples, infeasible, h_squared_sum = 0, 0, 0.
    result = _metadata(args, config, layout, checkpoint_path)
    result.update(evaluator="Mask2Former (pretrained Cityscapes)", dataset_split="val",
                  dataset_size=len(dataset), epoch=int(checkpoint["epoch"]) + 1,
                  log_path=str(output_path.with_suffix(".log")))

    def save_progress():
        metrics = {}
        for name, matrix in matrices.items():
            accuracy, miou = segmentation_metrics(matrix)
            metrics[name] = {"pixel_accuracy": accuracy, "miou": miou}
            if name in totals:
                values = totals[name]
                metrics[name].update(
                    uses_Q=name == "allocated", mean_latent_snr_db=values["latent_snr_db"] / num_samples,
                    mean_real_component_power=values["real_power"] / num_samples,
                    ddrm_initialization_clipped_images=values["clipped_images"],
                    ddrm_initialization_clipped_blocks=values["clipped_blocks"],
                )
        primary = "allocated" if "allocated" in metrics else selected[0]
        result.update(num_samples=num_samples, complete=num_samples == limit, strategies=metrics,
                      primary_strategy=primary, **metrics[primary],
                      gain_floor_infeasible_images=infeasible,
                      mean_h_squared=h_squared_sum / num_samples if transmitted else None,
                      sample_selection={"method": "dataset_order", "start_index": 0,
                                        "stop_index_exclusive": num_samples})
        _write_json(output_path, result)
        return " ".join(f"{name}:mIoU={values['miou']:.4f}" for name, values in metrics.items())

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if num_samples >= limit:
                break
            images, labels = images[:limit - num_samples].to(device), labels[:limit - num_samples].to(device)
            latent = encode_latents(images, vae, vae_generator)
            layout.check_latent(latent)
            restored = {}
            if transmitted:
                transmission = _transmit(args, latent, allocator, channel, config, channel_generator, transmitted)
                restored, diagnostics = restore_segmentation_transmissions(
                    transmission, sampler, diffusion_generator, verbose=False,
                )
                infeasible += int(transmission["allocation_info"]["gain_floor_fallback"].sum())
                h_squared_sum += transmission["h"].abs().square().sum().item()
                for name in transmitted:
                    received, diag = transmission["strategies"][name], diagnostics[name]
                    totals[name]["latent_snr_db"] += (10 * received["channel_info"]["snr"].log10()).mean(1).sum().item()
                    totals[name]["real_power"] += (received["frame_energy"] / latent[0].numel()).sum().item()
                    totals[name]["clipped_images"] += diag.get("initialization_clipped_images", 0)
                    totals[name]["clipped_blocks"] += diag.get("initialization_clipped_block_count", 0)
            for name in selected:
                update_confusion(matrices[name], evaluator, vae,
                                 latent if name == "uncompressed" else restored[name], labels)
            num_samples += images.shape[0]
            if batch_index == 0 or (batch_index + 1) % args.log_interval == 0 or num_samples == limit:
                summary = save_progress()
                logger.info("samples=%d/%d %s gain_floor_infeasible=%d DDRM_clipped=%s", num_samples, limit,
                            summary, infeasible,
                            {name: (values["clipped_images"], values["clipped_blocks"])
                             for name, values in totals.items()})
    save_progress()
    logger.info("Saved: %s", output_path)
    return result


def main(mode):
    args = parse_arguments(mode)
    if mode == "evaluation":
        return evaluate(args)
    return run_single(args, allocation_only=mode == "allocation")


def dispatch(mode, legacy_main):
    """Keep original CLI implementations available under an explicit legacy flag."""
    import sys
    if "--legacy_awgn" in sys.argv[1:]:
        original = sys.argv[:]
        sys.argv = [value for value in sys.argv if value != "--legacy_awgn"]
        try:
            return legacy_main()
        finally:
            sys.argv = original
    return main(mode)
