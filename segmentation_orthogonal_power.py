"""Cityscapes Q transport, Mask2Former evaluation and allocation plots."""

import argparse
import json
import logging
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.data_processing import CityscapesSemanticDataset
from inference import StaggeredReverseDiffusion, _freeze
from inference_oracle_gain import save_image, save_power_allocation
from orthogonal_power import allocator_fingerprint, load_gain_estimator
from posterior_sampling import add_restoration_arguments, build_restoration_sampler
from reconstruction_metrics import LPIPS_NETWORK, image_quality_batch, load_lpips
from segmentation_transport import (
    load_segmentation_transmitter, prepare_segmentation_transmissions,
    restore_segmentation_transmissions,
)
from train_power_allocator import VAE_MODEL_ID, VAE_MODEL_REVISION, encode_latents


def parse_arguments(mode, argv=None):
    if mode == "inference":
        from inference_orthogonal_power import build_parser, validate_arguments
        parser = build_parser()
        parser.set_defaults(diffusion_steps=25, seed=20260910, snr_db=None)
        args = parser.parse_args(argv)
        validate_arguments(args, parser)
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu or run on a GPU host")
        return args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, **(
        {"nargs": "+", "default": [0.0]} if mode == "evaluation" else {"default": None}
    ))
    parser.add_argument("--max_timestep", type=int, default=999,
                        help="Allocation noise ceiling; does not change the diffusion schedule")
    parser.add_argument("--min_gain", type=float, default=0.0)
    parser.add_argument("--receiver_mode", choices=("oracle", "gain_estimator") if mode == "evaluation" else ("oracle",),
                        default=None if mode == "evaluation" else "oracle",
                        help="Allocated receiver: estimator when supplied, otherwise oracle")
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="Use float16 for the diffusion model on CUDA; VAE and Mask2Former stay float32")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legacy_awgn", action="store_true",
                        help="Dispatch to the original CNN/AWGN implementation")
    if mode == "evaluation":
        parser.add_argument("--channel_model", choices=("awgn", "rayleigh"), default="rayleigh",
                            help="awgn: h=1; rayleigh: one CN(0,1) h per image shared by all blocks")
        parser.add_argument("--gain_estimator_checkpoint", default=None)
        parser.add_argument("--uniform_receiver_mode", choices=("oracle", "gain_estimator"), default="oracle")
        parser.add_argument("--batch_size", "--batch", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_samples", type=int, default=None,
                            help="Default: full Cityscapes val; otherwise a seeded subset without replacement")
        parser.add_argument("--log_interval", type=int, default=20)
        parser.add_argument("--output_path", default=None)
        parser.add_argument("--strategy", choices=("all", "both", "allocated", "uniform", "vae", "uncompressed"),
                            default="all", help="all: VAE/uniform/allocated; both: uniform/allocated; uncompressed: alias for vae")
        parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
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
    if args.snr_db is not None and not torch.isfinite(torch.tensor(args.snr_db)).all():
        parser.error("--snr_db must be finite")
    if mode == "evaluation":
        if args.batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0:
            parser.error("batch/log_interval must be positive and num_workers non-negative")
        if args.max_samples is not None and args.max_samples <= 0:
            parser.error("--max_samples must be positive")
        args.receiver_mode = args.receiver_mode or ("gain_estimator" if args.gain_estimator_checkpoint else "oracle")
        modes = {"allocated": args.receiver_mode, "uniform": args.uniform_receiver_mode}
        selected = evaluation_strategies(args.strategy)
        if any(modes.get(name) == "gain_estimator" for name in selected) and not args.gain_estimator_checkpoint:
            parser.error("--gain_estimator_checkpoint is required when an evaluated receiver uses gain_estimator")
        try:
            build_restoration_sampler(None, args, receiver_mode=args.receiver_mode)
        except ValueError as error:
            parser.error(str(error))
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


def _build_sampler(args, config, device, receiver_mode=None):
    reverse_model = StaggeredReverseDiffusion(
        args.diffusion_model or config["vae_model_id"], device,
        torch.float16 if args.amp and device.type == "cuda" else torch.float32,
        args.local_files_only,
        revision=config.get("vae_model_revision") if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    return build_restoration_sampler(reverse_model, args, receiver_mode=receiver_mode or args.receiver_mode)


def _run_tag(args, snr_db=None):
    snr_db = args.snr_db if snr_db is None else snr_db
    tag = f"{args.restoration_method}_{getattr(args, 'channel_model', 'rayleigh')}_snr_{snr_db:g}_steps_{args.diffusion_steps}"
    if args.restoration_method == "ddrm":
        tag += f"_eta_{args.ddrm_eta:g}_etab_{args.ddrm_eta_b:g}"
    else:
        tag += f"_guidance_{args.guidance_scale:g}_rms_{args.guidance_max_rms:g}_eta_{args.posterior_eta:g}"
    return f"{tag}_maxt_{args.max_timestep}_mingain_{args.min_gain:g}_seed_{args.seed}"


def _metadata(args, config, layout, checkpoint_path):
    return {
        "checkpoint": str(checkpoint_path), "dataset": "Cityscapes", **layout.config(),
        "channel_model": getattr(args, "channel_model", "rayleigh"),
        "receiver_mode": args.receiver_mode, "pilots_per_group": 0,
        "h_distribution": ("h=1" if getattr(args, "channel_model", "rayleigh") == "awgn" else
                           "CN(0,1), one complex h per image shared by all blocks"),
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
        "allocated_definition": "learned Q and power allocator with perfect channel CSI; selected gain receiver",
        "vae_definition": "VAE encode/decode without channel noise",
        "uncompressed_definition": "VAE encode/decode without channel noise",
    }


def _write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _transmit(args, latent, allocator, channel, config, generator, strategies, *, snr_db=None, gain_estimator=None):
    return prepare_segmentation_transmissions(
        latent, torch.full((latent.shape[0],), args.snr_db if snr_db is None else snr_db, device=latent.device),
        allocator, channel, generator, float(config.get("temperature", 1.0)),
        max_timestep=args.max_timestep, min_gain=args.min_gain, strategies=strategies,
        channel_model=getattr(args, "channel_model", "rayleigh"), receiver_mode=args.receiver_mode,
        uniform_receiver_mode=getattr(args, "uniform_receiver_mode", "oracle"), gain_estimator=gain_estimator,
    )


def run_single(args, allocation_only=False):
    if not allocation_only:
        from inference_orthogonal_power import run_inference
        return run_inference(args)
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
        save_image(image.unsqueeze(0), output_dir / "input.png")
        save_image(_colorize_ground_truth(labels).unsqueeze(0).float() / 255,
                   output_dir / "ground_truth.png")
        clean = _decode_latent(vae, latent)
        save_image(clean, output_dir / "vae_reconstruction.png")
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
                                    diffusion=None)
            tensor_state[name] = {key: received[key].cpu() for key in (
                "observation", "precision", "group_power", "direct_latent")}
        save_power_allocation(strategies, output_dir / "power_allocation.png", layout.rows, layout.cols)
        result.update(strategies=strategies, h_real=transmission["h"].real.item(),
                      h_imag=transmission["h"].imag.item(), h_abs=transmission["h"].abs().item(),
                      gain_floor_infeasible_images=int(transmission["allocation_info"]["gain_floor_fallback"].sum()))
        torch.save(tensor_state, output_dir / "latents.pt")
    _write_json(output_dir / "result.json", result)
    print(f"Saved: {output_dir}", flush=True)
    return result


EVALUATION_STRATEGIES = ("vae", "uniform", "allocated")


def evaluation_strategies(strategy):
    if strategy == "all":
        return EVALUATION_STRATEGIES
    if strategy == "both":
        return ("uniform", "allocated")
    return ("vae" if strategy == "uncompressed" else strategy,)


class IndexedCityscapes(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def select_evaluation_indices(dataset, max_samples, seed):
    """Cityscapes has multiple classes per image; sample images, not class quotas."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    limit = min(len(dataset), max_samples or len(dataset))
    if limit == 0:
        raise ValueError("Cityscapes validation dataset is empty")
    if limit == len(dataset):
        return list(range(limit))
    generator = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(len(dataset), generator=generator)[:limit].tolist())


def evaluation_output_path(args, checkpoint_path, snr_db, sample_count):
    if args.output_path:
        path = Path(args.output_path).expanduser()
        if not path.suffix:
            path = path.with_suffix(".json")
        if len(args.snr_db) > 1:
            path = path.with_name(f"{path.stem}_snr_{snr_db:g}{path.suffix}")
        return path
    return checkpoint_path.parent / "evaluation" / (
        f"mask2former_{_run_tag(args, snr_db)}_{args.strategy}_n{sample_count}"
        f"_receiver_{args.receiver_mode}_uniform_{args.uniform_receiver_mode}.json"
    )


def _evaluation_logger(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("segmentation_evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(output_path.with_suffix(".log"), encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


@torch.no_grad()
def evaluate(args):
    from eval_segmentation_power_allocator import update_confusion_rgb
    from inference_power import _decode_latent
    from models.models import construct_model
    from train_segmentation_power_allocator import NUM_CLASSES, segmentation_metrics

    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; run on a GPU host or specify --device cpu")
        device = torch.device("cuda", 0) if device.index is None else device
        torch.cuda.set_device(device)
    print(f"Loading transmitter: {args.checkpoint}", flush=True)
    allocator, channel, layout, checkpoint, checkpoint_path = load_segmentation_transmitter(args.checkpoint, device)
    config = checkpoint["config"]
    selected = evaluation_strategies(args.strategy)
    transmitted = tuple(name for name in selected if name != "vae")
    receiver_modes = {"uniform": args.uniform_receiver_mode, "allocated": args.receiver_mode}
    dataset_root = Path(args.data_path or config["data_path"]).expanduser()
    dataset = CityscapesSemanticDataset(dataset_root, "val")
    indices = select_evaluation_indices(dataset, args.max_samples, args.seed)
    limit = len(indices)
    loader = DataLoader(Subset(IndexedCityscapes(dataset), indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    estimator, estimator_config, estimator_path = None, None, None
    if any(receiver_modes[name] == "gain_estimator" for name in transmitted):
        estimator_path = Path(args.gain_estimator_checkpoint).expanduser()
        if estimator_path.is_dir():
            estimator_path /= "latest.pth"
        print(f"Loading gain estimator: {estimator_path}", flush=True)
        estimator, estimator_config = load_gain_estimator(
            estimator_path, allocator, device, check_allocator_fingerprint=False,
        )
    print("Loading VAE...", flush=True)
    vae = _load_vae(args, config, device)
    print("Loading pretrained Cityscapes Mask2Former...", flush=True)
    evaluator = _freeze(construct_model("Mask2Former").to(device))
    print(f"Loading image quality model (LPIPS={args.lpips})...", flush=True)
    lpips_model = load_lpips(device, enabled=args.lpips, local_files_only=args.local_files_only)
    samplers = {}
    if transmitted:
        print(f"Loading diffusion model ({args.restoration_method.upper()}, {args.diffusion_steps} steps)...", flush=True)
        # One frozen UNet/text encoder is shared even when receiver modes differ.
        first_mode = receiver_modes[transmitted[0]]
        sampler = _build_sampler(args, config, device, receiver_mode=first_mode)
        by_mode = {first_mode: sampler}
        for name in transmitted:
            mode = receiver_modes[name]
            if mode not in by_mode:
                by_mode[mode] = build_restoration_sampler(sampler.model, args, receiver_mode=mode)
            samplers[name] = by_mode[mode]

    current_fingerprint = allocator_fingerprint(allocator)
    trained_fingerprint = estimator_config.get("allocator_fingerprint") if estimator_config is not None else None
    result = _metadata(args, config, layout, checkpoint_path)
    result.update(
        evaluator="Mask2Former (pretrained Cityscapes)", model="mask2former_swin_base_cityscapes",
        dataset_split="val", dataset_size=len(dataset), epoch=int(checkpoint.get("epoch", -1)) + 1,
        batch_size=args.batch_size, amp=args.amp, strategy=args.strategy,
        uniform_receiver_mode=args.uniform_receiver_mode,
        strategy_receiver_modes={name: receiver_modes[name] for name in transmitted},
        uses_gain_estimator=estimator is not None,
        gain_estimator_checkpoint=str(estimator_path) if estimator_path is not None else None,
        gain_estimator_config=estimator_config, allocator_fingerprint=current_fingerprint,
        training_allocator_fingerprint=trained_fingerprint,
        gain_estimator_allocator_matches=(current_fingerprint == trained_fingerprint if estimator is not None else None),
        strict_allocator_binding=False, perfect_csi=True,
        lpips_network=LPIPS_NETWORK if lpips_model is not None else None,
        quality_reference="original_full_resolution_RGB", vae_latent_method="sample",
        accuracy_preprocessing="RGB_[0,255]_native_resolution_Mask2Former_internal_normalization",
        uniform_uses_Q=False, allocated_uses_Q=True,
        mean_symbol_power=2.0, target_frame_energy=layout.latent_channels * layout.latent_height * layout.latent_width,
        max_timestep=args.max_timestep, gain_floor_infeasible_policy="uniform_gain_same_frame_energy",
        ddrm_initialization_policies={name: getattr(sampler, "initialization_policy", None)
                                      for name, sampler in samplers.items()},
        num_samples_requested=limit,
        sample_selection={
            "method": "dataset_order" if limit == len(dataset) else "seeded_without_replacement",
            "seed": args.seed, "dataset_root": str(dataset_root.resolve()), "dataset_size": len(dataset),
            "indices": indices,
            "relative_paths": [str(Path(dataset.samples[index][0]).relative_to(dataset_root)
                                   if Path(dataset.samples[index][0]).is_relative_to(dataset_root)
                                   else dataset.samples[index][0]) for index in indices],
            "same_images_for_strategies": True, "same_images_for_snrs": True,
        },
        same_vae_sample_for_snrs=True, same_h_and_awgn_draws_for_snrs=True,
        rng_seeding="seed + 4*first_dataset_index_in_batch; VAE/channel/diffusion offsets 0/1/2",
        evaluations={},
    )
    print(f"Cityscapes val {limit}/{len(dataset)} images; strategies={selected}; "
          f"channel={args.channel_model}; receivers={receiver_modes}", flush=True)

    for snr_db in args.snr_db:
        snr_started = time.perf_counter()
        output_path = evaluation_output_path(args, checkpoint_path, snr_db, limit)
        logger = _evaluation_logger(output_path)
        logger.info("checkpoint=%s channel=%s SNR=%g Q=%dx%d grid=%dx%d receivers=%s samples=%d output=%s",
                    checkpoint_path, args.channel_model, snr_db, layout.num_groups, layout.num_groups,
                    layout.rows, layout.cols, receiver_modes, limit, output_path)
        matrices = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64) for name in selected}
        quality_totals = {name: {"mse": 0., "psnr_db": 0., "lpips": 0.} for name in selected}
        totals = {name: {"latent_snr_db": 0., "real_power": 0., "clipped_images": 0, "clipped_blocks": 0}
                  for name in transmitted}
        num_samples, infeasible, h_squared_sum, h_fourth_sum = 0, 0, 0., 0.

        def save_progress():
            metrics = {}
            for name, matrix in matrices.items():
                accuracy, miou = segmentation_metrics(matrix)
                union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
                metrics[name] = {
                    "num_samples": num_samples, "pixel_accuracy": accuracy, "miou": miou,
                    "mse": quality_totals[name]["mse"] / num_samples,
                    "psnr_db": quality_totals[name]["psnr_db"] / num_samples,
                    "lpips": quality_totals[name]["lpips"] / num_samples if lpips_model is not None else None,
                    "confusion_matrix": matrix.tolist(), "num_valid_pixels": int(matrix.sum()),
                    "per_class_iou": [int(tp) / int(u) if u else None for tp, u in zip(matrix.diag(), union)],
                }
                if name in totals:
                    values = totals[name]
                    metrics[name].update(
                        receiver_mode=receiver_modes[name], uses_Q=name == "allocated",
                        mean_latent_snr_db=values["latent_snr_db"] / num_samples,
                        latent_snr_definition="source power / receiver-assumed spatial noise variance",
                        mean_real_component_power=values["real_power"] / num_samples,
                        ddrm_initialization_clipped_images=values["clipped_images"],
                        ddrm_initialization_clipped_blocks=values["clipped_blocks"],
                    )
            primary = "allocated" if "allocated" in metrics else selected[0]
            mean_h_power = h_squared_sum / num_samples if transmitted else None
            summary = {
                "snr_db": snr_db, "num_samples": num_samples, "complete": num_samples == limit,
                "strategies": metrics, "primary_strategy": primary, "gain_floor_infeasible_images": infeasible,
                "mean_h_power": mean_h_power, "mean_h_squared": mean_h_power,
                "std_h_power": max(h_fourth_sum / num_samples - mean_h_power**2, 0.)**.5 if transmitted else None,
                "ddrm_initialization_clipped": {name: {"images": values["clipped_images"], "blocks": values["clipped_blocks"]}
                                                for name, values in totals.items()},
                "output_path": str(output_path), "log_path": str(output_path.with_suffix(".log")),
                "elapsed_seconds": time.perf_counter() - snr_started,
            }
            result["evaluations"][f"{snr_db:g}"] = summary
            # Keep the previous top-level metrics for consumers of single-SNR results.
            compatible = {**summary, **{key: value for key, value in metrics[primary].items() if key != "receiver_mode"}}
            result.update({key: value for key, value in compatible.items() if key != "snr_db"})
            result["total_elapsed_seconds"] = time.perf_counter() - started
            _write_json(output_path, {**result, **compatible,
                                     "evaluations": {f"{snr_db:g}": summary}})
            return metrics

        try:
            for batch_index, (images, labels, sample_indices) in enumerate(loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                # Replay the same VAE sample, h and epsilon at every SNR.
                batch_seed = args.seed + 4 * int(sample_indices[0])
                vae_generator = torch.Generator(device=device).manual_seed(batch_seed)
                channel_generator = torch.Generator(device=device).manual_seed(batch_seed + 1)
                diffusion_generator = torch.Generator(device=device).manual_seed(batch_seed + 2)
                latent = encode_latents(images, vae, vae_generator)
                layout.check_latent(latent)
                restored = {}
                if transmitted:
                    transmission = _transmit(args, latent, allocator, channel, config, channel_generator,
                                             transmitted, snr_db=snr_db, gain_estimator=estimator)
                    restored, diagnostics = restore_segmentation_transmissions(
                        transmission, samplers, diffusion_generator, verbose=False,
                    )
                    infeasible += int(transmission["allocation_info"]["gain_floor_fallback"].sum())
                    h_power = transmission["h"].abs().double().square()
                    h_squared_sum += h_power.sum().item()
                    h_fourth_sum += h_power.square().sum().item()
                    for name in transmitted:
                        received, diag = transmission["strategies"][name], diagnostics[name]
                        totals[name]["latent_snr_db"] += (10 * received["channel_info"]["snr"].log10()).mean(1).sum().item()
                        totals[name]["real_power"] += (received["frame_energy"] / latent[0].numel()).sum().item()
                        totals[name]["clipped_images"] += diag.get("initialization_clipped_images", 0)
                        totals[name]["clipped_blocks"] += diag.get("initialization_clipped_block_count", 0)
                for name in selected:
                    reconstructed = _decode_latent(vae, latent if name == "vae" else restored[name],
                                                  float(vae.config.scaling_factor))
                    update_confusion_rgb(matrices[name], evaluator, reconstructed, labels)
                    quality = image_quality_batch(reconstructed, images, lpips_model)
                    for key, values in quality.items():
                        quality_totals[name][key] += values.sum().item()
                num_samples += images.shape[0]
                if batch_index == 0 or (batch_index + 1) % args.log_interval == 0 or num_samples == limit:
                    metrics = save_progress()
                    for name, row in metrics.items():
                        logger.info("SNR=%g samples=%d/%d %s pixel_acc=%.3f%% mIoU=%.3f%% PSNR=%.3f LPIPS=%s",
                                    snr_db, num_samples, limit, name, 100 * row["pixel_accuracy"],
                                    100 * row["miou"], row["psnr_db"],
                                    f"{row['lpips']:.5f}" if row["lpips"] is not None else "disabled")
                    logger.info("gain_floor_infeasible=%d DDRM_initialization_clipped=%s", infeasible,
                                {name: (v["clipped_images"], v["clipped_blocks"]) for name, v in totals.items()})
                del latent, restored, reconstructed
                if transmitted:
                    del transmission
            logger.info("Saved: %s", output_path)
        finally:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    result["total_elapsed_seconds"] = time.perf_counter() - started
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
