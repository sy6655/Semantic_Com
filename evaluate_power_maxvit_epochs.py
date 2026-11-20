"""Compare allocated/Q restorations from allocator epochs with RGB MaxViT-Large.

Uses evaluate_power_maxvit's Rayleigh/oracle channel, restoration, preprocessing,
and image quality metrics. No uniform or VAE-only baseline is evaluated.
Default: epochs 10 and 15, one seeded validation image per ImageNet class.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import evaluate_power_maxvit as common


DEFAULT_MAX_SAMPLES = 1000


def load_epochs(args, device):
    directory = Path(args.checkpoint_dir).expanduser()
    paths = [directory / f"epoch_{epoch:03d}.pth" for epoch in args.epochs]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Epoch checkpoint not found: {path}")
    transmitters = {}
    shared_config = None
    for epoch, path in zip(args.epochs, paths):
        allocator, channel, config, _ = common.load_transmitter(path, device)
        # One VAE, input dataset and physical channel must serve every epoch.
        settings = {
            "image_size": config.get("image_size", 512),
            "latent_channels": config.get("latent_channels", 4),
            "vae_scale": config.get("vae_scale", common.VAE_SCALE),
        }
        if args.data_path is None:
            settings["data_path"] = config["data_path"]
        if args.vae_path is None:
            settings["vae_model_id"] = config["vae_model_id"]
            settings["vae_model_revision"] = config["vae_model_revision"]
        if args.diffusion_model is None:
            settings["diffusion_model"] = config["vae_model_id"]
            settings["diffusion_revision"] = config["vae_model_revision"]
        if shared_config is not None and settings != shared_config:
            mismatches = [key for key in settings if settings[key] != shared_config[key]]
            raise ValueError(f"Epoch checkpoints have incompatible shared settings: {mismatches}")
        shared_config = settings
        transmitters[f"epoch_{epoch:03d}"] = {
            "epoch": epoch, "path": path, "allocator": allocator, "channel": channel, "config": config,
        }
    return transmitters


def restore_epochs(latent, snr, transmitters, sampler, channel_generator,
                   diffusion_generator, min_gain=0.0, max_timestep=None):
    """Run each learned transmitter with common h, AWGN and diffusion noise."""
    first = next(iter(transmitters.values()))
    with torch.no_grad():
        h = first["channel"].sample_channel_coefficient(latent, channel_generator)
        n0 = first["channel"]._noise_spectral_density(latent, snr)
        epsilon = torch.randn(latent.shape, device=latent.device, dtype=torch.float32,
                              generator=channel_generator)
    diffusion_state = diffusion_generator.get_state()
    restored, fallbacks, diagnostics = {}, {}, {}
    for name, transmitter in transmitters.items():
        with torch.no_grad():
            _, power, info = common.allocate_power_for_inference(
                latent, snr, transmitter["allocator"], transmitter["channel"],
                float(transmitter["config"].get("temperature", 1.0)), channel_coefficient=h,
                min_gain=min_gain, max_timestep=max_timestep,
            )
            raw = common.sample_raw_channel(info["channel_input"], power, n0, h, epsilon=epsilon)
            received = common.oracle_receive(raw, power, n0, h)
            fallbacks[name] = info.get("gain_floor_fallback", torch.zeros_like(h.real, dtype=torch.bool))
        diffusion_generator.set_state(diffusion_state)
        restored[name] = sampler(received["observation"], received["precision"],
                                 info["orthogonal_matrix"], diffusion_generator, verbose=False)
        diagnostics[name] = dict(sampler.last_diagnostics)
    return restored, h, fallbacks, diagnostics


def summarize(stats, names):
    summaries = {}
    for name, values in zip(names, stats.cpu().tolist()):
        row = dict(zip(common.STAT_NAMES, values))
        count = int(row["count"])
        if count == 0:
            raise ValueError("No images were evaluated")
        summaries[name] = {
            "num_samples": count,
            "top1_correct": int(row["top1_correct"]), "top5_correct": int(row["top5_correct"]),
            "top1_accuracy": row["top1_correct"] / count, "top5_accuracy": row["top5_correct"] / count,
            "mse": row["mse_sum"] / count, "psnr_db": row["psnr_sum"] / count,
            "lpips": row["lpips_sum"] / count,
        }
    return summaries


def evaluation_output_path(args, transmitters, snr_db, sample_count):
    first = next(iter(transmitters.values()))
    path = common.evaluation_output_path(args, first["path"], snr_db, sample_count)
    if args.output_path is None:
        epochs = "_".join(f"{entry['epoch']:03d}" for entry in transmitters.values())
        path = path.with_name(f"maxvit_epochs_{epochs}_{path.name.removeprefix('maxvit_')}")
    return path


def evaluate(args, device):
    args.max_timestep = common.resolve_max_timestep(args)
    transmitters = load_epochs(args, device)
    names = list(transmitters)
    config = next(iter(transmitters.values()))["config"]
    lpips_model = common.load_lpips(device, local_files_only=args.local_files_only)
    vae_kwargs = {"torch_dtype": torch.float32, "local_files_only": args.local_files_only}
    if args.vae_path is None:
        vae_kwargs.update(subfolder="vae", revision=config["vae_model_revision"])
    vae = common._freeze(common.AutoencoderKL.from_pretrained(
        args.vae_path or config["vae_model_id"], **vae_kwargs,
    ).to(device))
    vae_scale = float(config.get("vae_scale", common.VAE_SCALE))
    reverse_model = common.StaggeredReverseDiffusion(
        args.diffusion_model or config["vae_model_id"], device, torch.float32, args.local_files_only,
        revision=config["vae_model_revision"] if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    sampler = common.build_restoration_sampler(reverse_model, args, receiver_mode="oracle")
    task_model = common._freeze(common.construct_model("MaxViT-Large", num_classes=1000, pretrained=True).to(device))
    image_size = int(config.get("image_size", 512))
    root = Path(args.data_path or config["data_path"]).expanduser()
    validation_root = root / "val" if (root / "val").is_dir() else root
    dataset = common.datasets.ImageFolder(str(validation_root), transform=common.transforms.Compose([
        common.transforms.Resize(image_size), common.transforms.CenterCrop(image_size), common.transforms.ToTensor(),
    ]))
    indices = common.select_balanced_indices(dataset, args.max_samples, args.seed)
    sample_count = len(indices)
    class_counts = [0] * len(dataset.classes)
    for index in indices:
        class_counts[dataset.targets[index]] += 1
    loader = DataLoader(Subset(common.IndexedDataset(dataset), indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    result = {
        "checkpoints": {name: {"epoch": item["epoch"], "path": str(item["path"]),
                               "temperature": float(item["config"].get("temperature", 1.0))}
                        for name, item in transmitters.items()},
        "model": "maxvit_large_tf_224.in1k", "dataset": "ImageNet",
        "reconstruction": args.restoration_method, "receiver_mode": "oracle",
        "channel_model": "rayleigh_block_fading", "perfect_csi": True,
        "fading_granularity": "one_complex_h_per_image_shared_by_all_blocks",
        "fading_distribution": "(randn(B)+1j*randn(B))/sqrt(2)",
        "mean_real_component_power": 1.0, "noise_real_component_variance": "N0",
        "allocated_uses_Q": True, "same_vae_sample_for_strategies": True,
        "same_h_and_awgn_for_strategies": True, "same_diffusion_noise_for_strategies": True,
        "vae_latent_method": "sample", "lpips_network": common.LPIPS_NETWORK,
        "quality_reference": "original_full_resolution_RGB",
        "accuracy_preprocessing": "bicubic_resize_235_center_crop_224_ImageNet_normalization",
        "diffusion_steps": args.diffusion_steps, "guidance_scale": args.guidance_scale,
        "guidance_max_rms": args.guidance_max_rms, "posterior_eta": args.posterior_eta,
        "ddrm_eta": args.ddrm_eta, "ddrm_eta_b": args.ddrm_eta_b,
        "ddrm_initialization_policy": getattr(sampler, "initialization_policy", None),
        "seed": args.seed, "batch_size": args.batch_size,
        "min_gain": args.min_gain, "max_timestep": args.max_timestep,
        "gain_floor_infeasible_policy": "uniform_gain_same_frame_energy",
        "num_samples_requested": args.max_samples, "num_samples": sample_count, "evaluations": {},
        "sample_selection": {
            "method": "class_balanced_without_replacement", "seed": args.seed,
            "dataset_root": str(validation_root.resolve()), "dataset_size": len(dataset),
            "num_classes": len(dataset.classes), "class_counts": dict(zip(dataset.classes, class_counts)),
            "indices": indices,
            "relative_paths": [str(Path(dataset.samples[index][0]).relative_to(validation_root)) for index in indices],
            "same_images_for_strategies": True, "same_images_for_snrs": True,
        },
    }
    for name, item in transmitters.items():
        print(f"{name} checkpoint: {item['path']}", flush=True)
    print(f"클래스 균등 추출: {len(dataset.classes)}개 클래스, 클래스당 "
          f"{min(class_counts)}~{max(class_counts)}장, seed={args.seed}입니다.", flush=True)
    print(f"ImageNet {sample_count}장 / allocated(Q)+{args.restoration_method.upper()} / Rayleigh / oracle", flush=True)
    if args.max_timestep is not None:
        print(f"송신 gain 하한에 max_timestep={args.max_timestep} 기준을 적용합니다.", flush=True)
    for snr_db in args.snr_db:
        stats = torch.zeros(len(names), len(common.STAT_NAMES), device=device, dtype=torch.float64)
        fading_stats = torch.zeros(3, device=device, dtype=torch.float64)
        fallback_counts = dict.fromkeys(names, 0)
        initialization_clips = {name: {"images": 0, "blocks": 0} for name in names}
        for batch_index, (images, labels, sample_indices) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Exactly the batch seeding used by evaluate_power_maxvit.
            batch_seed = args.seed + 4 * int(sample_indices[0])
            vae_generator = torch.Generator(device=device).manual_seed(batch_seed)
            channel_generator = torch.Generator(device=device).manual_seed(batch_seed + 1)
            diffusion_generator = torch.Generator(device=device).manual_seed(batch_seed + 2)
            with torch.no_grad():
                latent = common.encode_latents(images, vae, vae_generator)
                snr = torch.full((images.shape[0],), snr_db, device=device)
            restored, h, fallbacks, diagnostics = restore_epochs(
                latent, snr, transmitters, sampler, channel_generator, diffusion_generator,
                min_gain=args.min_gain, max_timestep=args.max_timestep,
            )
            with torch.no_grad():
                for index, name in enumerate(names):
                    reconstructed = common._decode_latent(vae, restored[name], vae_scale)
                    common.accumulate(stats[index], reconstructed, images, labels, task_model, lpips_model)
                    fallback_counts[name] += int(fallbacks[name].sum().item())
                    initialization_clips[name]["images"] += diagnostics[name].get("initialization_clipped_images", 0)
                    initialization_clips[name]["blocks"] += diagnostics[name].get("initialization_clipped_block_count", 0)
                h_power = h.abs().double().square()
                fading_stats += torch.stack((h_power.sum(), h_power.square().sum(),
                                              h_power.new_tensor(h_power.numel())))
            if (batch_index + 1) % args.log_interval == 0 or batch_index == 0:
                print(f"SNR={snr_db:g} dB, {int(stats[0, 0].item())}/{sample_count}장 완료", flush=True)
                common.print_summary(snr_db, summarize(stats, names), stage="중간 누적")
                if args.max_timestep is not None:
                    print(f"timestep 하한 전력 부족으로 균일 gain 적용: {fallback_counts}", flush=True)
                if args.restoration_method == "ddrm":
                    print(f"DDRM 초기화 추가 noise=0 적용 누적: {initialization_clips}", flush=True)
            del latent, restored, reconstructed
        summaries = summarize(stats, names)
        mean_h_power = (fading_stats[0] / fading_stats[2]).item()
        std_h_power = max((fading_stats[1] / fading_stats[2]).item() - mean_h_power**2, 0.0)**0.5
        result["evaluations"][f"{snr_db:g}"] = {
            "snr_db": snr_db, "num_samples": sample_count,
            "mean_h_power": mean_h_power, "std_h_power": std_h_power, "strategies": summaries,
            "gain_floor_infeasible_images": fallback_counts,
            "ddrm_initialization_clipped": initialization_clips,
        }
        common.print_summary(snr_db, summaries)
        output_path = evaluation_output_path(args, transmitters, snr_db, sample_count)
        snr_result = {**result, "evaluations": {f"{snr_db:g}": result["evaluations"][f"{snr_db:g}"]}}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snr_result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"평가 결과 저장: {output_path}", flush=True)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing epoch_010.pth and epoch_015.pth")
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 15], help="Completed checkpoint epochs (default: 10 15)")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0.0])
    parser.add_argument("--min_gain", type=float, default=0.0)
    parser.add_argument("--max_timestep", type=int, choices=range(1000), metavar="0..999", default=None,
                        help="Allocation noise ceiling (DDRM default: 999; DPS: opt-in)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Class-balanced count (default: 1000, one per ImageNet class)")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    common.add_restoration_arguments(parser)
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--guidance_max_rms", type=float, default=1.0)
    parser.add_argument("--posterior_eta", type=float, default=0.0)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if len(args.epochs) < 2 or len(set(args.epochs)) != len(args.epochs) or min(args.epochs) < 1:
        parser.error("Specify at least two distinct positive epochs")
    if min(args.batch_size, args.log_interval, args.max_samples) < 1 or args.num_workers < 0:
        parser.error("batch_size/log_interval/max_samples must be positive and num_workers must be non-negative")
    if not 1 <= args.diffusion_steps <= 1000 or not all(math.isfinite(x) for x in args.snr_db):
        parser.error("diffusion_steps must be in [1,1000] and all SNR values must be finite")
    try:
        common.build_restoration_sampler(None, args, receiver_mode="oracle")
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; run evaluation on an allocated GPU")
        torch.cuda.set_device(device)
    evaluate(args, device)


if __name__ == "__main__":
    main()
