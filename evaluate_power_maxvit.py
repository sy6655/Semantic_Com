"""ImageNet evaluation: VAE, uniform/no-Q, and learned Q/allocation restoration.

All three reconstructions use the same VAE posterior sample. The two radio
paths share a fresh complex Rayleigh h per image, channel noise, and diffusion
noise. Select DPS (default) or DDRM with --restoration_method.
Allocated reception can use --receiver_mode gain_estimator with
--gain_estimator_checkpoint; uniform reception remains oracle.
Accuracy is measured by the pretrained RGB MaxViT-Large.
By default, evaluate a seeded class-balanced subset of 5,000 images
(five per class for the 1,000-class ImageNet validation set).
"""

import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from power_utils import maxvit_input as _maxvit_input
from inference import StaggeredReverseDiffusion
from inference_oracle_gain import oracle_receive
from inference_power import _decode_latent, _freeze, allocate_power_for_inference
from models.models import construct_model
from orthogonal_power import (
    gain_estimator_receive, load_gain_estimator, patchify,
    prepare_gain_estimator_input, sample_raw_channel,
)
from posterior_sampling import add_restoration_arguments, build_restoration_sampler, resolve_max_timestep
from reconstruction_metrics import LPIPS_NETWORK, image_quality_batch, load_lpips
from train_gain_estimator import load_transmitter
from train_power_allocator import VAE_SCALE, encode_latents


STRATEGIES = ("vae", "uniform", "allocated")
STAT_NAMES = ("count", "top1_correct", "top5_correct", "mse_sum", "psnr_sum", "lpips_sum")
DEFAULT_MAX_SAMPLES = 5000


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def resolve_checkpoint(path):
    path = Path(path).expanduser()
    return path / "latest.pth" if path.is_dir() else path


def resolve_receiver_mode(args):
    estimator_path = getattr(args, "gain_estimator_checkpoint", None)
    mode = getattr(args, "receiver_mode", None) or ("gain_estimator" if estimator_path else "oracle")
    if mode not in ("oracle", "gain_estimator"):
        raise ValueError("receiver_mode must be oracle or gain_estimator")
    if mode == "gain_estimator" and estimator_path is None:
        raise ValueError("--gain_estimator_checkpoint is required for --receiver_mode gain_estimator")
    return mode


def select_balanced_indices(dataset, max_samples=DEFAULT_MAX_SAMPLES, seed=0):
    """Sample without replacement using a local RNG, then keep dataset order.

    Every class receives the same quota; when the requested count is not
    divisible by the class count, seeded random classes get one extra image.
    None also selects the default subset. Use --max_samples 50000 for full
    ImageNet validation evaluation.
    """
    limit = DEFAULT_MAX_SAMPLES if max_samples is None else max_samples
    if limit < 1:
        raise ValueError("max_samples must be positive")
    if not dataset.classes or len(dataset.targets) != len(dataset):
        raise ValueError("Class-balanced evaluation requires classes and one target per image")
    by_class = [[] for _ in dataset.classes]
    for index, target in enumerate(dataset.targets):
        by_class[target].append(index)
    sample_count = min(limit, len(dataset))
    if sample_count == 0:
        raise ValueError("No evaluation images were found")
    generator = torch.Generator().manual_seed(seed)
    per_class, remainder = divmod(sample_count, len(by_class))
    quotas = [per_class] * len(by_class)
    if remainder:
        for class_index in torch.randperm(len(by_class), generator=generator)[:remainder].tolist():
            quotas[class_index] += 1
    selected = []
    for class_name, candidates, quota in zip(dataset.classes, by_class, quotas):
        if len(candidates) < quota:
            raise ValueError(
                f"Class {class_name!r} has {len(candidates)} images, but balanced evaluation "
                f"requires {quota}; reduce --max_samples or check the validation dataset"
            )
        positions = torch.randperm(len(candidates), generator=generator)[:quota].tolist()
        selected.extend(candidates[position] for position in positions)
    return sorted(selected)


def evaluation_output_path(args, checkpoint_path, snr_db, sample_count=None):
    if args.output_path is not None:
        path = Path(args.output_path).expanduser()
        if len(args.snr_db) > 1:
            path = path.with_name(f"{path.stem}_snr_{snr_db:g}{path.suffix or '.json'}")
        return path
    method = getattr(args, "restoration_method", "dps")
    sampling_suffix = (f"_guide_{args.guidance_scale:g}" if method == "dps" else
                       f"_eta_{args.ddrm_eta:g}_etab_{args.ddrm_eta_b:g}")
    if sample_count is None:
        sample_count = getattr(args, "max_samples", None) or DEFAULT_MAX_SAMPLES
    max_timestep = resolve_max_timestep(args)
    return checkpoint_path.parent / "evaluation" / (
        f"maxvit_{method}_rayleigh_snr_{snr_db:g}_steps_{args.diffusion_steps}"
        f"{sampling_suffix}_seed_{args.seed}"
        f"_balanced_n{sample_count}"
        + (f"_mingain_{args.min_gain:g}" if getattr(args, "min_gain", 0.0) > 0 else "")
        + (f"_maxt_{max_timestep}" if max_timestep is not None else "")
        + ("_receiver_gain_estimator_uniform_oracle" if resolve_receiver_mode(args) == "gain_estimator" else "")
        + ".json"
    )


def restore_transmissions(latent, snr, allocator, channel, sampler,
                          channel_generator, diffusion_generator, temperature=1.0,
                          min_gain=0.0, max_timestep=None, diagnostics=None,
                          receiver_mode="oracle", gain_estimator=None, uniform_sampler=None):
    """Return both restorations with common h/noise and an unrotated baseline."""
    if receiver_mode not in ("oracle", "gain_estimator"):
        raise ValueError("receiver_mode must be oracle or gain_estimator")
    if receiver_mode == "gain_estimator" and gain_estimator is None:
        raise ValueError("Allocated gain_estimator reception requires a loaded estimator")
    with torch.no_grad():
        h = channel.sample_channel_coefficient(latent, channel_generator)
        _, allocated_power, info = allocate_power_for_inference(
            latent, snr, allocator, channel, temperature, channel_coefficient=h,
            min_gain=min_gain,
            max_timestep=max_timestep,
        )
        spatial_energy = patchify(latent).square().sum(-1)
        uniform_power = (
            info["target_frame_energy"] / spatial_energy.sum(1, keepdim=True)
        ).expand_as(allocated_power)
        n0 = channel._noise_spectral_density(latent, snr)
        epsilon = torch.randn(latent.shape, device=latent.device,
                              dtype=torch.float32, generator=channel_generator)
        q = info["orthogonal_matrix"]
        identity = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
    diffusion_state = diffusion_generator.get_state()
    restored = {}
    for name, transmitted, power, matrix in (
        ("uniform", latent, uniform_power, identity),
        ("allocated", info["channel_input"], allocated_power, q),
    ):
        with torch.no_grad():
            raw = sample_raw_channel(transmitted, power, n0, h, epsilon=epsilon)
            if name == "allocated" and receiver_mode == "gain_estimator":
                estimator_input = prepare_gain_estimator_input(raw, n0, h)
                log_gain = gain_estimator(estimator_input["channel_equalized_latent"],
                                          estimator_input["log_noise_variance"], estimator_input["log_power"])
                received = gain_estimator_receive(raw, log_gain, n0, h)
                observation = received["equalized_latent"]
            else:
                received = oracle_receive(raw, power, n0, h)
                observation = received["observation"]
        strategy_sampler = uniform_sampler if name == "uniform" and uniform_sampler is not None else sampler
        diffusion_generator.set_state(diffusion_state)
        restored[name] = strategy_sampler(observation, received["precision"], matrix,
                                          diffusion_generator, verbose=False)
        if diagnostics is not None:
            diagnostics[name] = strategy_sampler.last_diagnostics
    fallback = info.get("gain_floor_fallback", torch.zeros_like(h.real, dtype=torch.bool))
    return restored, h, fallback


@torch.no_grad()
def accumulate(stats, reconstructed, reference, labels, task_model, lpips_model):
    # Shared MaxViT preprocessing: RGB -> bicubic resize 235 -> crop 224 -> ImageNet normalization.
    task_dtype = next(task_model.parameters()).dtype
    logits = task_model(_maxvit_input(reconstructed.to(dtype=task_dtype)))
    predictions = logits.topk(5, dim=1).indices
    quality = image_quality_batch(reconstructed, reference, lpips_model)
    stats += stats.new_tensor([
        labels.numel(),
        (predictions[:, 0] == labels).sum().item(),
        predictions.eq(labels[:, None]).any(1).sum().item(),
        quality["mse"].sum().item(), quality["psnr_db"].sum().item(), quality["lpips"].sum().item(),
    ])


def summarize(stats):
    results = {}
    for name, values in zip(STRATEGIES, stats.cpu().tolist()):
        row = dict(zip(STAT_NAMES, values))
        count = int(row["count"])
        if count == 0:
            raise ValueError("No images were evaluated")
        results[name] = {
            "num_samples": count,
            "top1_correct": int(row["top1_correct"]), "top5_correct": int(row["top5_correct"]),
            "top1_accuracy": row["top1_correct"] / count, "top5_accuracy": row["top5_correct"] / count,
            "mse": row["mse_sum"] / count, "psnr_db": row["psnr_sum"] / count,
            "lpips": row["lpips_sum"] / count,
        }
    return results


def print_summary(snr_db, results, stage="최종"):
    print(f"\n[{stage}] Rayleigh / perfect CSI / SNR={snr_db:g} dB", flush=True)
    print(f"{'strategy':<12} {'Top-1 (%)':>12} {'Top-5 (%)':>12} {'PSNR (dB)':>12} {'LPIPS':>12}", flush=True)
    for name, row in results.items():
        print(f"{name:<12} {100*row['top1_accuracy']:12.3f} {100*row['top5_accuracy']:12.3f} "
              f"{row['psnr_db']:12.3f} {row['lpips']:12.5f}", flush=True)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; run evaluation on an allocated GPU")
        torch.cuda.set_device(device)
    return evaluate(args, device)


def evaluate(args, device):
    method = getattr(args, "restoration_method", "dps")
    args.max_timestep = resolve_max_timestep(args)
    receiver_mode = resolve_receiver_mode(args)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    allocator, channel, config, _ = load_transmitter(checkpoint_path, device)
    gain_estimator, estimator_config, estimator_path = None, None, None
    if receiver_mode == "gain_estimator":
        estimator_path = resolve_checkpoint(args.gain_estimator_checkpoint)
        gain_estimator, estimator_config = load_gain_estimator(
            estimator_path, allocator, device, check_allocator_fingerprint=False,
        )
    lpips_model = load_lpips(device, local_files_only=args.local_files_only)
    vae_kwargs = {"torch_dtype": torch.float32, "local_files_only": args.local_files_only}
    if args.vae_path is None:
        vae_kwargs.update(subfolder="vae", revision=config["vae_model_revision"])
    vae = _freeze(AutoencoderKL.from_pretrained(
        args.vae_path or config["vae_model_id"], **vae_kwargs
    ).to(device))
    vae_scale = float(config.get("vae_scale", VAE_SCALE))
    reverse_model = StaggeredReverseDiffusion(
        args.diffusion_model or config["vae_model_id"], device, torch.float32, args.local_files_only,
        revision=config["vae_model_revision"] if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    sampler = build_restoration_sampler(reverse_model, args, receiver_mode=receiver_mode)
    uniform_sampler = (sampler if receiver_mode == "oracle" else
                       build_restoration_sampler(reverse_model, args, receiver_mode="oracle"))
    task_model = _freeze(construct_model("MaxViT-Large", num_classes=1000, pretrained=True).to(device))
    image_size = int(config.get("image_size", 512))
    root = Path(args.data_path or config["data_path"]).expanduser()
    validation_root = root / "val" if (root / "val").is_dir() else root
    dataset = datasets.ImageFolder(str(validation_root), transform=transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
    ]))
    # Select once, outside the SNR and reconstruction loops. All strategies
    # therefore see exactly the same original dataset indices in the same order.
    indices = select_balanced_indices(dataset, args.max_samples, args.seed)
    sample_count = len(indices)
    class_counts = [0] * len(dataset.classes)
    for index in indices:
        class_counts[dataset.targets[index]] += 1
    loader = DataLoader(Subset(IndexedDataset(dataset), indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    result = {
        "checkpoint": str(checkpoint_path), "model": "maxvit_large_tf_224.in1k", "dataset": "ImageNet",
        "reconstruction": method, "receiver_mode": receiver_mode, "uniform_receiver_mode": "oracle",
        "strategy_receiver_modes": {"uniform": "oracle", "allocated": receiver_mode},
        "uses_gain_estimator": gain_estimator is not None,
        "gain_estimator_checkpoint": str(estimator_path) if estimator_path is not None else None,
        "gain_estimator_config": estimator_config,
        "channel_model": "rayleigh_block_fading",
        "perfect_csi": True, "fading_granularity": "one_complex_h_per_image_shared_by_all_blocks",
        "fading_distribution": "(randn(B)+1j*randn(B))/sqrt(2)",
        "mean_real_component_power": 1.0, "noise_real_component_variance": "N0",
        "uniform_uses_Q": False, "allocated_uses_Q": True,
        "same_vae_sample_for_strategies": True, "same_h_and_awgn_for_strategies": True,
        "same_diffusion_noise_for_strategies": True, "vae_latent_method": "sample",
        "lpips_network": LPIPS_NETWORK, "quality_reference": "original_full_resolution_RGB",
        "accuracy_preprocessing": "bicubic_resize_235_center_crop_224_ImageNet_normalization",
        "diffusion_steps": args.diffusion_steps, "guidance_scale": args.guidance_scale,
        "guidance_max_rms": args.guidance_max_rms, "posterior_eta": args.posterior_eta,
        "ddrm_eta": getattr(args, "ddrm_eta", 0.85), "ddrm_eta_b": getattr(args, "ddrm_eta_b", 1.0),
        "ddrm_initialization_policy": getattr(sampler, "initialization_policy", None),
        "ddrm_initialization_policies": {
            "uniform": getattr(uniform_sampler, "initialization_policy", None),
            "allocated": getattr(sampler, "initialization_policy", None),
        },
        "seed": args.seed, "batch_size": args.batch_size,
        "min_gain": getattr(args, "min_gain", 0.0),
        "max_timestep": getattr(args, "max_timestep", None),
        "gain_floor_infeasible_policy": "uniform_gain_same_frame_energy",
        "num_samples_requested": sample_count, "evaluations": {},
        "sample_selection": {
            "method": "class_balanced_without_replacement", "seed": args.seed,
            "dataset_root": str(validation_root.resolve()), "dataset_size": len(dataset),
            "num_classes": len(dataset.classes),
            "class_counts": dict(zip(dataset.classes, class_counts)),
            "indices": indices,
            "relative_paths": [str(Path(dataset.samples[index][0]).relative_to(validation_root))
                               for index in indices],
            "same_images_for_strategies": True, "same_images_for_snrs": True,
        },
    }
    print(f"평가 checkpoint: {checkpoint_path}", flush=True)
    print(f"수신 모드: uniform=oracle, allocated={receiver_mode}", flush=True)
    if estimator_path is not None:
        print(f"Gain estimator checkpoint: {estimator_path}", flush=True)
    if args.max_timestep is not None:
        print(f"송신 gain 하한에 max_timestep={args.max_timestep} 기준을 적용합니다.", flush=True)
    print(f"클래스 균등 추출: {len(dataset.classes)}개 클래스, 클래스당 "
          f"{min(class_counts)}~{max(class_counts)}장, seed={args.seed}입니다.", flush=True)
    print(f"ImageNet {sample_count}장, VAE / uniform(no Q)+{method.upper()} / allocated(Q)+{method.upper()}", flush=True)
    for snr_db in args.snr_db:
        stats = torch.zeros(len(STRATEGIES), len(STAT_NAMES), device=device, dtype=torch.float64)
        fading_stats = torch.zeros(4, device=device, dtype=torch.float64)
        initialization_clips = {name: {"images": 0, "blocks": 0} for name in STRATEGIES[1:]}
        for batch_index, (images, labels, sample_indices) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Independent VAE/channel/DPS streams; paired across transmission strategies.
            batch_seed = args.seed + 4 * int(sample_indices[0])
            vae_generator = torch.Generator(device=device).manual_seed(batch_seed)
            channel_generator = torch.Generator(device=device).manual_seed(batch_seed + 1)
            diffusion_generator = torch.Generator(device=device).manual_seed(batch_seed + 2)
            with torch.no_grad():
                latent = encode_latents(images, vae, vae_generator)
                vae_images = _decode_latent(vae, latent, vae_scale)
                accumulate(stats[0], vae_images, images, labels, task_model, lpips_model)
                snr = torch.full((images.shape[0],), snr_db, device=device)
            batch_diagnostics = {}
            restored, h, fallback = restore_transmissions(latent, snr, allocator, channel, sampler,
                                                channel_generator, diffusion_generator,
                                                float(config.get("temperature", 1.0)),
                                                min_gain=getattr(args, "min_gain", 0.0),
                                                max_timestep=getattr(args, "max_timestep", None),
                                                diagnostics=batch_diagnostics,
                                                receiver_mode=receiver_mode, gain_estimator=gain_estimator,
                                                uniform_sampler=uniform_sampler)
            for name, diagnostics in batch_diagnostics.items():
                initialization_clips[name]["images"] += diagnostics.get("initialization_clipped_images", 0)
                initialization_clips[name]["blocks"] += diagnostics.get("initialization_clipped_block_count", 0)
            with torch.no_grad():
                for index, name in enumerate(STRATEGIES[1:], 1):
                    reconstructed = _decode_latent(vae, restored[name], vae_scale)
                    accumulate(stats[index], reconstructed, images, labels, task_model, lpips_model)
                h_power = h.abs().double().square()
                fading_stats += torch.stack((h_power.sum(), h_power.square().sum(),
                                              h_power.new_tensor(h_power.numel()),
                                              fallback.sum().to(dtype=torch.float64)))
            if (batch_index + 1) % args.log_interval == 0 or batch_index == 0:
                print(f"SNR={snr_db:g} dB, {int(stats[0,0].item())}/{sample_count}장 완료", flush=True)
                print_summary(snr_db, summarize(stats), stage="중간 누적")
                if getattr(args, "max_timestep", None) is not None:
                    print(f"timestep 하한 전력 부족으로 균일 gain 적용: {int(fading_stats[3].item())}장", flush=True)
                if method == "ddrm":
                    counts = ", ".join(f"{name} {values['images']}장/{values['blocks']}블록"
                                       for name, values in initialization_clips.items())
                    print(f"DDRM 초기화 추가 noise=0 적용 누적: {counts}", flush=True)
            del latent, restored, vae_images, reconstructed
        summaries = summarize(stats)
        mean_h_power = (fading_stats[0] / fading_stats[2]).item()
        std_h_power = max((fading_stats[1] / fading_stats[2]).item() - mean_h_power**2, 0.0)**0.5
        result["evaluations"][f"{snr_db:g}"] = {
            "snr_db": snr_db, "num_samples": summaries["vae"]["num_samples"],
            "mean_h_power": mean_h_power, "std_h_power": std_h_power, "strategies": summaries,
            "gain_floor_infeasible_images": int(fading_stats[3].item()),
            "ddrm_initialization_clipped": initialization_clips,
        }
        print_summary(snr_db, summaries)
        output_path = evaluation_output_path(args, checkpoint_path, snr_db, sample_count)
        snr_result = {**result, "evaluations": {f"{snr_db:g}": result["evaluations"][f"{snr_db:g}"]}}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(snr_result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"평가 결과 저장: {output_path}", flush=True)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receiver_mode", choices=("oracle", "gain_estimator"), default=None,
                        help="Allocated receiver: gain_estimator when supplied, otherwise oracle; uniform stays oracle")
    parser.add_argument("--gain_estimator_checkpoint", default=None,
                        help="Gain estimator checkpoint or directory containing latest.pth")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0.0])
    parser.add_argument("--min_gain", type=float, default=0.0,
                        help="Minimum allocated amplitude a_b during inference; 0 disables the floor")
    parser.add_argument("--max_timestep", type=int, choices=range(1000), metavar="0..999", default=None,
                        help="Allocation observation-noise ceiling (DDRM default: 999; DPS: opt-in); "
                             "does not change diffusion sampling timesteps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Class-balanced image count (default: 5000, five per ImageNet class); "
                             "use 50000 for the full ImageNet validation set")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    add_restoration_arguments(parser)
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
    if args.batch_size < 1 or args.num_workers < 0 or args.log_interval < 1:
        parser.error("batch_size/log_interval must be positive and num_workers must be non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max_samples must be positive")
    if not 1 <= args.diffusion_steps <= 1000 or not all(math.isfinite(x) for x in args.snr_db):
        parser.error("diffusion_steps must be in [1,1000] and all SNR values must be finite")
    try:
        receiver_mode = resolve_receiver_mode(args)
        build_restoration_sampler(None, args, receiver_mode=receiver_mode)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()
