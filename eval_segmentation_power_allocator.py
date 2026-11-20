"""Evaluate diffusion-restored transmissions with pretrained Mask2Former."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from datasets.data_processing import CityscapesSemanticDataset
from inference import StaggeredReverseDiffusion
from inference_power import ActualPowerTransport, _match_diffusion_start
from models.models import construct_model
from train_power_allocator import PowerAllocator, encode_latents
from train_segmentation_power_allocator import (
    NUM_CLASSES,
    AnalogAWGNChannel,
    allocate_power_with_timestep_floor,
    confusion_matrix,
    segmentation_metrics,
)


def uniform_group_power(
    latent: torch.Tensor,
    channel: AnalogAWGNChannel,
) -> torch.Tensor:
    """Use one gain per image under the same payload-only frame budget."""
    frame_channel_uses = latent[0].numel() // 2
    unscaled_frame_energy = (
        latent / channel.kappa_m
    ).square().flatten(1).sum(dim=1)
    gain_power = frame_channel_uses / unscaled_frame_energy
    return gain_power[:, None].expand(
        -1,
        channel.group_rows * channel.group_cols,
    )


def update_confusion(
    matrix: torch.Tensor,
    evaluator: torch.nn.Module,
    vae: AutoencoderKL,
    latent: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    decoded = vae.decode(
        latent / float(vae.config.scaling_factor)
    ).sample
    reconstructed_rgb = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
    height, width = labels.shape[-2:]
    predictions = evaluator(
        [
            {
                "image": image.float() * 255.0,
                "height": height,
                "width": width,
            }
            for image in reconstructed_rgb
        ]
    )
    for prediction, label in zip(predictions, labels):
        scores = prediction["sem_seg"]
        if scores.shape != (NUM_CLASSES, *label.shape):
            raise ValueError(
                "Mask2Former output must have shape "
                f"({NUM_CLASSES}, {height}, {width}), got "
                f"{tuple(scores.shape)}"
            )
        matrix += confusion_matrix(
            scores.unsqueeze(0),
            label.to(scores.device).unsqueeze(0),
        ).cpu()


def save_result(
    output_path: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    snr_setting: float,
    num_samples: int,
    frame_snr_sum: float,
    matrices: dict[str, torch.Tensor],
    latent_snr_sums: dict[str, float],
    symbol_power_sums: dict[str, float],
    unsupported_patch_counts: dict[str, int],
    diffusion_model: str,
    diffusion_steps: int,
) -> dict:
    strategies = {}
    for name, matrix in matrices.items():
        pixel_accuracy, miou = segmentation_metrics(matrix)
        strategy_result = {
            "pixel_accuracy": pixel_accuracy,
            "miou": miou,
        }
        if name in latent_snr_sums:
            strategy_result["mean_latent_snr_db"] = (
                latent_snr_sums[name] / num_samples
            )
            strategy_result["mean_symbol_power"] = (
                symbol_power_sums[name] / num_samples
            )
            strategy_result["diffusion_unsupported_patch_count"] = (
                unsupported_patch_counts[name]
            )
        strategies[name] = strategy_result

    primary_name = "allocated" if "allocated" in strategies else next(iter(strategies))
    result = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]) + 1,
        "evaluator": "Mask2Former (pretrained Cityscapes)",
        "num_samples": num_samples,
        "snr_db": snr_setting,
        "mean_frame_snr_db": frame_snr_sum / num_samples,
        "receiver_mode": "oracle",
        "pilots_per_group": 0,
        "diffusion_model": diffusion_model,
        "diffusion_steps": diffusion_steps,
        "same_channel_noise_for_allocated_and_uniform": (
            "allocated" in strategies and "uniform" in strategies
        ),
        "same_diffusion_noise_for_allocated_and_uniform": (
            "allocated" in strategies and "uniform" in strategies
        ),
        "uncompressed_definition": "VAE encode/decode without AWGN",
        "primary_strategy": primary_name,
        "pixel_accuracy": strategies[primary_name]["pixel_accuracy"],
        "miou": strategies[primary_name]["miou"],
        "strategies": strategies,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def legacy_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--snr_db", type=float, required=True)
    parser.add_argument("--diffusion_steps", type=int, default=25)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--strategy",
        choices=("all", "allocated", "uniform", "uncompressed"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.batch <= 0 or args.num_workers < 0:
        parser.error("--batch must be positive and --num_workers non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    if args.log_interval <= 0:
        parser.error("--log_interval must be positive")
    if not 1 <= args.diffusion_steps <= 1000:
        parser.error("--diffusion_steps must be between 1 and 1000")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run on a GPU host or use --device cpu")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "latest.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (
        config.get("receiver_mode") != "oracle"
        or int(config.get("pilots_per_group", -1)) != 0
        or config.get("transmission_mode") != "analog"
    ):
        parser.error("checkpoint must use pilot-free oracle analog transmission")
    selected_strategies = (
        ("allocated", "uniform", "uncompressed")
        if args.strategy == "all"
        else (args.strategy,)
    )
    run_diffusion = any(
        name != "uncompressed" for name in selected_strategies
    )
    diffusion_model = args.diffusion_model or config["vae_model_id"]
    output_path = Path(
        args.output_path
        or checkpoint_path.parent
        / (
            f"mask2former_diffusion_validation_snr_{args.snr_db:g}"
            f"{'_' + args.strategy if args.strategy != 'all' else ''}.json"
        )
    )

    dataset = CityscapesSemanticDataset(
        args.data_path or config["data_path"],
        "val",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = AutoencoderKL.from_pretrained(
        config["vae_model_id"],
        subfolder="vae",
        revision=config["vae_model_revision"],
        torch_dtype=torch.float32,
    ).to(device).eval()
    latent_channels = int(vae.config.latent_channels)
    allocator = PowerAllocator(
        latent_channels,
        int(config["num_groups"]),
        int(config["hidden_dim"]),
        sampling=False,
    ).to(device).eval()
    allocator.load_state_dict(checkpoint["allocator"])
    # The trained latent-input SegFormer is intentionally not used here.
    evaluator = construct_model("Mask2Former").to(device).eval()

    channel = AnalogAWGNChannel(
        allocator.group_rows,
        allocator.group_cols,
        args.snr_db,
        config.get("channel_gain_abs", [1.0]),
        0,
    ).to(device)
    transport = ActualPowerTransport(channel, "oracle")
    reverse_model = None
    ddim_timesteps = None
    if run_diffusion:
        reverse_model = StaggeredReverseDiffusion(
            diffusion_model,
            device,
            torch.float16 if args.amp and device.type == "cuda" else torch.float32,
            args.local_files_only,
            revision=(
                config.get("vae_model_revision")
                if args.diffusion_model is None
                else None
            ),
            num_inference_steps=args.diffusion_steps,
        )
        ddim_timesteps = reverse_model.inference_timesteps(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    amp_enabled = args.amp and device.type == "cuda"

    matrices = {
        name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        for name in selected_strategies
    }
    latent_snr_sums = {
        name: 0.0 for name in selected_strategies if name != "uncompressed"
    }
    symbol_power_sums = dict.fromkeys(latent_snr_sums, 0.0)
    unsupported_patch_counts = dict.fromkeys(latent_snr_sums, 0)
    num_samples = 0
    frame_snr_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            remaining = (
                None
                if args.max_samples is None
                else args.max_samples - num_samples
            )
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and images.shape[0] > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            previous_num_samples = num_samples
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.shape[0]
            snr_db = torch.full((batch_size,), args.snr_db, device=device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                latent = encode_latents(images, vae)
                if "allocated" in matrices:
                    _, allocated_power, _ = (
                        allocate_power_with_timestep_floor(
                            latent,
                            snr_db,
                            allocator,
                            channel,
                            float(config.get("temperature", 1.0)),
                            int(config.get("max_timestep", 999)),
                            allocator_snr_scale_db=float(
                                config.get("allocator_snr_scale_db", 1.0)
                            ),
                        )
                    )
                if "uniform" in matrices:
                    uniform_power = uniform_group_power(latent, channel)

                if run_diffusion:
                    matching_generator = torch.Generator(device=device)
                    matching_generator.manual_seed(
                        args.seed + previous_num_samples + 1
                    )
                    matching_epsilon = torch.randn(
                        latent.shape,
                        device=device,
                        dtype=latent.dtype,
                        generator=matching_generator,
                    )
                    diffusion_generator = torch.Generator(device=device)
                    diffusion_generator.manual_seed(
                        args.seed + previous_num_samples + 2
                    )
                    diffusion_state = diffusion_generator.get_state()

                # Reset the generator before each transmitted strategy so
                # allocated and uniform see exactly the same AWGN samples.
                channel_state = generator.get_state()
                for name in selected_strategies:
                    if name == "uncompressed":
                        evaluated_latent = latent
                    else:
                        generator.set_state(channel_state)
                        group_power = (
                            allocated_power
                            if name == "allocated"
                            else uniform_power
                        )
                        evaluated_latent, channel_info, transport_info = (
                            transport(
                                latent,
                                group_power,
                                snr_db,
                                generator,
                            )
                        )
                        latent_snr_sums[name] += (
                            (10.0 * torch.log10(channel_info["snr"]))
                            .mean()
                            .item()
                            * batch_size
                        )
                        frame_channel_uses = latent[0].numel() // 2
                        symbol_power_sums[name] += (
                            transport_info["transmit_frame_energy"]
                            .mean()
                            .item()
                            / frame_channel_uses
                            * batch_size
                        )
                        assert reverse_model is not None
                        assert ddim_timesteps is not None
                        matched_latent, reverse_context, matching_info = (
                            _match_diffusion_start(
                                evaluated_latent,
                                channel_info["noise_variance"],
                                channel,
                                matching_epsilon,
                                ddim_timesteps,
                            )
                        )
                        unsupported_patch_counts[name] += matching_info[
                            "unsupported_patch_count"
                        ]
                        diffusion_generator.set_state(diffusion_state)
                        evaluated_latent = reverse_model(
                            matched_latent,
                            reverse_context,
                            diffusion_generator,
                            verbose=False,
                        )
                    update_confusion(
                        matrices[name],
                        evaluator,
                        vae,
                        evaluated_latent,
                        labels,
                    )

            num_samples += batch_size
            frame_snr_sum += args.snr_db * batch_size
            if (
                previous_num_samples == 0
                or num_samples // args.log_interval
                > previous_num_samples // args.log_interval
            ):
                result = save_result(
                    output_path,
                    checkpoint_path,
                    checkpoint,
                    args.snr_db,
                    num_samples,
                    frame_snr_sum,
                    matrices,
                    latent_snr_sums,
                    symbol_power_sums,
                    unsupported_patch_counts,
                    diffusion_model,
                    args.diffusion_steps,
                )
                progress = " ".join(
                    f"{name}_miou={values['miou']:.4f}"
                    for name, values in result["strategies"].items()
                )
                print(f"samples={num_samples}/{len(dataset)} {progress}", flush=True)

    result = save_result(
        output_path,
        checkpoint_path,
        checkpoint,
        args.snr_db,
        num_samples,
        frame_snr_sum,
        matrices,
        latent_snr_sums,
        symbol_power_sums,
        unsupported_patch_counts,
        diffusion_model,
        args.diffusion_steps,
    )
    print(json.dumps(result, indent=2))
    print(f"Saved validation result to {output_path}")


def main() -> None:
    from segmentation_orthogonal_power import dispatch
    dispatch("evaluation", legacy_main)


if __name__ == "__main__":
    main()
