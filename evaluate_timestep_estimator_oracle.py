"""Evaluate a timestep estimator against a clean-latent gain oracle."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    encode_latents,
)
from train_timestep_estimator import (
    ESTIMATOR_VERSION,
    NUM_TIMESTEPS,
    POWER_MAP_SOURCES,
    PatchTimestepEstimator,
    build_dataloaders,
    ddim_schedule_timesteps,
    estimate_channel_from_ddim_logits,
    load_frozen_power_allocator,
    noise_variance_to_continuous_timestep,
    project_alpha_bar_estimates,
    project_ddim_class_estimates,
    project_log_lambda_estimates,
    project_power_estimates_to_ddim_classes,
    sample_channel_batch,
    timestep_to_ddim_class,
)


def freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def resolve_checkpoint(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path.resolve()


def predict_estimator_class(
    model_output: dict[str, torch.Tensor],
    channel_batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor:
    prediction_mode = config.get("prediction_mode", "alpha_bar")
    ddim_steps = int(config.get("ddim_steps", 25))
    temperature = float(
        config.get(
            "classification_temperature",
            0.5
            if prediction_mode == "projected_power_classification"
            else 5.0,
        )
    )
    if prediction_mode in ("alpha_bar", "log_lambda"):
        projection_function = (
            project_alpha_bar_estimates
            if prediction_mode == "alpha_bar"
            else project_log_lambda_estimates
        )
        estimates = projection_function(
            model_output,
            channel_batch["noise_spectral_density"],
            channel_batch["target_frame_energy"],
        )
        return timestep_to_ddim_class(
            estimates["timestep"].round(),
            ddim_steps,
        )
    if prediction_mode == "ddim_classification":
        inference_mode = config.get(
            "classification_inference_mode",
            "projected",
        )
        if inference_mode == "raw_logits":
            estimates = estimate_channel_from_ddim_logits(
                model_output["raw_class_logits"],
                channel_batch["noise_spectral_density"],
                ddim_steps,
            )
            return estimates["predicted_class"]
        estimates = project_ddim_class_estimates(
            model_output,
            channel_batch["noise_spectral_density"],
            channel_batch["target_frame_energy"],
            ddim_steps,
            temperature,
            int(config.get("projected_logit_version", 1)) >= 2,
        )
        return estimates["class_logits"].argmax(dim=1, keepdim=True)
    if prediction_mode == "projected_power_classification":
        estimates = project_power_estimates_to_ddim_classes(
            model_output,
            channel_batch["noise_spectral_density"],
            channel_batch["target_frame_energy"],
            ddim_steps,
            temperature,
        )
        return estimates["predicted_class"]
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def clean_latent_oracle_class(
    clean_latent: torch.Tensor,
    channel_batch: dict[str, torch.Tensor],
    patch_height: int,
    patch_width: int,
    ddim_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate positive normalized gain using unavailable clean latent z."""
    noise_standard_deviation = torch.sqrt(
        channel_batch["noise_spectral_density"].float() / 2.0
    )
    normalized_received = channel_batch["received_latent"].float() / (
        noise_standard_deviation[:, None, None, None]
    )
    unfold_options = {
        "kernel_size": (patch_height, patch_width),
        "stride": (patch_height, patch_width),
    }
    clean_patches = F.unfold(clean_latent.float(), **unfold_options)
    received_patches = F.unfold(normalized_received, **unfold_options)
    clean_energy = clean_patches.square().sum(dim=1).clamp_min(1e-20)
    unconstrained_gain = (
        received_patches * clean_patches
    ).sum(dim=1) / clean_energy

    # The modeled channel gain is positive. Project negative LS estimates to
    # the boundary rather than making them positive by squaring.
    estimated_gain = unconstrained_gain.clamp_min(0.0)
    estimated_noise_variance = (
        estimated_gain.clamp_min(1e-12).square().reciprocal()
    )
    estimated_timestep = noise_variance_to_continuous_timestep(
        estimated_noise_variance
    )
    oracle_class = timestep_to_ddim_class(
        estimated_timestep.round(),
        ddim_steps,
    )

    batch_size, _, height, width = clean_latent.shape
    grid_height = height // patch_height
    grid_width = width // patch_width
    return (
        oracle_class.reshape(batch_size, 1, grid_height, grid_width),
        unconstrained_gain,
    )


def add_class_counts(
    totals: torch.Tensor,
    target_class: torch.Tensor,
    estimator_class: torch.Tensor,
    oracle_class: torch.Tensor,
) -> None:
    target = target_class.reshape(-1).long()
    estimator = estimator_class.reshape(-1).long()
    oracle = oracle_class.reshape(-1).long()
    num_classes = totals.shape[1]
    totals[0] += torch.bincount(target, minlength=num_classes).cpu()
    totals[1] += torch.bincount(
        target[estimator.eq(target)],
        minlength=num_classes,
    ).cpu()
    totals[2] += torch.bincount(
        target[oracle.eq(target)],
        minlength=num_classes,
    ).cpu()
    totals[3] += torch.bincount(
        target[(estimator - target).abs().le(1)],
        minlength=num_classes,
    ).cpu()
    totals[4] += torch.bincount(
        target[(oracle - target).abs().le(1)],
        minlength=num_classes,
    ).cpu()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:6.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare class-wise estimator and clean-latent oracle accuracy "
            "on a full validation set"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snr_db", type=float, required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--allocator_checkpoint", default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.batch is not None and args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be nonnegative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if config.get("stage") != "timestep_estimator":
        raise ValueError("Checkpoint is not a timestep estimator")
    if config.get("estimator_version") != ESTIMATOR_VERSION:
        raise ValueError("Checkpoint uses an incompatible estimator version")

    data_path = args.data_path or config.get("data_path")
    if data_path is None:
        raise ValueError("Dataset path is missing; pass --data_path")
    batch_size = args.batch or int(config.get("batch", 1))
    _, validation_loader = build_dataloaders(
        config["dataset"],
        data_path,
        batch_size,
        int(config.get("image_size", 512)),
        False,
        args.seed,
        args.num_workers,
    )

    vae_path = args.vae_path or config.get("vae_model_id", VAE_MODEL_ID)
    vae_kwargs = {
        "local_files_only": args.local_files_only,
        "torch_dtype": torch.float32,
    }
    if args.vae_path is None:
        vae_kwargs["subfolder"] = "vae"
        vae_kwargs["revision"] = config.get(
            "vae_model_revision",
            VAE_MODEL_REVISION,
        )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        **vae_kwargs,
    ).to(device)
    freeze(vae)

    patch_height, patch_width = map(int, config["patch_shape"])
    ddim_steps = int(config.get("ddim_steps", 25))
    estimator = PatchTimestepEstimator(
        int(config["latent_channels"]),
        patch_height,
        patch_width,
        int(config["hidden_dim"]),
        config.get("prediction_mode", "alpha_bar"),
        ddim_steps,
        config.get("statistic_mode", "basic"),
        config.get("input_mode", "statistics"),
    ).to(device)
    estimator.load_state_dict(checkpoint["estimator"])
    freeze(estimator)

    power_map_source = config.get("power_map_source", "random")
    if power_map_source not in POWER_MAP_SOURCES:
        raise ValueError(f"Unknown power_map_source: {power_map_source}")
    power_allocator = None
    allocator_checkpoint_path = None
    allocator_temperature = 1.0
    allocator_snr_scale_db = 1.0
    if power_map_source == "allocator":
        allocator_checkpoint_path = (
            args.allocator_checkpoint
            or config.get("power_allocator_checkpoint")
            or config.get("allocator_checkpoint")
        )
        if allocator_checkpoint_path is None:
            raise ValueError(
                "Allocator-trained estimator requires its allocator "
                "checkpoint; pass --allocator_checkpoint"
            )
        (
            power_allocator,
            _allocator_config,
            allocator_checkpoint_path,
        ) = load_frozen_power_allocator(
            allocator_checkpoint_path,
            int(config["latent_channels"]),
            device,
        )
        allocator_temperature = float(
            config.get("allocator_temperature_effective")
            or config.get("allocator_temperature")
            or 1.0
        )
        allocator_snr_scale_db = float(
            config.get("allocator_snr_scale_db_effective")
            or config.get("allocator_snr_scale_db")
            or 1.0
        )

    totals = torch.zeros((5, ddim_steps), dtype=torch.long)
    processed_samples = 0
    negative_oracle_gain_count = 0
    channel_generator = torch.Generator(device=device)
    channel_generator.manual_seed(args.seed + 1)
    amp_enabled = args.amp and device.type == "cuda"

    with torch.inference_mode():
        for batch_index, (images, _labels) in enumerate(validation_loader):
            if args.max_samples is not None:
                remaining = args.max_samples - processed_samples
                if remaining <= 0:
                    break
                images = images[:remaining]
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                clean_latent = encode_latents(images, vae)

            snr_db = torch.full(
                (images.shape[0],),
                args.snr_db,
                device=device,
                dtype=torch.float32,
            )
            channel_batch = sample_channel_batch(
                clean_latent,
                snr_db,
                patch_height,
                patch_width,
                channel_generator,
                int(config.get("max_timestep", NUM_TIMESTEPS - 1)),
                float(config.get("power_log_std_min", 0.25)),
                float(config.get("power_log_std_max", 1.25)),
                config.get("statistic_mode", "basic"),
                power_map_source,
                power_allocator,
                allocator_temperature,
                allocator_snr_scale_db,
                config.get("input_mode", "statistics"),
            )
            estimator_input = (
                channel_batch["statistic_map"]
                if config.get("input_mode", "statistics")
                == "statistics"
                else channel_batch["normalized_received_latent"]
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                model_output = estimator(estimator_input)
            estimator_class = predict_estimator_class(
                model_output,
                channel_batch,
                config,
            )
            target_class = timestep_to_ddim_class(
                channel_batch["target_timestep"],
                ddim_steps,
            )
            oracle_class, unconstrained_gain = clean_latent_oracle_class(
                clean_latent,
                channel_batch,
                patch_height,
                patch_width,
                ddim_steps,
            )
            negative_oracle_gain_count += int(
                unconstrained_gain.le(0.0).sum().item()
            )
            add_class_counts(
                totals,
                target_class,
                estimator_class,
                oracle_class,
            )
            processed_samples += images.shape[0]

            if (
                batch_index == 0
                or (batch_index + 1) % args.log_interval == 0
            ):
                print(
                    f"processed={processed_samples} "
                    f"batches={batch_index + 1}/{len(validation_loader)}",
                    flush=True,
                )

    count, estimator_exact, oracle_exact, estimator_one, oracle_one = (
        totals.tolist()
    )
    rows = []
    for class_index in range(ddim_steps):
        rows.append(
            {
                "class": class_index,
                "count": count[class_index],
                "estimator_accuracy": ratio(
                    estimator_exact[class_index],
                    count[class_index],
                ),
                "oracle_accuracy": ratio(
                    oracle_exact[class_index],
                    count[class_index],
                ),
                "estimator_within_one": ratio(
                    estimator_one[class_index],
                    count[class_index],
                ),
                "oracle_within_one": ratio(
                    oracle_one[class_index],
                    count[class_index],
                ),
            }
        )

    total_patches = sum(count)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "dataset": config["dataset"],
        "dataset_size": len(validation_loader.dataset),
        "processed_samples": processed_samples,
        "snr_db": args.snr_db,
        "seed": args.seed,
        "batch_size": batch_size,
        "prediction_mode": config.get("prediction_mode", "alpha_bar"),
        "input_mode": config.get("input_mode", "statistics"),
        "statistic_mode": config.get("statistic_mode", "basic"),
        "power_map_source": power_map_source,
        "allocator_checkpoint": allocator_checkpoint_path,
        "patch_shape": [patch_height, patch_width],
        "ddim_steps": ddim_steps,
        "ddim_schedule_timesteps": ddim_schedule_timesteps(
            ddim_steps
        ).tolist(),
        "oracle": {
            "method": "positive_clean_latent_least_squares_gain",
            "normalized_observation": "u=y/sqrt(N0/2)",
            "gain": "max(0, dot(u,z)/squared_norm(z))",
            "noise_variance": "1/max(gain,1e-12)^2",
            "negative_unconstrained_gain_count": (
                negative_oracle_gain_count
            ),
        },
        "overall": {
            "patch_count": total_patches,
            "estimator_accuracy": ratio(
                sum(estimator_exact),
                total_patches,
            ),
            "oracle_accuracy": ratio(sum(oracle_exact), total_patches),
            "estimator_within_one": ratio(
                sum(estimator_one),
                total_patches,
            ),
            "oracle_within_one": ratio(
                sum(oracle_one),
                total_patches,
            ),
        },
        "classes": rows,
    }

    output_path = Path(
        args.output
        or checkpoint_path.parent
        / f"oracle_validation_snr_{args.snr_db:g}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(
        "class | count | estimator acc | oracle acc | estimator +/-1 | "
        "oracle +/-1"
    )
    for row in rows:
        print(
            f"{row['class']:5d} | {row['count']:7d} | "
            f"{format_percent(row['estimator_accuracy']):>13s} | "
            f"{format_percent(row['oracle_accuracy']):>10s} | "
            f"{format_percent(row['estimator_within_one']):>14s} | "
            f"{format_percent(row['oracle_within_one']):>11s}"
        )
    print(
        f"overall: "
        f"estimator={format_percent(result['overall']['estimator_accuracy'])} "
        f"oracle={format_percent(result['overall']['oracle_accuracy'])} "
        f"estimator+/-1="
        f"{format_percent(result['overall']['estimator_within_one'])} "
        f"oracle+/-1={format_percent(result['overall']['oracle_within_one'])}"
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
