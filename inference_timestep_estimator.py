import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from datasets.data_processing import (
    CityscapesSemanticDataset,
    construct_dataloaders,
)
from inference import _freeze
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    encode_latents,
)
from train_timestep_estimator import (
    ESTIMATOR_VERSION,
    NUM_TIMESTEPS,
    PatchTimestepEstimator,
    ddim_class_to_timestep,
    estimate_channel_from_ddim_logits,
    noise_variance_to_normalized_log_lambda,
    project_alpha_bar_estimates,
    project_ddim_class_estimates,
    project_log_lambda_estimates,
    project_power_estimates_to_ddim_classes,
    load_frozen_power_allocator,
    sample_channel_batch,
    sample_uniform_snr_db,
    timestep_to_ddim_class,
)


def load_validation_image(
    config: dict,
    data_path: str,
    sample_index: int | None,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    dataset_name = config["dataset"]
    if dataset_name == "Cityscapes":
        dataset = CityscapesSemanticDataset(data_path, "val")
    elif dataset_name == "ImageNet":
        _, validation_loader = construct_dataloaders(
            dataset="ImageNet",
            data_path=data_path,
            batch_size=1,
            normalize=False,
            image_size=int(config.get("image_size", 256)),
        )
        dataset = validation_loader.dataset
    else:
        raise ValueError(
            "Estimator checkpoint dataset must be Cityscapes or ImageNet"
        )

    if sample_index is None:
        sample_index = random.Random(seed).randrange(len(dataset))
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample_index must be in [0, {len(dataset) - 1}]"
        )
    image, _label = dataset[sample_index]
    return image.unsqueeze(0), sample_index, len(dataset)


def timestep_heatmap(
    timestep_map: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    normalized = timestep_map.detach().float().reshape(1, 1, *timestep_map.shape[-2:])
    normalized = (normalized / float(NUM_TIMESTEPS - 1)).clamp(0.0, 1.0)
    normalized = F.interpolate(
        normalized,
        size=output_size,
        mode="nearest",
    )[0, 0]
    return torch.stack(
        [
            normalized,
            1.0 - (2.0 * normalized - 1.0).abs(),
            1.0 - normalized,
        ],
        dim=0,
    )


def error_heatmap(
    absolute_error: torch.Tensor,
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    maximum = max(float(absolute_error.max().item()), 1.0)
    normalized = absolute_error.detach().float().reshape(
        1,
        1,
        *absolute_error.shape[-2:],
    )
    normalized = (normalized / maximum).clamp(0.0, 1.0)
    normalized = F.interpolate(
        normalized,
        size=output_size,
        mode="nearest",
    )[0, 0]
    heatmap = torch.stack(
        [
            normalized,
            1.0 - normalized,
            torch.zeros_like(normalized),
        ],
        dim=0,
    )
    return heatmap, maximum


def _map_font(cell_width: int, cell_height: int) -> ImageFont.ImageFont:
    font_size = max(12, min(cell_width, cell_height) // 3)
    font_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_patch_values(
    visualization: torch.Tensor,
    values: torch.Tensor,
    formatter,
) -> Image.Image:
    image = to_pil_image(
        visualization.detach().float().cpu().clamp(0.0, 1.0)
    ).convert("RGB")
    value_grid = values.detach().float().squeeze().cpu()
    if value_grid.ndim != 2:
        raise ValueError("Patch value map must be two-dimensional")
    rows, cols = value_grid.shape
    cell_width = image.width // cols
    cell_height = image.height // rows
    font = _map_font(cell_width, cell_height)
    draw = ImageDraw.Draw(image)

    for row in range(rows):
        top = row * cell_height
        bottom = image.height if row + 1 == rows else (row + 1) * cell_height
        for col in range(cols):
            left = col * cell_width
            right = image.width if col + 1 == cols else (col + 1) * cell_width
            text = formatter(float(value_grid[row, col].item()))
            text_box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            x = left + (right - left - text_width) / 2
            y = top + (bottom - top - text_height) / 2 - text_box[1]
            draw.text(
                (x, y),
                text,
                font=font,
                fill="black",
                stroke_width=2,
                stroke_fill="white",
            )

    for row in range(rows + 1):
        y = min(row * cell_height, image.height - 1)
        draw.line((0, y, image.width, y), fill="black", width=1)
    for col in range(cols + 1):
        x = min(col * cell_width, image.width - 1)
        draw.line((x, 0, x, image.height), fill="black", width=1)
    return image


def save_vertical_comparison(
    panels: list[tuple[str, Image.Image]],
    output_path: Path,
) -> None:
    width = max(panel.width for _, panel in panels)
    title_height = 36
    height = sum(panel.height + title_height for _, panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    title_font = _map_font(width, 72)
    draw = ImageDraw.Draw(canvas)
    top = 0
    for title, panel in panels:
        draw.text((8, top + 6), title, font=title_font, fill="black")
        top += title_height
        canvas.paste(panel, (0, top))
        top += panel.height
    canvas.save(output_path)


def save_map_outputs(
    image: torch.Tensor,
    actual_timestep: torch.Tensor,
    predicted_timestep: torch.Tensor,
    absolute_error: torch.Tensor,
    output_dir: Path,
) -> float:
    output_size = tuple(image.shape[-2:])
    actual_visualization = timestep_heatmap(actual_timestep, output_size)
    predicted_visualization = timestep_heatmap(
        predicted_timestep,
        output_size,
    )
    error_visualization, maximum_visualized_error = error_heatmap(
        absolute_error,
        output_size,
    )
    input_image = to_pil_image(
        image[0].detach().float().cpu().clamp(0.0, 1.0)
    ).convert("RGB")
    actual_image = annotate_patch_values(
        actual_visualization,
        actual_timestep,
        lambda value: f"{value:.0f}",
    )
    predicted_image = annotate_patch_values(
        predicted_visualization,
        predicted_timestep,
        lambda value: f"{value:.0f}",
    )
    error_image = annotate_patch_values(
        error_visualization,
        absolute_error,
        lambda value: f"{value:.0f}",
    )

    input_image.save(output_dir / "input.png")
    actual_image.save(output_dir / "actual_timestep_map.png")
    predicted_image.save(output_dir / "predicted_timestep_map.png")
    error_image.save(output_dir / "absolute_error_map.png")
    save_vertical_comparison(
        [
            ("Input image", input_image),
            (
                "Actual timestep: fixed scale, 0=blue and 999=red",
                actual_image,
            ),
            (
                "Predicted timestep: fixed scale, 0=blue and 999=red",
                predicted_image,
            ),
            (
                "Absolute timestep error: green=0 and red=sample maximum",
                error_image,
            ),
        ],
        output_dir / "comparison.png",
    )
    return maximum_visualized_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one random channel realization through a trained patch "
            "timestep estimator and save actual/predicted timestep maps"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument(
        "--allocator_checkpoint",
        default=None,
        help="Override the allocator checkpoint saved in estimator config",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
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

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if config.get("stage") != "timestep_estimator":
        raise ValueError("Checkpoint is not a timestep estimator")
    if config.get("estimator_version") != ESTIMATOR_VERSION:
        raise ValueError(
            "Checkpoint uses the old latent-input timestep estimator"
        )
    if int(config.get("num_timesteps", NUM_TIMESTEPS)) != NUM_TIMESTEPS:
        raise ValueError("Checkpoint uses an incompatible timestep schedule")

    data_path = args.data_path or config["data_path"]
    image, sample_index, dataset_size = load_validation_image(
        config,
        data_path,
        args.sample_index,
        args.seed,
    )
    image = image.to(device)

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
    _freeze(vae)

    patch_height, patch_width = config["patch_shape"]
    prediction_mode = config.get("prediction_mode", "alpha_bar")
    ddim_steps = int(config.get("ddim_steps", 25))
    classification_inference_mode = config.get(
        "classification_inference_mode",
        "projected",
    )
    classification_temperature = float(
        config.get(
            "classification_temperature",
            0.5
            if prediction_mode == "projected_power_classification"
            else 5.0,
        )
    )
    normalize_projected_distance = (
        int(config.get("projected_logit_version") or 1) >= 2
    )
    estimator = PatchTimestepEstimator(
        int(config["latent_channels"]),
        int(patch_height),
        int(patch_width),
        int(config["hidden_dim"]),
        prediction_mode,
        ddim_steps,
        config.get("statistic_mode", "basic"),
        config.get("input_mode", "statistics"),
    ).to(device)
    estimator.load_state_dict(checkpoint["estimator"])
    _freeze(estimator)

    power_map_source = config.get("power_map_source", "random")
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
                "Allocator-trained estimator is missing its power "
                "allocator checkpoint; pass --allocator_checkpoint"
            )
        (
            power_allocator,
            _power_allocator_config,
            allocator_checkpoint_path,
        ) = load_frozen_power_allocator(
            allocator_checkpoint_path,
            int(config["latent_channels"]),
            device,
        )
        allocator_temperature = float(
            config.get(
                "allocator_temperature_effective",
                config.get("allocator_temperature", 1.0),
            )
        )
        allocator_snr_scale_db = float(
            config.get(
                "allocator_snr_scale_db_effective",
                config.get("allocator_snr_scale_db", 1.0),
            )
        )

    amp_enabled = args.amp and device.type == "cuda"
    channel_generator = torch.Generator(device=device)
    channel_generator.manual_seed(args.seed + 1)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            latent = encode_latents(image, vae)

        if args.snr_db is None:
            snr_db = sample_uniform_snr_db(
                1,
                device,
                float(config.get("snr_db_min", -10.0)),
                float(config.get("snr_db_max", 10.0)),
                channel_generator,
            )
        else:
            snr_db = torch.tensor(
                [args.snr_db],
                device=device,
                dtype=torch.float32,
            )
        channel_batch = sample_channel_batch(
            latent,
            snr_db,
            int(patch_height),
            int(patch_width),
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
            if config.get("input_mode", "statistics") == "statistics"
            else channel_batch["normalized_received_latent"]
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            model_output = estimator(estimator_input)
            if prediction_mode in ("alpha_bar", "log_lambda"):
                projection_function = (
                    project_alpha_bar_estimates
                    if prediction_mode == "alpha_bar"
                    else project_log_lambda_estimates
                )
                projected = projection_function(
                    model_output,
                    channel_batch["noise_spectral_density"],
                    channel_batch["target_frame_energy"],
                )
            elif prediction_mode == "ddim_classification":
                if classification_inference_mode == "raw_logits":
                    class_estimates = estimate_channel_from_ddim_logits(
                        model_output["raw_class_logits"],
                        channel_batch["noise_spectral_density"],
                        ddim_steps,
                    )
                else:
                    class_estimates = project_ddim_class_estimates(
                        model_output,
                        channel_batch["noise_spectral_density"],
                        channel_batch["target_frame_energy"],
                        ddim_steps,
                        classification_temperature,
                        normalize_projected_distance,
                    )
            elif prediction_mode == "projected_power_classification":
                class_estimates = project_power_estimates_to_ddim_classes(
                    model_output,
                    channel_batch["noise_spectral_density"],
                    channel_batch["target_frame_energy"],
                    ddim_steps,
                    classification_temperature,
                )

    oracle_timestep = channel_batch["target_timestep"].float()
    actual_class = timestep_to_ddim_class(oracle_timestep, ddim_steps)
    if prediction_mode in ("alpha_bar", "log_lambda"):
        predicted_timestep = projected["timestep"].float()
        actual_timestep = oracle_timestep
        predicted_alpha_bar = projected["alpha_bar"].float()
        predicted_power = projected["projected_power"].float()
        actual_alpha_bar = 1.0 / (
            1.0
            + channel_batch["noise_variance"].reshape_as(predicted_alpha_bar)
        )
        predicted_class = timestep_to_ddim_class(
            predicted_timestep.round(),
            ddim_steps,
        )
        class_probabilities = None
        raw_class_probabilities = None
        raw_predicted_class = predicted_class
    elif prediction_mode == "ddim_classification":
        if classification_inference_mode == "raw_logits":
            class_probabilities = class_estimates["class_probabilities"]
            raw_class_probabilities = class_probabilities
            predicted_class = class_estimates["predicted_class"]
            raw_predicted_class = predicted_class
            predicted_alpha_bar = class_estimates["alpha_bar"].float()
            predicted_power = class_estimates["power"].float()
            raw_alpha_bar = predicted_alpha_bar
            predicted_patch_energy = None
        else:
            class_probabilities = class_estimates[
                "class_logits"
            ].softmax(dim=1)
            predicted_class = class_probabilities.argmax(
                dim=1,
                keepdim=True,
            )
            raw_class_probabilities = class_estimates[
                "raw_class_probabilities"
            ]
            raw_predicted_class = raw_class_probabilities.argmax(
                dim=1,
                keepdim=True,
            )
            predicted_alpha_bar = class_estimates["alpha_bar"].float()
            predicted_power = class_estimates["projected_power"].float()
            raw_alpha_bar = class_estimates["raw_alpha_bar"].float()
            predicted_patch_energy = class_estimates[
                "predicted_patch_energy"
            ].float()
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            ddim_steps,
        ).float()
        actual_timestep = ddim_class_to_timestep(
            actual_class,
            ddim_steps,
        ).float()
        actual_alpha_bar = 1.0 / (
            1.0
            + channel_batch["noise_variance"].reshape_as(predicted_alpha_bar)
        )
    elif prediction_mode == "projected_power_classification":
        class_probabilities = class_estimates["class_probabilities"]
        raw_class_probabilities = class_probabilities
        predicted_class = class_estimates["predicted_class"]
        raw_predicted_class = predicted_class
        predicted_alpha_bar = class_estimates["alpha_bar"].float()
        predicted_power = class_estimates["projected_power"].float()
        raw_alpha_bar = predicted_alpha_bar
        predicted_patch_energy = class_estimates[
            "predicted_patch_energy"
        ].float()
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            ddim_steps,
        ).float()
        actual_timestep = ddim_class_to_timestep(
            actual_class,
            ddim_steps,
        ).float()
        actual_alpha_bar = 1.0 / (
            1.0
            + channel_batch["noise_variance"].reshape_as(predicted_alpha_bar)
        )
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")
    rounded_timestep = predicted_timestep.round().clamp(
        0,
        NUM_TIMESTEPS - 1,
    )
    absolute_error = (predicted_timestep - actual_timestep).abs()
    squared_error = (predicted_timestep - actual_timestep).square()
    class_distance = (predicted_class - actual_class).abs()
    predicted_frame_energy = (
        predicted_power * channel_batch["clean_patch_energy"].float()
    ).flatten(1).sum(dim=1)
    budget_relative_error = (
        predicted_frame_energy / channel_batch["target_frame_energy"] - 1.0
    )

    output_dir = Path(
        args.output_dir
        or checkpoint_path.parent
        / "inference"
        / (
            f"sample_{sample_index:06d}_seed_{args.seed}_"
            f"snr_{float(snr_db.item()):g}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    maximum_visualized_error = save_map_outputs(
        image,
        actual_timestep,
        predicted_timestep,
        absolute_error,
        output_dir,
    )

    maps_path = output_dir / "timestep_maps.pt"
    map_payload = {
            "actual_timestep": actual_timestep.cpu(),
            "oracle_timestep": oracle_timestep.cpu(),
            "predicted_timestep": predicted_timestep.cpu(),
            "rounded_timestep": rounded_timestep.cpu().to(torch.long),
            "absolute_error": absolute_error.cpu(),
            "actual_power": channel_batch["group_power"].cpu(),
            "actual_patch_energy": channel_batch[
                "clean_patch_energy"
            ].cpu(),
            "actual_ddim_class": actual_class.cpu(),
            "predicted_ddim_class": predicted_class.cpu(),
            "estimator_input": estimator_input.cpu(),
            "effective_gain": channel_batch["effective_gain"].cpu(),
            "noise_variance": channel_batch["noise_variance"].cpu(),
    }
    if channel_batch["statistic_map"] is not None:
        map_payload["statistic_map"] = channel_batch[
            "statistic_map"
        ].cpu()
    else:
        map_payload["normalized_received_latent"] = estimator_input.cpu()
    if prediction_mode in ("alpha_bar", "log_lambda"):
        map_payload.update(
            {
                "actual_alpha_bar": actual_alpha_bar.cpu(),
                "predicted_alpha_bar": predicted_alpha_bar.cpu(),
                "raw_alpha_bar": projected["raw_alpha_bar"].cpu(),
                "predicted_power": projected["projected_power"].cpu(),
                "predicted_patch_energy": projected[
                    "predicted_patch_energy"
                ].cpu(),
            }
        )
        if prediction_mode == "log_lambda":
            actual_normalized_log_lambda = (
                noise_variance_to_normalized_log_lambda(
                    channel_batch["noise_variance"].reshape_as(
                        predicted_timestep
                    )
                )
            )
            map_payload.update(
                {
                    "actual_normalized_log_lambda": (
                        actual_normalized_log_lambda.cpu()
                    ),
                    "predicted_normalized_log_lambda": projected[
                        "normalized_log_lambda"
                    ].cpu(),
                    "raw_normalized_log_lambda": projected[
                        "raw_normalized_log_lambda"
                    ].cpu(),
                }
            )
    else:
        map_payload.update(
            {
                "class_probabilities": class_probabilities.cpu(),
                "raw_class_probabilities": raw_class_probabilities.cpu(),
                "raw_predicted_ddim_class": raw_predicted_class.cpu(),
                "actual_alpha_bar": actual_alpha_bar.cpu(),
                "predicted_alpha_bar": predicted_alpha_bar.cpu(),
                "raw_alpha_bar": raw_alpha_bar.cpu(),
                "predicted_power": predicted_power.cpu(),
            }
        )
        if predicted_patch_energy is not None:
            map_payload["predicted_patch_energy"] = (
                predicted_patch_energy.cpu()
            )
        if prediction_mode == "projected_power_classification":
            map_payload["raw_log_power"] = class_estimates[
                "raw_log_power"
            ].cpu()
    torch.save(map_payload, maps_path)

    patch_count = actual_timestep.numel()
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "dataset": config["dataset"],
        "prediction_mode": prediction_mode,
        "input_mode": config.get("input_mode", "statistics"),
        "statistic_mode": config.get("statistic_mode", "basic"),
        "power_map_source": power_map_source,
        "allocator_checkpoint": allocator_checkpoint_path,
        "classification_inference_mode": (
            classification_inference_mode
            if prediction_mode
            in ("ddim_classification", "projected_power_classification")
            else None
        ),
        "ddim_steps": ddim_steps,
        "dataset_size": dataset_size,
        "sample_index": sample_index,
        "seed": args.seed,
        "snr_db": float(snr_db.item()),
        "noise_spectral_density": float(
            channel_batch["noise_spectral_density"].item()
        ),
        "image_shape": list(image.shape),
        "latent_shape": list(latent.shape),
        "patch_shape": [int(patch_height), int(patch_width)],
        "timestep_map_shape": list(actual_timestep.shape),
        "num_patches": patch_count,
        "metrics": {
            "alpha_bar_mse": (
                float(
                    F.mse_loss(
                        predicted_alpha_bar,
                        actual_alpha_bar,
                    ).item()
                )
            ),
            "timestep_rmse": math.sqrt(squared_error.mean().item()),
            "timestep_mae": float(absolute_error.mean().item()),
            "within_5": float((absolute_error <= 5.0).float().mean().item()),
            "within_10": float((absolute_error <= 10.0).float().mean().item()),
            "within_25": float((absolute_error <= 25.0).float().mean().item()),
            "exact_rounded": float(
                rounded_timestep.eq(actual_timestep).float().mean().item()
            ),
            "same_ddim_class": float(
                predicted_class.eq(actual_class).float().mean().item()
            ),
            "raw_same_ddim_class": float(
                raw_predicted_class.eq(actual_class).float().mean().item()
            ),
            "within_one_ddim_class": float(
                (class_distance <= 1).float().mean().item()
            ),
            "mean_ddim_class_distance": float(
                class_distance.float().mean().item()
            ),
            "budget_relative_error": float(
                budget_relative_error.abs().mean().item()
            ),
        },
        "actual_timestep_range": [
            int(actual_timestep.min().item()),
            int(actual_timestep.max().item()),
        ],
        "predicted_timestep_range": [
            float(predicted_timestep.min().item()),
            float(predicted_timestep.max().item()),
        ],
        "files": {
            "input": "input.png",
            "actual_map": "actual_timestep_map.png",
            "predicted_map": "predicted_timestep_map.png",
            "absolute_error_map": "absolute_error_map.png",
            "comparison": "comparison.png",
            "raw_maps": maps_path.name,
        },
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved timestep estimator inference output to {output_dir}")


if __name__ == "__main__":
    main()
