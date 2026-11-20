import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torchvision.transforms.functional import to_pil_image

from datasets.data_processing import CityscapesSemanticDataset
from inference import (
    StaggeredReverseDiffusion,
    _decode_latent,
    _freeze,
    _save_labeled_row,
)
from inference_power import ActualPowerTransport, _match_diffusion_start
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    PowerAllocator,
    encode_latents,
)
from train_segmentation_power_allocator import (
    AnalogAWGNChannel,
    allocate_power_with_timestep_floor,
)
from train_timestep_estimator import (
    ESTIMATOR_VERSION,
    NUM_TIMESTEPS,
    PatchTimestepEstimator,
    build_patch_statistic_map,
    ddim_class_to_timestep,
    estimate_channel_from_ddim_logits,
    project_alpha_bar_estimates,
    project_ddim_class_estimates,
    project_log_lambda_estimates,
    project_power_estimates_to_ddim_classes,
)
from visualization_utils import (
    noise_variance_to_diffusion_timestep,
    save_diffusion_timestep_overlay,
)


def raw_received_frame_to_latent(
    received_frame: torch.Tensor,
    reference_latent: torch.Tensor,
    group_rows: int,
    group_cols: int,
    kappa_m: float,
) -> torch.Tensor:
    """Unpack the pilot-free complex channel frame into real latent maps."""
    batch_size, channels, height, width = reference_latent.shape
    if height % group_rows or width % group_cols:
        raise ValueError("Latent shape is incompatible with receiver groups")
    group_height = height // group_rows
    group_width = width // group_cols
    real_values_per_group = channels * group_height * group_width
    if real_values_per_group % 2:
        raise ValueError("Each latent group must contain an even value count")
    complex_symbols_per_group = real_values_per_group // 2
    num_groups = group_rows * group_cols
    expected_symbols = num_groups * complex_symbols_per_group
    if received_frame.shape != (batch_size, expected_symbols):
        raise ValueError(
            "Pilot-free received frame has an unexpected shape: "
            f"expected {(batch_size, expected_symbols)}, got "
            f"{tuple(received_frame.shape)}"
        )

    complex_groups = received_frame.reshape(
        batch_size,
        num_groups,
        complex_symbols_per_group,
    )
    groups = []
    for group_index in range(num_groups):
        payload = complex_groups[:, group_index]
        flattened = torch.empty(
            batch_size,
            real_values_per_group,
            device=reference_latent.device,
            dtype=reference_latent.dtype,
        )
        flattened[:, 0::2] = float(kappa_m) * payload.real
        flattened[:, 1::2] = float(kappa_m) * payload.imag
        groups.append(
            flattened.reshape(
                batch_size,
                channels,
                group_height,
                group_width,
            )
        )
    rows = [
        torch.cat(
            groups[row * group_cols:(row + 1) * group_cols],
            dim=3,
        )
        for row in range(group_rows)
    ]
    return torch.cat(rows, dim=2)


def timestep_estimator_receive(
    reference_latent: torch.Tensor,
    channel_info: dict,
    transport_info: dict,
    estimator: PatchTimestepEstimator,
    group_rows: int,
    group_cols: int,
    kappa_m: float,
    classification_temperature: float = 5.0,
    normalize_projected_distance: bool = False,
    classification_inference_mode: str = "projected",
) -> dict:
    raw_received_latent = raw_received_frame_to_latent(
        transport_info["received_frame"],
        reference_latent,
        group_rows,
        group_cols,
        kappa_m,
    )
    noise_spectral_density = channel_info["noise_spectral_density"]
    noise_standard_deviation = torch.sqrt(
        noise_spectral_density / 2.0
    )
    estimator_input = raw_received_latent / noise_standard_deviation[
        :, None, None, None
    ].to(dtype=raw_received_latent.dtype)
    statistic_map = (
        build_patch_statistic_map(
            estimator_input,
            estimator.patch_height,
            estimator.patch_width,
            estimator.statistic_mode,
        )
        if estimator.input_mode == "statistics"
        else None
    )
    model_output = estimator(
        statistic_map
        if estimator.input_mode == "statistics"
        else estimator_input
    )
    predicted_class = None
    raw_predicted_class = None
    class_probabilities = None
    normalized_log_lambda_map = None
    raw_normalized_log_lambda_map = None
    if estimator.prediction_mode in ("alpha_bar", "log_lambda"):
        projection_function = (
            project_alpha_bar_estimates
            if estimator.prediction_mode == "alpha_bar"
            else project_log_lambda_estimates
        )
        projected = projection_function(
            model_output,
            noise_spectral_density,
            reference_latent[0].numel() / 2.0,
        )
        predicted_timestep = projected["timestep"]
        estimated_noise_variance_map = projected["noise_variance"]
        estimated_power_map = projected["projected_power"]
        alpha_bar_map = projected["alpha_bar"]
        raw_alpha_bar_map = projected["raw_alpha_bar"]
        predicted_patch_energy = projected["predicted_patch_energy"]
        if estimator.prediction_mode == "log_lambda":
            normalized_log_lambda_map = projected[
                "normalized_log_lambda"
            ]
            raw_normalized_log_lambda_map = projected[
                "raw_normalized_log_lambda"
            ]
    elif estimator.prediction_mode == "ddim_classification":
        if classification_inference_mode == "raw_logits":
            estimates = estimate_channel_from_ddim_logits(
                model_output["raw_class_logits"],
                noise_spectral_density,
                estimator.ddim_steps,
            )
            class_probabilities = estimates["class_probabilities"]
            predicted_class = estimates["predicted_class"]
            raw_predicted_class = predicted_class
            alpha_bar_map = estimates["alpha_bar"]
            raw_alpha_bar_map = alpha_bar_map
            estimated_noise_variance_map = estimates["noise_variance"]
            estimated_power_map = estimates["power"]
            predicted_patch_energy = None
        else:
            estimates = project_ddim_class_estimates(
                model_output,
                noise_spectral_density,
                reference_latent[0].numel() / 2.0,
                estimator.ddim_steps,
                classification_temperature,
                normalize_projected_distance,
            )
            class_probabilities = estimates["class_logits"].softmax(dim=1)
            predicted_class = class_probabilities.argmax(
                dim=1,
                keepdim=True,
            )
            raw_predicted_class = estimates[
                "raw_class_probabilities"
            ].argmax(dim=1, keepdim=True)
            alpha_bar_map = estimates["alpha_bar"]
            raw_alpha_bar_map = estimates["raw_alpha_bar"]
            estimated_noise_variance_map = estimates["noise_variance"]
            estimated_power_map = estimates["projected_power"]
            predicted_patch_energy = estimates["predicted_patch_energy"]
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            estimator.ddim_steps,
        ).float()
    elif estimator.prediction_mode == "projected_power_classification":
        estimates = project_power_estimates_to_ddim_classes(
            model_output,
            noise_spectral_density,
            reference_latent[0].numel() / 2.0,
            estimator.ddim_steps,
            classification_temperature,
        )
        class_probabilities = estimates["class_probabilities"]
        predicted_class = estimates["predicted_class"]
        raw_predicted_class = predicted_class
        alpha_bar_map = estimates["alpha_bar"]
        raw_alpha_bar_map = alpha_bar_map
        estimated_noise_variance_map = estimates["noise_variance"]
        estimated_power_map = estimates["projected_power"]
        predicted_patch_energy = estimates["predicted_patch_energy"]
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            estimator.ddim_steps,
        ).float()
    else:
        raise ValueError(
            f"Unknown estimator prediction mode: {estimator.prediction_mode}"
        )
    if predicted_timestep.shape[-2:] != (group_rows, group_cols):
        raise ValueError(
            "Timestep estimator output grid does not match power groups: "
            f"expected {(group_rows, group_cols)}, got "
            f"{tuple(predicted_timestep.shape[-2:])}"
        )
    rounded_timestep = predicted_timestep.round().clamp(
        0,
        NUM_TIMESTEPS - 1,
    ).to(torch.long)
    flattened_timestep = rounded_timestep.flatten(1)

    estimated_noise_variance = estimated_noise_variance_map.flatten(1)
    estimated_effective_gain = torch.sqrt(
        estimated_power_map.flatten(1)
    )
    gain_map = estimated_effective_gain.reshape(
        reference_latent.shape[0],
        1,
        group_rows,
        group_cols,
    )
    gain_map = gain_map.repeat_interleave(
        reference_latent.shape[-2] // group_rows,
        dim=2,
    ).repeat_interleave(
        reference_latent.shape[-1] // group_cols,
        dim=3,
    )
    equalized_latent = raw_received_latent / gain_map.to(
        dtype=raw_received_latent.dtype
    )

    oracle_timestep = noise_variance_to_diffusion_timestep(
        channel_info["noise_variance"]
    ).to(torch.long)
    timestep_error = (
        flattened_timestep.float() - oracle_timestep.float()
    ).abs()
    return {
        "received_latent": equalized_latent,
        "noise_variance": estimated_noise_variance,
        "alpha_bar": alpha_bar_map.flatten(1),
        "raw_alpha_bar": raw_alpha_bar_map.flatten(1),
        "normalized_log_lambda": (
            normalized_log_lambda_map.flatten(1)
            if normalized_log_lambda_map is not None
            else None
        ),
        "raw_normalized_log_lambda": (
            raw_normalized_log_lambda_map.flatten(1)
            if raw_normalized_log_lambda_map is not None
            else None
        ),
        "prediction_mode": estimator.prediction_mode,
        "predicted_ddim_class": (
            predicted_class.flatten(1) if predicted_class is not None else None
        ),
        "raw_predicted_ddim_class": (
            raw_predicted_class.flatten(1)
            if raw_predicted_class is not None
            else None
        ),
        "class_probabilities": class_probabilities,
        "statistic_map": statistic_map,
        "predicted_patch_energy": (
            predicted_patch_energy.flatten(1)
            if predicted_patch_energy is not None
            else None
        ),
        "predicted_timestep": predicted_timestep.flatten(1),
        "rounded_timestep": flattened_timestep,
        "oracle_timestep": oracle_timestep,
        "estimated_effective_gain": estimated_effective_gain,
        "timestep_mae": timestep_error.mean(),
        "timestep_rmse": timestep_error.square().mean().sqrt(),
        "within_25": (timestep_error <= 25.0).float().mean(),
    }


def timestep_estimator_result(info: dict | None) -> dict | None:
    if info is None:
        return None
    return {
        "predicted_timestep": info["predicted_timestep"][0].cpu().tolist(),
        "rounded_timestep": info["rounded_timestep"][0].cpu().tolist(),
        "oracle_timestep": info["oracle_timestep"][0].cpu().tolist(),
        "estimated_effective_gain": info[
            "estimated_effective_gain"
        ][0].cpu().tolist(),
        "alpha_bar": info["alpha_bar"][0].cpu().tolist(),
        "raw_alpha_bar": info["raw_alpha_bar"][0].cpu().tolist(),
        "normalized_log_lambda": (
            info["normalized_log_lambda"][0].cpu().tolist()
            if info["normalized_log_lambda"] is not None
            else None
        ),
        "raw_normalized_log_lambda": (
            info["raw_normalized_log_lambda"][0].cpu().tolist()
            if info["raw_normalized_log_lambda"] is not None
            else None
        ),
        "prediction_mode": info["prediction_mode"],
        "predicted_ddim_class": (
            info["predicted_ddim_class"][0].cpu().tolist()
            if info["predicted_ddim_class"] is not None
            else None
        ),
        "raw_predicted_ddim_class": (
            info["raw_predicted_ddim_class"][0].cpu().tolist()
            if info["raw_predicted_ddim_class"] is not None
            else None
        ),
        "timestep_mae": float(info["timestep_mae"].item()),
        "timestep_rmse": float(info["timestep_rmse"].item()),
        "within_25": float(info["within_25"].item()),
    }


def legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--max_timestep", type=int, default=999)
    parser.add_argument("--pilots_per_group", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=25)
    parser.add_argument(
        "--receiver_mode",
        choices=("pilot", "oracle", "estimator"),
        default="pilot",
    )
    parser.add_argument("--timestep_estimator_checkpoint", default=None)
    parser.add_argument(
        "--channel_gain_abs",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    configured_snr = config.get("snr_db", [10.0])
    if isinstance(configured_snr, (int, float)):
        configured_snr = [float(configured_snr)]
    if args.snr_db is not None:
        snr_db = args.snr_db
    elif config.get("snr_sampling") == "uniform_continuous":
        snr_db = 0.5 * (configured_snr[0] + configured_snr[-1])
    else:
        snr_db = configured_snr[0]
    max_timestep = args.max_timestep
    channel_gain_abs = (
        args.channel_gain_abs
        if args.channel_gain_abs is not None
        else config.get("channel_gain_abs", [1.0])
    )
    pilots_per_group = (
        args.pilots_per_group
        if args.pilots_per_group is not None
        else int(config.get("pilots_per_group", 4))
    )
    if args.receiver_mode == "estimator":
        if args.timestep_estimator_checkpoint is None:
            raise ValueError(
                "--timestep_estimator_checkpoint is required when "
                "--receiver_mode estimator"
            )
        if pilots_per_group != 0:
            raise ValueError(
                "estimator receiver is pilot-free; set "
                "--pilots_per_group 0 or use a pilot-free checkpoint"
            )
    output_dir = Path(
        args.output_dir
        or checkpoint_path.parent
        / "inference"
        / (
            f"sample_{args.sample_index:06d}_snr_{snr_db:g}_"
            f"{args.receiver_mode}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    latent_channels = int(vae.config.latent_channels)
    allocator = PowerAllocator(
        latent_channels,
        int(config["num_groups"]),
        int(config["hidden_dim"]),
        sampling=False,
    ).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    _freeze(allocator)

    timestep_estimator = None
    timestep_estimator_path = None
    timestep_estimator_checkpoint = None
    timestep_estimator_config = None
    estimator_classification_temperature = 5.0
    estimator_normalize_projected_distance = False
    estimator_classification_inference_mode = "projected"
    if args.receiver_mode == "estimator":
        timestep_estimator_path = Path(
            args.timestep_estimator_checkpoint
        )
        if timestep_estimator_path.is_dir():
            timestep_estimator_path = timestep_estimator_path / "best.pth"
        timestep_estimator_checkpoint = torch.load(
            timestep_estimator_path,
            map_location="cpu",
            weights_only=False,
        )
        timestep_estimator_config = timestep_estimator_checkpoint["config"]
        if timestep_estimator_config.get("stage") != "timestep_estimator":
            raise ValueError(
                "--timestep_estimator_checkpoint is not an estimator "
                "checkpoint"
            )
        if (
            timestep_estimator_config.get("estimator_version")
            != ESTIMATOR_VERSION
        ):
            raise ValueError(
                "--timestep_estimator_checkpoint uses the old latent-input "
                "estimator"
            )
        if timestep_estimator_config.get("dataset") != "Cityscapes":
            raise ValueError(
                "Segmentation inference requires a Cityscapes timestep "
                "estimator"
            )
        patch_height, patch_width = timestep_estimator_config["patch_shape"]
        estimator_prediction_mode = timestep_estimator_config.get(
            "prediction_mode",
            "alpha_bar",
        )
        estimator_ddim_steps = int(
            timestep_estimator_config.get("ddim_steps", 25)
        )
        estimator_classification_temperature = float(
            timestep_estimator_config.get(
                "classification_temperature",
                0.5
                if estimator_prediction_mode
                == "projected_power_classification"
                else 5.0,
            )
        )
        estimator_normalize_projected_distance = (
            int(
                timestep_estimator_config.get("projected_logit_version")
                or 1
            )
            >= 2
        )
        estimator_classification_inference_mode = (
            timestep_estimator_config.get(
                "classification_inference_mode",
                "projected",
            )
        )
        if (
            estimator_prediction_mode
            in ("ddim_classification", "projected_power_classification")
            and estimator_ddim_steps != args.diffusion_steps
        ):
            raise ValueError(
                "Classification estimator DDIM grid must match "
                "--diffusion_steps"
            )
        timestep_estimator = PatchTimestepEstimator(
            latent_channels,
            int(patch_height),
            int(patch_width),
            int(timestep_estimator_config["hidden_dim"]),
            estimator_prediction_mode,
            estimator_ddim_steps,
            timestep_estimator_config.get("statistic_mode", "basic"),
            timestep_estimator_config.get("input_mode", "statistics"),
        ).to(device)
        timestep_estimator.load_state_dict(
            timestep_estimator_checkpoint["estimator"]
        )
        _freeze(timestep_estimator)

    channel = AnalogAWGNChannel(
        allocator.group_rows,
        allocator.group_cols,
        snr_db,
        channel_gain_abs,
        pilots_per_group,
    ).to(device)
    transport = ActualPowerTransport(
        channel,
        "oracle" if args.receiver_mode == "estimator" else args.receiver_mode,
    )

    dataset = CityscapesSemanticDataset(
        args.data_path or config["data_path"],
        "val",
    )
    image, _ = dataset[args.sample_index]
    image = image.unsqueeze(0).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.inference_mode():
        latent = encode_latents(image, vae)
        clean_reconstruction = _decode_latent(vae, latent)
        snr = torch.full((1,), snr_db, device=device)
        allocated_ratio, allocated_power, allocation_info = (
            allocate_power_with_timestep_floor(
                latent,
                snr,
                allocator,
                channel,
                float(config.get("temperature", 1.0)),
                max_timestep,
                allocator_snr_scale_db=float(
                    config.get("allocator_snr_scale_db", 1.0)
                ),
            )
        )

        unscaled_energy = allocation_info["unscaled_block_energy"]
        frame_channel_uses = int(allocation_info["frame_channel_uses"])
        uniform_value = frame_channel_uses / unscaled_energy.sum(
            dim=1,
            keepdim=True,
        )
        uniform_power = uniform_value.expand_as(allocated_power)
        uniform_ratio = (
            uniform_power * unscaled_energy / frame_channel_uses
        )

        channel_state = generator.get_state()
        (
            allocated_received,
            allocated_channel_info,
            allocated_transport_info,
        ) = transport(
            latent,
            allocated_power,
            snr,
            generator,
        )
        generator.set_state(channel_state)
        (
            uniform_received,
            uniform_channel_info,
            uniform_transport_info,
        ) = transport(
            latent,
            uniform_power,
            snr,
            generator,
            pool_pilots=True,
        )

        allocated_estimator_info = None
        uniform_estimator_info = None
        allocated_matching_noise_variance = allocated_channel_info[
            "noise_variance"
        ]
        uniform_matching_noise_variance = uniform_channel_info[
            "noise_variance"
        ]
        if args.receiver_mode == "estimator":
            assert timestep_estimator is not None
            allocated_estimator_info = timestep_estimator_receive(
                latent,
                allocated_channel_info,
                allocated_transport_info,
                timestep_estimator,
                channel.group_rows,
                channel.group_cols,
                channel.kappa_m,
                estimator_classification_temperature,
                estimator_normalize_projected_distance,
                estimator_classification_inference_mode,
            )
            uniform_estimator_info = timestep_estimator_receive(
                latent,
                uniform_channel_info,
                uniform_transport_info,
                timestep_estimator,
                channel.group_rows,
                channel.group_cols,
                channel.kappa_m,
                estimator_classification_temperature,
                estimator_normalize_projected_distance,
                estimator_classification_inference_mode,
            )
            allocated_received = allocated_estimator_info[
                "received_latent"
            ]
            uniform_received = uniform_estimator_info["received_latent"]
            allocated_matching_noise_variance = allocated_estimator_info[
                "noise_variance"
            ]
            uniform_matching_noise_variance = uniform_estimator_info[
                "noise_variance"
            ]

        matching_generator = torch.Generator(device=device)
        matching_generator.manual_seed(args.seed + 1)
        matching_epsilon = torch.randn(
            latent.shape,
            device=device,
            dtype=latent.dtype,
            generator=matching_generator,
        )
        diffusion_model = (
            args.diffusion_model
            or config.get("vae_model_id", VAE_MODEL_ID)
        )
        reverse_model = StaggeredReverseDiffusion(
            diffusion_model,
            device,
            torch.float32,
            args.local_files_only,
            revision=(
                config.get("vae_model_revision", VAE_MODEL_REVISION)
                if args.diffusion_model is None
                else None
            ),
            num_inference_steps=args.diffusion_steps,
        )
        ddim_timesteps = reverse_model.inference_timesteps(device)
        allocated_start, allocated_context, allocated_matching = (
            _match_diffusion_start(
                allocated_received,
                allocated_matching_noise_variance,
                channel,
                matching_epsilon,
                ddim_timesteps,
            )
        )
        uniform_start, uniform_context, uniform_matching = (
            _match_diffusion_start(
                uniform_received,
                uniform_matching_noise_variance,
                channel,
                matching_epsilon,
                ddim_timesteps,
            )
        )

        allocated_direct = _decode_latent(vae, allocated_received)
        uniform_direct = _decode_latent(vae, uniform_received)

        diffusion_generator = torch.Generator(device=device)
        diffusion_generator.manual_seed(args.seed + 2)
        diffusion_state = diffusion_generator.get_state()
        allocated_denoised_latent = reverse_model(
            allocated_start,
            allocated_context,
            diffusion_generator,
        )
        diffusion_generator.set_state(diffusion_state)
        uniform_denoised_latent = reverse_model(
            uniform_start,
            uniform_context,
            diffusion_generator,
        )
        allocated_denoised = _decode_latent(
            vae,
            allocated_denoised_latent,
        )
        uniform_denoised = _decode_latent(
            vae,
            uniform_denoised_latent,
        )

    original = clean_reconstruction[0].float().cpu()
    to_pil_image(original).save(output_dir / "input.png")
    to_pil_image(allocated_direct[0].float().cpu()).save(
        output_dir / "allocated_received_direct.png"
    )
    to_pil_image(uniform_direct[0].float().cpu()).save(
        output_dir / "uniform_received_direct.png"
    )
    to_pil_image(allocated_denoised[0].float().cpu()).save(
        output_dir / "allocated_denoised.png"
    )
    to_pil_image(uniform_denoised[0].float().cpu()).save(
        output_dir / "uniform_denoised.png"
    )
    _save_labeled_row(
        [
            original,
            allocated_direct[0].float().cpu(),
            allocated_denoised[0].float().cpu(),
        ],
        [
            "VAE reconstruction",
            "Learned received",
            "Learned diffusion denoised",
        ],
        output_dir / "allocated_denoising_comparison.png",
    )
    _save_labeled_row(
        [
            original,
            uniform_direct[0].float().cpu(),
            uniform_denoised[0].float().cpu(),
        ],
        [
            "VAE reconstruction",
            "Uniform received",
            "Uniform diffusion denoised",
        ],
        output_dir / "uniform_denoising_comparison.png",
    )
    _save_labeled_row(
        [
            original,
            allocated_denoised[0].float().cpu(),
            uniform_denoised[0].float().cpu(),
        ],
        ["VAE reconstruction", "Learned diffusion", "Uniform diffusion"],
        output_dir / "denoised_comparison.png",
    )
    allocated_timesteps = torch.tensor(
        allocated_matching["schedule_timestep"],
    )
    save_diffusion_timestep_overlay(
        original,
        allocated_timesteps,
        allocator.group_rows,
        allocator.group_cols,
        output_dir / "diffusion_timestep_overlay.png",
    )
    if allocated_estimator_info is not None:
        save_diffusion_timestep_overlay(
            original,
            allocated_estimator_info["rounded_timestep"][0],
            allocator.group_rows,
            allocator.group_cols,
            output_dir / "estimated_timestep_overlay.png",
        )
        save_diffusion_timestep_overlay(
            original,
            allocated_estimator_info["oracle_timestep"][0],
            allocator.group_rows,
            allocator.group_cols,
            output_dir / "oracle_timestep_overlay.png",
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]) + 1,
        "sample_index": args.sample_index,
        "snr_db": float(snr_db),
        "max_timestep_power_floor": max_timestep,
        "pilots_per_group": pilots_per_group,
        "seed": args.seed,
        "receiver_mode": args.receiver_mode,
        "timestep_estimator_checkpoint": (
            str(timestep_estimator_path)
            if timestep_estimator_path is not None
            else None
        ),
        "timestep_estimator_epoch": (
            int(timestep_estimator_checkpoint["epoch"]) + 1
            if timestep_estimator_checkpoint is not None
            else None
        ),
        "same_channel_noise_for_strategies": True,
        "same_diffusion_noise_for_strategies": True,
        "diffusion_model": diffusion_model,
        "diffusion_steps": args.diffusion_steps,
        "ddim_timesteps": ddim_timesteps.cpu().tolist(),
        "strategies": {
            "allocated": {
                "pilot_pooling": (
                    "timestep_estimator"
                    if allocated_estimator_info is not None
                    else allocated_transport_info["pilot_pooling"]
                ),
                "group_power": allocated_power[0].cpu().tolist(),
                "minimum_group_power": allocation_info[
                    "minimum_group_power"
                ][0].cpu().tolist(),
                "pilot_uncertainty_factor": allocation_info[
                    "pilot_uncertainty_factor"
                ][0].cpu().tolist(),
                "energy_allocation_ratio": (
                    allocated_ratio[0].cpu().tolist()
                ),
                "noise_variance": allocated_matching_noise_variance[
                    0
                ].cpu().tolist(),
                "oracle_noise_variance": (
                    allocated_channel_info["noise_variance"][0]
                    .cpu()
                    .tolist()
                    if allocated_estimator_info is not None
                    else None
                ),
                "timestep_estimator": timestep_estimator_result(
                    allocated_estimator_info
                ),
                "diffusion": allocated_matching,
            },
            "uniform": {
                "pilot_pooling": (
                    "timestep_estimator"
                    if uniform_estimator_info is not None
                    else uniform_transport_info["pilot_pooling"]
                ),
                "group_power": uniform_power[0].cpu().tolist(),
                "energy_allocation_ratio": uniform_ratio[0].cpu().tolist(),
                "noise_variance": uniform_matching_noise_variance[
                    0
                ].cpu().tolist(),
                "oracle_noise_variance": (
                    uniform_channel_info["noise_variance"][0]
                    .cpu()
                    .tolist()
                    if uniform_estimator_info is not None
                    else None
                ),
                "timestep_estimator": timestep_estimator_result(
                    uniform_estimator_info
                ),
                "diffusion": uniform_matching,
            },
        },
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Saved segmentation allocator diffusion output to {output_dir}")


def main() -> None:
    from segmentation_orthogonal_power import dispatch
    dispatch("inference", legacy_main)


if __name__ == "__main__":
    main()
