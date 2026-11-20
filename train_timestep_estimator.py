import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.data_processing import (
    construct_cityscapes_dataloaders,
    construct_dataloaders,
)
from train_power_allocator import (
    PowerAllocator,
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    VAE_SCALE,
    build_logger,
    cleanup_distributed,
    encode_latents,
    initialize_distributed,
    install_termination_handlers,
    save_config,
    unwrap_model,
)
from power_utils import sd15_alpha_bars


NUM_TIMESTEPS = 1000
ESTIMATOR_VERSION = 2
CLASSIFICATION_LOSS_VERSION = 5
PROJECTED_LOGIT_VERSION = 2
PROJECTED_POWER_CLASSIFICATION_VERSION = 2
PREDICTION_MODES = (
    "alpha_bar",
    "log_lambda",
    "ddim_classification",
    "projected_power_classification",
)
BASIC_PATCH_STATISTIC_NAMES = (
    "mean",
    "variance",
    "rms",
    "mean_absolute",
    "lag1_correlation",
)
ENHANCED_CHANNEL_STATISTIC_NAMES = (
    "mean",
    "variance",
    "rms",
    "mean_absolute",
    "skewness",
    "excess_kurtosis",
    "horizontal_lag1_correlation",
    "vertical_lag1_correlation",
    "horizontal_lag2_correlation",
    "vertical_lag2_correlation",
    "horizontal_gradient_rms",
    "vertical_gradient_rms",
    "laplacian_rms",
    "relative_variance",
    "relative_rms",
    "relative_mean_absolute",
)
STATISTIC_MODES = ("basic", "enhanced")
INPUT_MODES = ("statistics", "raw")
POWER_MAP_SOURCES = ("random", "allocator")
# Backward-compatible name used by older imports and checkpoints.
PATCH_STATISTIC_NAMES = BASIC_PATCH_STATISTIC_NAMES


def statistic_channel_names(
    statistic_mode: str,
    latent_channels: int,
) -> tuple[str, ...]:
    if statistic_mode not in STATISTIC_MODES:
        raise ValueError(f"Unknown statistic_mode: {statistic_mode}")
    per_channel_names = (
        BASIC_PATCH_STATISTIC_NAMES
        if statistic_mode == "basic"
        else ENHANCED_CHANNEL_STATISTIC_NAMES
    )
    names = [
        f"channel_{channel}_{name}"
        for channel in range(latent_channels)
        for name in per_channel_names
    ]
    if statistic_mode == "enhanced":
        for first_channel in range(latent_channels):
            for second_channel in range(first_channel + 1, latent_channels):
                pair = f"channel_{first_channel}_{second_channel}"
                names.extend(
                    [
                        f"{pair}_covariance",
                        f"{pair}_correlation",
                    ]
                )
    return tuple(names)


DEFAULT_DATA_PATHS = {
    "Cityscapes": "/data/bsy/",
    "ImageNet": "/dataset/ImageNet",
}


def load_frozen_power_allocator(
    checkpoint_path: str | Path,
    latent_channels: int,
    device: torch.device,
) -> tuple[nn.Module, dict, str]:
    resolved_path = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint = torch.load(
        resolved_path,
        map_location="cpu",
        weights_only=False,
    )
    if "allocator" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Power allocator checkpoint must contain allocator and config"
        )
    config = checkpoint["config"]
    if "num_groups" not in config:
        raise ValueError(
            "Power allocator checkpoint config is missing num_groups"
        )
    allocator = PowerAllocator(
        latent_channels,
        int(config["num_groups"]),
        int(config.get("hidden_dim", 64)),
        sampling=False,
    ).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    allocator.eval()
    for parameter in allocator.parameters():
        parameter.requires_grad_(False)
    return allocator, config, resolved_path


class StatisticResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.silu(features + self.block(features), inplace=True)


class PatchTimestepEstimator(nn.Module):
    """Estimate a globally coupled timestep map from receiver observations."""

    def __init__(
        self,
        latent_channels: int,
        patch_height: int = 8,
        patch_width: int = 8,
        hidden_dim: int = 64,
        prediction_mode: str = "alpha_bar",
        ddim_steps: int = 25,
        statistic_mode: str = "basic",
        input_mode: str = "statistics",
    ) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("patch dimensions must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"Unknown prediction_mode: {prediction_mode}")
        if statistic_mode not in STATISTIC_MODES:
            raise ValueError(f"Unknown statistic_mode: {statistic_mode}")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        if not 1 <= ddim_steps <= NUM_TIMESTEPS:
            raise ValueError("ddim_steps must be between 1 and 1000")

        self.latent_channels = latent_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.hidden_dim = hidden_dim
        self.prediction_mode = prediction_mode
        self.ddim_steps = ddim_steps
        self.statistic_mode = statistic_mode
        self.input_mode = input_mode
        self.statistic_names = statistic_channel_names(
            statistic_mode,
            latent_channels,
        )
        self.statistic_channels = len(self.statistic_names)
        if input_mode == "statistics":
            self.stem = nn.Sequential(
                nn.Conv2d(
                    self.statistic_channels,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    latent_channels,
                    2 * hidden_dim,
                    kernel_size=(patch_height, patch_width),
                    stride=(patch_height, patch_width),
                ),
                nn.GroupNorm(1, 2 * hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(inplace=True),
            )
        self.spatial_backbone = nn.Sequential(
            StatisticResidualBlock(hidden_dim),
            StatisticResidualBlock(hidden_dim),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        if prediction_mode == "alpha_bar":
            self.alpha_bar_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.alpha_bar_head.weight)
            initial_alpha_bar = float(sd15_alpha_bars()[NUM_TIMESTEPS // 2])
            initial_alpha_logit = math.log(
                initial_alpha_bar / (1.0 - initial_alpha_bar)
            )
            nn.init.constant_(self.alpha_bar_head.bias, initial_alpha_logit)
        elif prediction_mode == "log_lambda":
            self.normalized_log_lambda_head = nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=1,
            )
            nn.init.zeros_(self.normalized_log_lambda_head.weight)
            alpha_bars = sd15_alpha_bars().float()
            log_lambda = torch.log((1.0 - alpha_bars) / alpha_bars)
            initial_value = (
                log_lambda[NUM_TIMESTEPS // 2] - log_lambda[0]
            ) / (log_lambda[-1] - log_lambda[0])
            initial_logit = torch.logit(
                initial_value.clamp(1e-6, 1.0 - 1e-6)
            ).item()
            nn.init.constant_(
                self.normalized_log_lambda_head.bias,
                initial_logit,
            )
        elif prediction_mode == "ddim_classification":
            self.class_head = nn.Conv2d(
                hidden_dim,
                ddim_steps,
                kernel_size=1,
            )
        else:
            self.log_power_head = nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(self.log_power_head.weight)
        self.log_energy_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.log_energy_head.weight)
        initial_energy = latent_channels * patch_height * patch_width
        nn.init.constant_(
            self.log_energy_head.bias,
            math.log(initial_energy),
        )
        if prediction_mode == "ddim_classification":
            self.log_energy_head.requires_grad_(False)

    def forward(self, estimator_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if estimator_input.ndim != 4:
            raise ValueError("estimator_input must be four-dimensional")
        expected_channels = (
            self.statistic_channels
            if self.input_mode == "statistics"
            else self.latent_channels
        )
        if estimator_input.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels, "
                f"got {estimator_input.shape[1]}"
            )
        if self.input_mode == "raw" and (
            estimator_input.shape[2] % self.patch_height
            or estimator_input.shape[3] % self.patch_width
        ):
            raise ValueError(
                "Raw estimator input dimensions must be divisible by the "
                "patch size"
            )
        features = self.spatial_backbone(self.stem(estimator_input))
        features = features + self.global_context(features)
        if self.prediction_mode == "alpha_bar":
            return {
                "raw_alpha_bar": torch.sigmoid(
                    self.alpha_bar_head(features)
                ),
                "predicted_log_patch_energy": self.log_energy_head(features),
            }
        if self.prediction_mode == "log_lambda":
            return {
                "raw_normalized_log_lambda": torch.sigmoid(
                    self.normalized_log_lambda_head(features)
                ),
                "predicted_log_patch_energy": self.log_energy_head(features),
            }
        if self.prediction_mode == "ddim_classification":
            return {
                "raw_class_logits": self.class_head(features),
                "predicted_log_patch_energy": self.log_energy_head(features),
            }
        return {
            "raw_log_power": self.log_power_head(features),
            "predicted_log_patch_energy": self.log_energy_head(features),
        }


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * torch.log1p(value.abs())


def build_patch_statistic_map(
    normalized_received_latent: torch.Tensor,
    patch_height: int,
    patch_width: int,
    statistic_mode: str = "basic",
) -> torch.Tensor:
    """Convert y/sqrt(N0/2) into a per-patch statistic map."""
    if normalized_received_latent.ndim != 4:
        raise ValueError(
            "normalized_received_latent must have shape [B, C, H, W]"
        )
    if statistic_mode not in STATISTIC_MODES:
        raise ValueError(f"Unknown statistic_mode: {statistic_mode}")
    batch_size, channels, height, width = normalized_received_latent.shape
    if height % patch_height or width % patch_width:
        raise ValueError(
            "Latent spatial dimensions must be divisible by the patch size"
        )
    grid_height = height // patch_height
    grid_width = width // patch_width
    patch_area = patch_height * patch_width
    normalized_received_latent = normalized_received_latent.float()
    patches = F.unfold(
        normalized_received_latent,
        kernel_size=(patch_height, patch_width),
        stride=(patch_height, patch_width),
    ).reshape(batch_size, channels, patch_area, -1)

    mean = patches.mean(dim=2)
    variance = patches.var(dim=2, unbiased=False)
    rms = patches.square().mean(dim=2).sqrt()
    mean_absolute = patches.abs().mean(dim=2)
    spatial_patches = patches.reshape(
        batch_size,
        channels,
        patch_height,
        patch_width,
        -1,
    )

    if statistic_mode == "basic":
        correlations = []
        if patch_width > 1:
            correlations.append(
                (
                    spatial_patches[:, :, :, 1:, :]
                    * spatial_patches[:, :, :, :-1, :]
                ).mean(dim=(2, 3))
            )
        if patch_height > 1:
            correlations.append(
                (
                    spatial_patches[:, :, 1:, :, :]
                    * spatial_patches[:, :, :-1, :, :]
                ).mean(dim=(2, 3))
            )
        if correlations:
            lag1_correlation = torch.stack(correlations, dim=0).mean(dim=0)
        else:
            lag1_correlation = mean.square()

        statistics = torch.stack(
            [
                _signed_log1p(mean),
                torch.log1p(variance),
                torch.log1p(rms),
                torch.log1p(mean_absolute),
                _signed_log1p(lag1_correlation),
            ],
            dim=2,
        )
        return statistics.reshape(
            batch_size,
            channels * len(BASIC_PATCH_STATISTIC_NAMES),
            grid_height,
            grid_width,
        )

    epsilon = 1e-6
    centered = patches - mean.unsqueeze(2)
    standard_deviation = variance.clamp_min(epsilon).sqrt()
    standardized = centered / standard_deviation.unsqueeze(2)
    skewness = standardized.pow(3).mean(dim=2)
    excess_kurtosis = standardized.pow(4).mean(dim=2) - 3.0
    centered_spatial = centered.reshape_as(spatial_patches)

    def spatial_correlation(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        covariance = (first * second).mean(dim=(2, 3))
        return (
            covariance / variance.clamp_min(epsilon)
        ).clamp(-1.0, 1.0)

    zero_statistic = torch.zeros_like(mean)
    horizontal_lag1 = (
        spatial_correlation(
            centered_spatial[:, :, :, 1:, :],
            centered_spatial[:, :, :, :-1, :],
        )
        if patch_width > 1
        else zero_statistic
    )
    vertical_lag1 = (
        spatial_correlation(
            centered_spatial[:, :, 1:, :, :],
            centered_spatial[:, :, :-1, :, :],
        )
        if patch_height > 1
        else zero_statistic
    )
    horizontal_lag2 = (
        spatial_correlation(
            centered_spatial[:, :, :, 2:, :],
            centered_spatial[:, :, :, :-2, :],
        )
        if patch_width > 2
        else zero_statistic
    )
    vertical_lag2 = (
        spatial_correlation(
            centered_spatial[:, :, 2:, :, :],
            centered_spatial[:, :, :-2, :, :],
        )
        if patch_height > 2
        else zero_statistic
    )
    horizontal_gradient_rms = (
        (
            spatial_patches[:, :, :, 1:, :]
            - spatial_patches[:, :, :, :-1, :]
        ).square().mean(dim=(2, 3)).sqrt()
        if patch_width > 1
        else zero_statistic
    )
    vertical_gradient_rms = (
        (
            spatial_patches[:, :, 1:, :, :]
            - spatial_patches[:, :, :-1, :, :]
        ).square().mean(dim=(2, 3)).sqrt()
        if patch_height > 1
        else zero_statistic
    )
    if patch_height > 2 and patch_width > 2:
        center = spatial_patches[:, :, 1:-1, 1:-1, :]
        laplacian = (
            4.0 * center
            - spatial_patches[:, :, :-2, 1:-1, :]
            - spatial_patches[:, :, 2:, 1:-1, :]
            - spatial_patches[:, :, 1:-1, :-2, :]
            - spatial_patches[:, :, 1:-1, 2:, :]
        )
        laplacian_rms = laplacian.square().mean(dim=(2, 3)).sqrt()
    else:
        laplacian_rms = zero_statistic

    frame_variance = normalized_received_latent.var(
        dim=(2, 3),
        unbiased=False,
    )
    frame_rms = normalized_received_latent.square().mean(
        dim=(2, 3)
    ).sqrt()
    frame_mean_absolute = normalized_received_latent.abs().mean(dim=(2, 3))
    relative_variance = torch.log(
        (variance + epsilon) / (frame_variance[:, :, None] + epsilon)
    ).clamp(-10.0, 10.0)
    relative_rms = torch.log(
        (rms + epsilon) / (frame_rms[:, :, None] + epsilon)
    ).clamp(-10.0, 10.0)
    relative_mean_absolute = torch.log(
        (mean_absolute + epsilon)
        / (frame_mean_absolute[:, :, None] + epsilon)
    ).clamp(-10.0, 10.0)

    channel_statistics = torch.stack(
        [
            _signed_log1p(mean),
            torch.log1p(variance),
            torch.log1p(rms),
            torch.log1p(mean_absolute),
            _signed_log1p(skewness),
            _signed_log1p(excess_kurtosis),
            horizontal_lag1,
            vertical_lag1,
            horizontal_lag2,
            vertical_lag2,
            torch.log1p(horizontal_gradient_rms),
            torch.log1p(vertical_gradient_rms),
            torch.log1p(laplacian_rms),
            relative_variance,
            relative_rms,
            relative_mean_absolute,
        ],
        dim=2,
    ).reshape(
        batch_size,
        channels * len(ENHANCED_CHANNEL_STATISTIC_NAMES),
        grid_height,
        grid_width,
    )

    cross_channel_statistics = []
    for first_channel in range(channels):
        for second_channel in range(first_channel + 1, channels):
            covariance = (
                centered[:, first_channel]
                * centered[:, second_channel]
            ).mean(dim=1)
            correlation = covariance / (
                standard_deviation[:, first_channel]
                * standard_deviation[:, second_channel]
            ).clamp_min(epsilon)
            cross_channel_statistics.extend(
                [
                    _signed_log1p(covariance),
                    correlation.clamp(-1.0, 1.0),
                ]
            )
    if cross_channel_statistics:
        cross_channel_map = torch.stack(
            cross_channel_statistics,
            dim=1,
        ).reshape(batch_size, -1, grid_height, grid_width)
        return torch.cat([channel_statistics, cross_channel_map], dim=1)
    return channel_statistics


def sample_uniform_snr_db(
    batch_size: int,
    device: torch.device,
    minimum: float,
    maximum: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if minimum == maximum:
        return torch.full(
            (batch_size,),
            minimum,
            device=device,
            dtype=torch.float32,
        )
    return minimum + (maximum - minimum) * torch.rand(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def oracle_gain_to_timestep(
    effective_gain: torch.Tensor,
    noise_spectral_density: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same oracle gain-to-timestep rule as inference_power.py."""
    if effective_gain.ndim != 2:
        raise ValueError("effective_gain must have shape [B, num_patches]")
    if noise_spectral_density.shape != (effective_gain.shape[0],):
        raise ValueError("noise_spectral_density must have shape [B]")

    noise_variance = noise_spectral_density[:, None] / (
        2.0 * effective_gain.square().clamp_min(1e-20)
    )
    alpha_bars = sd15_alpha_bars().to(
        device=effective_gain.device,
        dtype=torch.float32,
    )
    mathematical_alpha_bars = torch.cat(
        [alpha_bars.new_ones(1), alpha_bars]
    )
    lambda_schedule = (
        (1.0 - mathematical_alpha_bars) / mathematical_alpha_bars
    )
    mathematical_timestep = torch.searchsorted(
        lambda_schedule,
        noise_variance.float().clamp_max(lambda_schedule[-1]).contiguous(),
        right=False,
    )
    timestep = (mathematical_timestep - 1).clamp(
        min=0,
        max=NUM_TIMESTEPS - 1,
    )
    return timestep, noise_variance


def ddim_schedule_timesteps(
    ddim_steps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the ascending trailing-spaced DDIM grid used by inference."""
    if not 1 <= ddim_steps <= NUM_TIMESTEPS:
        raise ValueError("ddim_steps must be between 1 and 1000")
    step_ratio = NUM_TIMESTEPS / ddim_steps
    descending = (
        torch.arange(
            ddim_steps,
            0,
            -1,
            device=device,
            dtype=torch.float64,
        )
        * step_ratio
    ).round().to(torch.long) - 1
    return torch.unique(descending, sorted=True)


def timestep_to_ddim_class(
    timestep: torch.Tensor,
    ddim_steps: int,
) -> torch.Tensor:
    schedule = ddim_schedule_timesteps(ddim_steps, timestep.device)
    class_index = torch.searchsorted(
        schedule,
        timestep.to(torch.long).clamp_min(0).contiguous(),
        right=False,
    )
    return class_index.clamp_max(schedule.numel() - 1)


def ddim_class_to_timestep(
    class_index: torch.Tensor,
    ddim_steps: int,
) -> torch.Tensor:
    schedule = ddim_schedule_timesteps(ddim_steps, class_index.device)
    return schedule[class_index.to(torch.long)]


def ddim_class_representative_noise_variance(
    ddim_steps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the log-midpoint noise variance for each DDIM class."""
    upper_timestep = ddim_schedule_timesteps(ddim_steps, device)
    lower_timestep = torch.cat(
        [
            upper_timestep.new_zeros(1),
            upper_timestep[:-1] + 1,
        ]
    )
    alpha_bars = sd15_alpha_bars().to(
        device=upper_timestep.device,
        dtype=torch.float32,
    )
    noise_variance = (1.0 - alpha_bars) / alpha_bars
    return torch.sqrt(
        noise_variance[lower_timestep] * noise_variance[upper_timestep]
    )


def estimate_channel_from_ddim_logits(
    class_logits: torch.Tensor,
    noise_spectral_density: torch.Tensor,
    ddim_steps: int,
) -> dict[str, torch.Tensor]:
    """Convert raw class probabilities directly to channel quantities."""
    class_logits = class_logits.float()
    if class_logits.shape[1] != ddim_steps:
        raise ValueError("class_logits channel count must match ddim_steps")
    if noise_spectral_density.shape != (class_logits.shape[0],):
        raise ValueError("noise_spectral_density must have shape [B]")
    class_probabilities = class_logits.softmax(dim=1)
    representative_noise_variance = (
        ddim_class_representative_noise_variance(
            ddim_steps,
            class_logits.device,
        )
    )
    class_unit_power = 1.0 / (
        2.0 * representative_noise_variance.clamp_min(1e-20)
    )
    expected_unit_power = (
        class_probabilities * class_unit_power[None, :, None, None]
    ).sum(dim=1, keepdim=True)
    estimated_power = noise_spectral_density[:, None, None, None].float() * (
        expected_unit_power
    )
    estimated_noise_variance = noise_spectral_density[
        :, None, None, None
    ].float() / (2.0 * estimated_power.clamp_min(1e-20))
    return {
        "class_logits": class_logits,
        "class_probabilities": class_probabilities,
        "predicted_class": class_logits.argmax(dim=1, keepdim=True),
        "power": estimated_power,
        "noise_variance": estimated_noise_variance,
        "alpha_bar": 1.0 / (1.0 + estimated_noise_variance),
        "continuous_timestep": noise_variance_to_continuous_timestep(
            estimated_noise_variance
        ),
    }


def noise_variance_to_continuous_timestep(
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    """Invert the diffusion variance schedule with log-space interpolation."""
    alpha_bars = sd15_alpha_bars().to(
        device=noise_variance.device,
        dtype=torch.float32,
    )
    log_schedule = torch.log((1.0 - alpha_bars) / alpha_bars)
    log_variance = torch.log(noise_variance.float().clamp_min(1e-20))
    bounded = log_variance.clamp(log_schedule[0], log_schedule[-1])
    upper_index = torch.searchsorted(
        log_schedule,
        bounded.contiguous(),
        right=False,
    ).clamp(max=NUM_TIMESTEPS - 1)
    lower_index = (upper_index - 1).clamp_min(0)
    lower = log_schedule[lower_index]
    upper = log_schedule[upper_index]
    fraction = (bounded - lower) / (upper - lower).clamp_min(1e-12)
    timestep = lower_index.float() + fraction
    timestep = torch.where(
        upper_index == lower_index,
        upper_index.float(),
        timestep,
    )
    return timestep.clamp(0.0, NUM_TIMESTEPS - 1)


def noise_variance_to_normalized_log_lambda(
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    """Map lambda=(1-alpha_bar)/alpha_bar to a normalized log scale."""
    alpha_bars = sd15_alpha_bars().to(
        device=noise_variance.device,
        dtype=torch.float32,
    )
    log_schedule = torch.log((1.0 - alpha_bars) / alpha_bars)
    log_variance = torch.log(noise_variance.float().clamp_min(1e-20))
    return (log_variance - log_schedule[0]) / (
        log_schedule[-1] - log_schedule[0]
    )


def normalized_log_lambda_to_noise_variance(
    normalized_log_lambda: torch.Tensor,
) -> torch.Tensor:
    """Invert the normalized log-lambda regression target."""
    alpha_bars = sd15_alpha_bars().to(
        device=normalized_log_lambda.device,
        dtype=torch.float32,
    )
    log_schedule = torch.log((1.0 - alpha_bars) / alpha_bars)
    log_variance = log_schedule[0] + normalized_log_lambda.float() * (
        log_schedule[-1] - log_schedule[0]
    )
    return torch.exp(log_variance)


def project_alpha_bar_estimates(
    model_output: dict[str, torch.Tensor],
    noise_spectral_density: torch.Tensor,
    target_frame_energy: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Project raw alpha-bar estimates onto the frame-energy budget."""
    raw_alpha_bar = model_output["raw_alpha_bar"].float().clamp(
        min=1e-6,
        max=1.0 - 1e-6,
    )
    predicted_log_energy = model_output[
        "predicted_log_patch_energy"
    ].float()
    if noise_spectral_density.shape != (raw_alpha_bar.shape[0],):
        raise ValueError("noise_spectral_density must have shape [B]")
    raw_noise_variance = (1.0 - raw_alpha_bar) / raw_alpha_bar
    noise_spectral_density = noise_spectral_density.float()
    raw_power = noise_spectral_density[:, None, None, None] / (
        2.0 * raw_noise_variance
    )
    predicted_patch_energy = torch.exp(
        predicted_log_energy.clamp(min=-12.0, max=16.0)
    )
    target_frame_energy = torch.as_tensor(
        target_frame_energy,
        device=raw_power.device,
        dtype=torch.float32,
    )
    if target_frame_energy.ndim == 0:
        target_frame_energy = target_frame_energy.expand(raw_power.shape[0])
    if target_frame_energy.shape != (raw_power.shape[0],):
        raise ValueError("target_frame_energy must be scalar or shape [B]")
    raw_estimated_energy = (
        raw_power * predicted_patch_energy
    ).flatten(1).sum(dim=1)
    power_scale = target_frame_energy / raw_estimated_energy.clamp_min(1e-12)
    projected_power = raw_power * power_scale[:, None, None, None]
    projected_noise_variance = noise_spectral_density[
        :, None, None, None
    ] / (2.0 * projected_power.clamp_min(1e-20))
    projected_timestep = noise_variance_to_continuous_timestep(
        projected_noise_variance
    )
    projected_alpha_bar = 1.0 / (1.0 + projected_noise_variance)
    return {
        "raw_alpha_bar": raw_alpha_bar,
        "raw_timestep": noise_variance_to_continuous_timestep(
            raw_noise_variance
        ),
        "predicted_log_patch_energy": predicted_log_energy,
        "predicted_patch_energy": predicted_patch_energy,
        "raw_power": raw_power,
        "power_scale": power_scale,
        "projected_power": projected_power,
        "noise_variance": projected_noise_variance,
        "alpha_bar": projected_alpha_bar,
        "timestep": projected_timestep,
        "normalized_timestep": projected_timestep / (NUM_TIMESTEPS - 1),
        "estimated_frame_energy": (
            projected_power * predicted_patch_energy
        ).flatten(1).sum(dim=1),
    }


def project_log_lambda_estimates(
    model_output: dict[str, torch.Tensor],
    noise_spectral_density: torch.Tensor,
    target_frame_energy: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Project normalized log-lambda estimates onto the frame-energy budget."""
    raw_normalized_log_lambda = model_output[
        "raw_normalized_log_lambda"
    ].float().clamp(min=1e-6, max=1.0 - 1e-6)
    predicted_log_energy = model_output[
        "predicted_log_patch_energy"
    ].float()
    if noise_spectral_density.shape != (
        raw_normalized_log_lambda.shape[0],
    ):
        raise ValueError("noise_spectral_density must have shape [B]")

    raw_noise_variance = normalized_log_lambda_to_noise_variance(
        raw_normalized_log_lambda
    )
    noise_spectral_density = noise_spectral_density.float()
    raw_power = noise_spectral_density[:, None, None, None] / (
        2.0 * raw_noise_variance
    )
    predicted_patch_energy = torch.exp(
        predicted_log_energy.clamp(min=-12.0, max=16.0)
    )
    target_frame_energy = torch.as_tensor(
        target_frame_energy,
        device=raw_power.device,
        dtype=torch.float32,
    )
    if target_frame_energy.ndim == 0:
        target_frame_energy = target_frame_energy.expand(raw_power.shape[0])
    if target_frame_energy.shape != (raw_power.shape[0],):
        raise ValueError("target_frame_energy must be scalar or shape [B]")

    raw_estimated_energy = (
        raw_power * predicted_patch_energy
    ).flatten(1).sum(dim=1)
    power_scale = target_frame_energy / raw_estimated_energy.clamp_min(1e-12)
    projected_power = raw_power * power_scale[:, None, None, None]
    projected_noise_variance = noise_spectral_density[
        :, None, None, None
    ] / (2.0 * projected_power.clamp_min(1e-20))
    projected_timestep = noise_variance_to_continuous_timestep(
        projected_noise_variance
    )
    projected_normalized_log_lambda = (
        noise_variance_to_normalized_log_lambda(
            projected_noise_variance
        )
    )
    raw_alpha_bar = 1.0 / (1.0 + raw_noise_variance)
    projected_alpha_bar = 1.0 / (1.0 + projected_noise_variance)
    return {
        "raw_normalized_log_lambda": raw_normalized_log_lambda,
        "normalized_log_lambda": projected_normalized_log_lambda,
        "raw_alpha_bar": raw_alpha_bar,
        "raw_timestep": noise_variance_to_continuous_timestep(
            raw_noise_variance
        ),
        "predicted_log_patch_energy": predicted_log_energy,
        "predicted_patch_energy": predicted_patch_energy,
        "raw_power": raw_power,
        "power_scale": power_scale,
        "projected_power": projected_power,
        "noise_variance": projected_noise_variance,
        "alpha_bar": projected_alpha_bar,
        "timestep": projected_timestep,
        "normalized_timestep": projected_timestep / (NUM_TIMESTEPS - 1),
        "estimated_frame_energy": (
            projected_power * predicted_patch_energy
        ).flatten(1).sum(dim=1),
    }


def continuous_timestep_to_ddim_logits(
    timestep: torch.Tensor,
    ddim_steps: int,
    temperature: float = 0.5,
    normalize_timestep_distance: bool = True,
) -> torch.Tensor:
    """Turn a continuous timestep map into differentiable DDIM-class logits."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    schedule = ddim_schedule_timesteps(ddim_steps, timestep.device)
    upper_bound = schedule.float()
    lower_bound = torch.cat(
        [
            upper_bound.new_zeros(1),
            upper_bound[:-1] + 1.0,
        ]
    )
    distance_to_interval = (
        F.relu(lower_bound[None, :, None, None] - timestep)
        + F.relu(timestep - upper_bound[None, :, None, None])
    )
    if normalize_timestep_distance:
        distance_to_interval = distance_to_interval / (
            NUM_TIMESTEPS / ddim_steps
        )
    return -distance_to_interval / temperature


def project_ddim_class_estimates(
    model_output: dict[str, torch.Tensor],
    noise_spectral_density: torch.Tensor,
    target_frame_energy: torch.Tensor | float,
    ddim_steps: int,
    temperature: float = 0.5,
    normalize_timestep_distance: bool = True,
) -> dict[str, torch.Tensor]:
    """Budget-project class predictions and turn them into DDIM logits."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    raw_class_logits = model_output["raw_class_logits"].float()
    if raw_class_logits.shape[1] != ddim_steps:
        raise ValueError(
            "raw_class_logits channel count must match ddim_steps"
        )

    class_probabilities = raw_class_logits.softmax(dim=1)
    schedule = ddim_schedule_timesteps(
        ddim_steps,
        raw_class_logits.device,
    )
    alpha_schedule = sd15_alpha_bars().to(
        device=raw_class_logits.device,
        dtype=torch.float32,
    )[schedule]
    class_log_noise_variance = torch.log(
        ((1.0 - alpha_schedule) / alpha_schedule).clamp_min(1e-20)
    )
    raw_log_noise_variance = (
        class_probabilities
        * class_log_noise_variance[None, :, None, None]
    ).sum(dim=1, keepdim=True)
    raw_alpha_bar = 1.0 / (1.0 + torch.exp(raw_log_noise_variance))
    projected = project_alpha_bar_estimates(
        {
            "raw_alpha_bar": raw_alpha_bar,
            "predicted_log_patch_energy": model_output[
                "predicted_log_patch_energy"
            ],
        },
        noise_spectral_density,
        target_frame_energy,
    )

    projected_timestep = projected["timestep"]
    projected.update(
        {
            "raw_class_logits": raw_class_logits,
            "raw_class_probabilities": class_probabilities,
            "class_logits": continuous_timestep_to_ddim_logits(
                projected_timestep,
                ddim_steps,
                temperature,
                normalize_timestep_distance,
            ),
        }
    )
    return projected


def project_power_estimates_to_ddim_classes(
    model_output: dict[str, torch.Tensor],
    noise_spectral_density: torch.Tensor,
    target_frame_energy: torch.Tensor | float,
    ddim_steps: int,
    temperature: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Hard-project a relative power map, then form DDIM-class logits."""
    raw_log_power = model_output["raw_log_power"].float()
    predicted_log_energy = model_output[
        "predicted_log_patch_energy"
    ].float()
    if noise_spectral_density.shape != (raw_log_power.shape[0],):
        raise ValueError("noise_spectral_density must have shape [B]")
    if predicted_log_energy.shape != raw_log_power.shape:
        raise ValueError(
            "predicted_log_patch_energy must match raw_log_power shape"
        )

    # A frame-wide log-power offset is unidentifiable after projection. Remove
    # it explicitly so the head only learns the relative patch allocation.
    centered_log_power = raw_log_power - raw_log_power.mean(
        dim=(2, 3),
        keepdim=True,
    )
    relative_power = torch.exp(centered_log_power.clamp(-10.0, 10.0))
    predicted_patch_energy = torch.exp(
        predicted_log_energy.clamp(min=-12.0, max=16.0)
    )
    target_frame_energy = torch.as_tensor(
        target_frame_energy,
        device=raw_log_power.device,
        dtype=torch.float32,
    )
    if target_frame_energy.ndim == 0:
        target_frame_energy = target_frame_energy.expand(
            raw_log_power.shape[0]
        )
    if target_frame_energy.shape != (raw_log_power.shape[0],):
        raise ValueError("target_frame_energy must be scalar or shape [B]")

    raw_estimated_energy = (
        relative_power * predicted_patch_energy
    ).flatten(1).sum(dim=1)
    power_scale = target_frame_energy / raw_estimated_energy.clamp_min(1e-12)
    projected_power = relative_power * power_scale[:, None, None, None]
    noise_spectral_density = noise_spectral_density.float()
    projected_noise_variance = noise_spectral_density[
        :, None, None, None
    ] / (2.0 * projected_power.clamp_min(1e-20))
    projected_timestep = noise_variance_to_continuous_timestep(
        projected_noise_variance
    )
    class_logits = continuous_timestep_to_ddim_logits(
        projected_timestep,
        ddim_steps,
        temperature,
    )
    class_probabilities = class_logits.softmax(dim=1)
    return {
        "raw_log_power": raw_log_power,
        "centered_log_power": centered_log_power,
        "relative_power": relative_power,
        "predicted_log_patch_energy": predicted_log_energy,
        "predicted_patch_energy": predicted_patch_energy,
        "power_scale": power_scale,
        "projected_power": projected_power,
        "noise_variance": projected_noise_variance,
        "alpha_bar": 1.0 / (1.0 + projected_noise_variance),
        "timestep": projected_timestep,
        "normalized_timestep": projected_timestep / (NUM_TIMESTEPS - 1),
        "class_logits": class_logits,
        "class_probabilities": class_probabilities,
        "predicted_class": class_logits.argmax(dim=1, keepdim=True),
        "estimated_frame_energy": (
            projected_power * predicted_patch_energy
        ).flatten(1).sum(dim=1),
    }


def sample_channel_batch(
    z0: torch.Tensor,
    snr_db: torch.Tensor,
    patch_height: int,
    patch_width: int,
    generator: torch.Generator | None = None,
    max_timestep: int = NUM_TIMESTEPS - 1,
    power_log_std_min: float = 0.25,
    power_log_std_max: float = 1.25,
    statistic_mode: str = "basic",
    power_map_source: str = "random",
    allocator: nn.Module | None = None,
    allocator_temperature: float = 1.0,
    allocator_snr_scale_db: float = 1.0,
    input_mode: str = "statistics",
) -> dict[str, torch.Tensor]:
    """Generate allocator-independent, energy-constrained y=a*z+n."""
    if z0.ndim != 4:
        raise ValueError("z0 must have shape [B, C, H, W]")
    batch_size, _, height, width = z0.shape
    if height % patch_height or width % patch_width:
        raise ValueError(
            "Latent spatial dimensions must be divisible by the patch size"
        )
    if snr_db.shape != (batch_size,):
        raise ValueError("snr_db must have shape [B]")
    if not 0 <= max_timestep < NUM_TIMESTEPS:
        raise ValueError("max_timestep must be between 0 and 999")
    if not 0.0 <= power_log_std_min <= power_log_std_max:
        raise ValueError("Invalid power log-standard-deviation range")
    if power_map_source not in POWER_MAP_SOURCES:
        raise ValueError(f"Unknown power_map_source: {power_map_source}")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    if allocator_temperature <= 0.0:
        raise ValueError("allocator_temperature must be positive")
    if allocator_snr_scale_db <= 0.0:
        raise ValueError("allocator_snr_scale_db must be positive")
    if power_map_source == "allocator" and allocator is None:
        raise ValueError(
            "allocator is required when power_map_source=allocator"
        )

    grid_height = height // patch_height
    grid_width = width // patch_width
    clean_patches = F.unfold(
        z0.float(),
        kernel_size=(patch_height, patch_width),
        stride=(patch_height, patch_width),
    )
    clean_patch_energy = clean_patches.square().sum(dim=1)
    target_frame_energy = torch.full(
        (batch_size,),
        z0[0].numel() / 2.0,
        device=z0.device,
        dtype=torch.float32,
    )

    alpha_bars = sd15_alpha_bars().to(
        device=z0.device,
        dtype=torch.float32,
    )
    lambda_schedule = (1.0 - alpha_bars) / alpha_bars
    noise_spectral_density = torch.pow(10.0, -snr_db.float() / 10.0)
    minimum_power = noise_spectral_density[:, None] / (
        2.0 * lambda_schedule[max_timestep]
    )
    minimum_frame_energy = (
        minimum_power * clean_patch_energy
    ).sum(dim=1)
    remaining_frame_energy = target_frame_energy - minimum_frame_energy
    if (remaining_frame_energy <= 0.0).any():
        failed = int((remaining_frame_energy <= 0.0).nonzero()[0, 0].item())
        raise ValueError(
            f"max_timestep={max_timestep} is infeasible for sample {failed}: "
            f"minimum energy {minimum_frame_energy[failed].item():.6f} "
            f">= budget {target_frame_energy[failed].item():.6f}"
        )

    if power_map_source == "random":
        random_field = torch.randn(
            (batch_size, 1, grid_height, grid_width),
            device=z0.device,
            dtype=torch.float32,
            generator=generator,
        )
        random_field = random_field - random_field.mean(
            dim=(2, 3),
            keepdim=True,
        )
        random_field = random_field / random_field.std(
            dim=(2, 3),
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        log_std = power_log_std_min + (
            power_log_std_max - power_log_std_min
        ) * torch.rand(
            (batch_size, 1, 1, 1),
            device=z0.device,
            dtype=torch.float32,
            generator=generator,
        )
        relative_power = torch.exp(
            (log_std * random_field).clamp(-6.0, 6.0)
        ).flatten(1)
    else:
        allocator_module = getattr(allocator, "module", allocator)
        if (
            allocator_module.group_rows != grid_height
            or allocator_module.group_cols != grid_width
        ):
            raise ValueError(
                "Allocator group grid "
                f"{allocator_module.group_rows}x{allocator_module.group_cols} "
                "does not match estimator patch grid "
                f"{grid_height}x{grid_width}"
            )
        with torch.no_grad():
            allocator_logits = allocator(
                z0.float(),
                snr_db.float() / allocator_snr_scale_db,
            ).float()
        expected_shape = (batch_size, grid_height * grid_width)
        if allocator_logits.shape != expected_shape:
            raise ValueError(
                f"Allocator output shape {tuple(allocator_logits.shape)} "
                f"does not match {expected_shape}"
            )
        relative_power = torch.softmax(
            allocator_logits / allocator_temperature,
            dim=1,
        ).clamp_min(1e-12)

    extra_power_scale = remaining_frame_energy / (
        relative_power * clean_patch_energy
    ).sum(dim=1).clamp_min(1e-12)
    group_power = minimum_power + extra_power_scale[:, None] * relative_power
    effective_gain = torch.sqrt(group_power)
    target_timestep, oracle_noise_variance = oracle_gain_to_timestep(
        effective_gain,
        noise_spectral_density,
    )
    if (target_timestep > max_timestep).any():
        raise RuntimeError(
            "Generated power violates the requested maximum timestep"
        )

    gain_map = effective_gain.reshape(
        batch_size,
        1,
        grid_height,
        grid_width,
    )
    gain_map = gain_map.repeat_interleave(patch_height, dim=2)
    gain_map = gain_map.repeat_interleave(patch_width, dim=3)
    noise_standard_deviation = torch.sqrt(
        noise_spectral_density / 2.0
    )
    noise = torch.randn(
        z0.shape,
        device=z0.device,
        dtype=z0.dtype,
        generator=generator,
    ) * noise_standard_deviation[:, None, None, None]
    received_latent = gain_map.to(dtype=z0.dtype) * z0 + noise

    # N0 is known at the receiver. Dividing by sqrt(N0/2) removes an
    # irrelevant frame-wide scale without using a, pilots, or clean z.
    normalized_received_latent = received_latent / noise_standard_deviation[
        :, None, None, None
    ].to(dtype=received_latent.dtype)
    statistic_map = (
        build_patch_statistic_map(
            normalized_received_latent,
            patch_height,
            patch_width,
            statistic_mode,
        )
        if input_mode == "statistics"
        else None
    )
    target_map = target_timestep.reshape(
        batch_size,
        1,
        grid_height,
        grid_width,
    )
    return {
        "statistic_map": statistic_map,
        "normalized_received_latent": normalized_received_latent,
        "received_latent": received_latent,
        "target_timestep": target_map,
        "effective_gain": effective_gain,
        "group_power": group_power.reshape(
            batch_size, 1, grid_height, grid_width
        ),
        "clean_patch_energy": clean_patch_energy.reshape(
            batch_size, 1, grid_height, grid_width
        ),
        "allocation_ratio": (
            group_power * clean_patch_energy / target_frame_energy[:, None]
        ).reshape(batch_size, 1, grid_height, grid_width),
        "target_frame_energy": target_frame_energy,
        "frame_energy": (group_power * clean_patch_energy).sum(dim=1),
        "noise_variance": oracle_noise_variance,
        "noise_spectral_density": noise_spectral_density,
    }


def estimator_step(
    images: torch.Tensor,
    vae: AutoencoderKL,
    estimator: nn.Module,
    snr_db: torch.Tensor,
    patch_height: int,
    patch_width: int,
    max_timestep: int,
    power_log_std_min: float,
    power_log_std_max: float,
    prediction_mode: str,
    ddim_steps: int,
    classification_temperature: float,
    budget_loss_weight: float,
    ordinal_loss_weight: float,
    generator: torch.Generator | None = None,
    statistic_mode: str = "basic",
    power_loss_weight: float = 0.0,
    power_map_source: str = "random",
    allocator: nn.Module | None = None,
    allocator_temperature: float = 1.0,
    allocator_snr_scale_db: float = 1.0,
    input_mode: str = "statistics",
) -> dict[str, torch.Tensor]:
    z0 = encode_latents(images, vae, generator)
    channel_batch = sample_channel_batch(
        z0,
        snr_db,
        patch_height,
        patch_width,
        generator,
        max_timestep,
        power_log_std_min,
        power_log_std_max,
        statistic_mode,
        power_map_source,
        allocator,
        allocator_temperature,
        allocator_snr_scale_db,
        input_mode,
    )
    estimator_input = (
        channel_batch["statistic_map"]
        if input_mode == "statistics"
        else channel_batch["normalized_received_latent"]
    )
    model_output = estimator(estimator_input)
    target_timestep = channel_batch["target_timestep"]
    target_class = timestep_to_ddim_class(target_timestep, ddim_steps)
    if prediction_mode in ("alpha_bar", "log_lambda"):
        if prediction_mode == "alpha_bar":
            projected = project_alpha_bar_estimates(
                model_output,
                channel_batch["noise_spectral_density"],
                channel_batch["target_frame_energy"],
            )
            target_regression = 1.0 / (
                1.0
                + channel_batch["noise_variance"].reshape_as(target_timestep)
            )
            predicted_regression = projected["alpha_bar"]
        else:
            projected = project_log_lambda_estimates(
                model_output,
                channel_batch["noise_spectral_density"],
                channel_batch["target_frame_energy"],
            )
            target_regression = noise_variance_to_normalized_log_lambda(
                channel_batch["noise_variance"].reshape_as(target_timestep)
            )
            predicted_regression = projected["normalized_log_lambda"]
        loss = F.mse_loss(
            predicted_regression,
            target_regression.float(),
        )
        classification_ce_loss = loss.new_zeros(())
        budget_loss = loss.new_zeros(())
        ordinal_loss = loss.new_zeros(())
        power_loss = loss.new_zeros(())
        predicted_timestep = projected["timestep"].detach()
        error_target_timestep = target_timestep.float()
        predicted_class = timestep_to_ddim_class(
            predicted_timestep.round(),
            ddim_steps,
        )
        target_power = channel_batch["group_power"].float()
        target_log_energy = torch.log(
            channel_batch["clean_patch_energy"].float().clamp_min(1e-20)
        )
        true_energy_frame = (
            projected["projected_power"]
            * channel_batch["clean_patch_energy"].float()
        ).flatten(1).sum(dim=1)
        budget_relative_error = (
            true_energy_frame / channel_batch["target_frame_energy"] - 1.0
        )
        power_log_absolute_error_sum = (
            torch.log(
                projected["projected_power"].detach().clamp_min(1e-20)
            )
            - torch.log(target_power.clamp_min(1e-20))
        ).abs().sum()
        energy_log_absolute_error_sum = (
            projected["predicted_log_patch_energy"].detach()
            - target_log_energy
        ).abs().sum()
    elif prediction_mode == "ddim_classification":
        channel_estimates = estimate_channel_from_ddim_logits(
            model_output["raw_class_logits"],
            channel_batch["noise_spectral_density"],
            ddim_steps,
        )
        class_logits = channel_estimates["class_logits"]
        class_target = target_class.squeeze(1)
        classification_ce_loss = F.cross_entropy(
            class_logits,
            class_target,
        )
        class_indices = torch.arange(
            ddim_steps,
            device=images.device,
            dtype=torch.float32,
        )[None, :, None, None]
        ordinal_loss = (
            channel_estimates["class_probabilities"]
            * (class_indices - target_class.float()).abs()
        ).sum(dim=1).mean()
        predicted_class = channel_estimates["predicted_class"].detach()
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            ddim_steps,
        ).float()
        error_target_timestep = ddim_class_to_timestep(
            target_class,
            ddim_steps,
        ).float()
        target_power = channel_batch["group_power"].float()
        estimated_frame_energy = (
            channel_estimates["power"]
            * channel_batch["clean_patch_energy"].float()
        ).flatten(1).sum(dim=1)
        budget_ratio = estimated_frame_energy / channel_batch[
            "target_frame_energy"
        ]
        budget_loss = torch.log(
            budget_ratio.clamp_min(1e-12)
        ).square().mean()
        loss = (
            classification_ce_loss
            + budget_loss_weight * budget_loss
            + ordinal_loss_weight * ordinal_loss
        )
        budget_relative_error = budget_ratio - 1.0
        power_log_absolute_error_sum = (
            torch.log(channel_estimates["power"].detach().clamp_min(1e-20))
            - torch.log(target_power.clamp_min(1e-20))
        ).abs().sum()
        energy_log_absolute_error_sum = loss.new_zeros(())
        power_loss = loss.new_zeros(())
    elif prediction_mode == "projected_power_classification":
        projected = project_power_estimates_to_ddim_classes(
            model_output,
            channel_batch["noise_spectral_density"],
            channel_batch["target_frame_energy"],
            ddim_steps,
            classification_temperature,
        )
        class_logits = projected["class_logits"]
        class_target = target_class.squeeze(1)
        classification_ce_loss = F.cross_entropy(
            class_logits,
            class_target,
        )
        class_indices = torch.arange(
            ddim_steps,
            device=images.device,
            dtype=torch.float32,
        )[None, :, None, None]
        ordinal_loss = (
            projected["class_probabilities"]
            * (class_indices - target_class.float()).square()
        ).sum(dim=1).mean()
        budget_loss = classification_ce_loss.new_zeros(())
        target_power = channel_batch["group_power"].float()
        power_loss = F.smooth_l1_loss(
            torch.log(projected["projected_power"].clamp_min(1e-20)),
            torch.log(target_power.clamp_min(1e-20)),
        )
        loss = (
            classification_ce_loss
            + ordinal_loss_weight * ordinal_loss
            + power_loss_weight * power_loss
        )
        predicted_class = projected["predicted_class"].detach()
        predicted_timestep = ddim_class_to_timestep(
            predicted_class,
            ddim_steps,
        ).float()
        error_target_timestep = ddim_class_to_timestep(
            target_class,
            ddim_steps,
        ).float()
        target_power = channel_batch["group_power"].float()
        target_log_energy = torch.log(
            channel_batch["clean_patch_energy"].float().clamp_min(1e-20)
        )
        true_energy_frame = (
            projected["projected_power"]
            * channel_batch["clean_patch_energy"].float()
        ).flatten(1).sum(dim=1)
        budget_relative_error = (
            true_energy_frame / channel_batch["target_frame_energy"] - 1.0
        )
        power_log_absolute_error_sum = (
            torch.log(
                projected["projected_power"].detach().clamp_min(1e-20)
            )
            - torch.log(target_power.clamp_min(1e-20))
        ).abs().sum()
        energy_log_absolute_error_sum = (
            projected["predicted_log_patch_energy"].detach()
            - target_log_energy
        ).abs().sum()
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    absolute_error = (predicted_timestep - error_target_timestep).abs()
    class_distance = (predicted_class - target_class).abs()
    return {
        "loss": loss,
        "squared_error_sum": absolute_error.square().sum(),
        "absolute_error_sum": absolute_error.sum(),
        "within_5_sum": (absolute_error <= 5.0).sum(),
        "exact_sum": predicted_class.eq(target_class).sum(),
        "same_ddim_class_sum": predicted_class.eq(target_class).sum(),
        "within_one_ddim_class_sum": (class_distance <= 1).sum(),
        "ddim_class_distance_sum": class_distance.sum(),
        "classification_ce_loss": classification_ce_loss,
        "budget_loss": budget_loss,
        "ordinal_loss": ordinal_loss,
        "power_loss": power_loss,
        "patch_count": torch.tensor(
            target_timestep.numel(),
            device=images.device,
            dtype=torch.float64,
        ),
        "gain_min": channel_batch["effective_gain"].min().detach(),
        "gain_max": channel_batch["effective_gain"].max().detach(),
        "target_min": target_timestep.min().detach(),
        "target_max": target_timestep.max().detach(),
        "power_log_absolute_error_sum": power_log_absolute_error_sum,
        "energy_log_absolute_error_sum": energy_log_absolute_error_sum,
        "budget_relative_error_sum": budget_relative_error.detach().abs().sum(),
        "frame_count": torch.tensor(
            images.shape[0],
            device=images.device,
            dtype=torch.float64,
        ),
    }


def build_dataloaders(
    dataset: str,
    data_path: str,
    batch_size: int,
    image_size: int,
    distributed: bool,
    seed: int,
    num_workers: int,
):
    if dataset == "Cityscapes":
        return construct_cityscapes_dataloaders(
            data_path,
            batch_size,
            distributed,
            seed,
            num_workers,
        )
    return construct_dataloaders(
        dataset="ImageNet",
        data_path=data_path,
        batch_size=batch_size,
        normalize=False,
        image_size=image_size,
        distributed=distributed,
        seed=seed,
    )


def reduce_metrics(totals: torch.Tensor, distributed: bool) -> dict[str, float]:
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    patch_count = totals[4].clamp_min(1.0)
    frame_count = totals[9].clamp_min(1.0)
    accuracy = (totals[10] / patch_count).item()
    return {
        "objective_loss": (totals[0] / patch_count).item(),
        "loss": (totals[0] / patch_count).item(),
        "timestep_rmse": math.sqrt((totals[1] / patch_count).item()),
        "timestep_mae": (totals[2] / patch_count).item(),
        "within_5": (totals[3] / patch_count).item(),
        "exact": (totals[5] / patch_count).item(),
        "log_power_mae": (totals[6] / patch_count).item(),
        "log_energy_mae": (totals[7] / patch_count).item(),
        "budget_relative_error": (totals[8] / frame_count).item(),
        "same_ddim_class": accuracy,
        "accuracy": accuracy,
        "within_one_ddim_class": (totals[11] / patch_count).item(),
        "mean_ddim_class_distance": (totals[12] / patch_count).item(),
        "classification_ce_loss": (totals[13] / patch_count).item(),
        "budget_loss": (totals[14] / frame_count).item(),
        "ordinal_loss": (totals[15] / patch_count).item(),
        "power_loss": (totals[16] / patch_count).item(),
    }


def add_batch_metrics(
    totals: torch.Tensor,
    result: dict[str, torch.Tensor],
) -> None:
    patch_count = result["patch_count"]
    totals += torch.stack(
        [
            result["loss"].detach().double() * patch_count,
            result["squared_error_sum"].detach().double(),
            result["absolute_error_sum"].detach().double(),
            result["within_5_sum"].detach().double(),
            patch_count,
            result["exact_sum"].detach().double(),
            result["power_log_absolute_error_sum"].detach().double(),
            result["energy_log_absolute_error_sum"].detach().double(),
            result["budget_relative_error_sum"].detach().double(),
            result["frame_count"],
            result["same_ddim_class_sum"].detach().double(),
            result["within_one_ddim_class_sum"].detach().double(),
            result["ddim_class_distance_sum"].detach().double(),
            result["classification_ce_loss"].detach().double()
            * patch_count,
            result["budget_loss"].detach().double()
            * result["frame_count"],
            result["ordinal_loss"].detach().double() * patch_count,
            result["power_loss"].detach().double() * patch_count,
        ]
    )


def run_training() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a pilot-free alpha-bar or DDIM-class estimator from "
            "received SD1.5 VAE latents"
        )
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument(
        "--dataset",
        choices=["ImageNet", "Cityscapes"],
        default="Cityscapes",
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patch_height", type=int, default=8)
    parser.add_argument("--patch_width", type=int, default=8)
    parser.add_argument("--snr_db_min", type=float, default=-10.0)
    parser.add_argument("--snr_db_max", type=float, default=10.0)
    parser.add_argument("--max_timestep", type=int, default=999)
    parser.add_argument("--power_log_std_min", type=float, default=0.25)
    parser.add_argument("--power_log_std_max", type=float, default=1.25)
    parser.add_argument(
        "--power_map_source",
        choices=POWER_MAP_SOURCES,
        default="random",
        help="Generate GT patch power randomly or with a frozen allocator",
    )
    parser.add_argument("--allocator_checkpoint", default=None)
    parser.add_argument(
        "--allocator_temperature",
        type=float,
        default=None,
        help="Override allocator checkpoint temperature",
    )
    parser.add_argument(
        "--allocator_snr_scale_db",
        type=float,
        default=None,
        help="Override allocator checkpoint SNR input scale",
    )
    parser.add_argument(
        "--input_mode",
        choices=INPUT_MODES,
        default="statistics",
    )
    parser.add_argument(
        "--statistic_mode",
        choices=STATISTIC_MODES,
        default="enhanced",
    )
    parser.add_argument(
        "--prediction_mode",
        choices=PREDICTION_MODES,
        default="alpha_bar",
    )
    parser.add_argument("--ddim_steps", type=int, default=25)
    parser.add_argument(
        "--classification_temperature",
        type=float,
        default=0.5,
        help=(
            "DDIM-class logit temperature used only by "
            "projected_power_classification"
        ),
    )
    parser.add_argument("--budget_loss_weight", type=float, default=1.0)
    parser.add_argument("--ordinal_loss_weight", type=float, default=0.1)
    parser.add_argument("--power_loss_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_path", default="latest.pth")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive")
    if args.patch_height <= 0 or args.patch_width <= 0:
        raise ValueError("patch dimensions must be positive")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("--snr_db_min must not exceed --snr_db_max")
    if not 0 <= args.max_timestep < NUM_TIMESTEPS:
        raise ValueError("--max_timestep must be between 0 and 999")
    if not 0.0 <= args.power_log_std_min <= args.power_log_std_max:
        raise ValueError("Invalid power log-standard-deviation range")
    if (
        args.power_map_source == "allocator"
        and args.allocator_checkpoint is None
    ):
        raise ValueError(
            "--allocator_checkpoint is required when "
            "--power_map_source=allocator"
        )
    if (
        args.allocator_temperature is not None
        and args.allocator_temperature <= 0.0
    ):
        raise ValueError("--allocator_temperature must be positive")
    if (
        args.allocator_snr_scale_db is not None
        and args.allocator_snr_scale_db <= 0.0
    ):
        raise ValueError("--allocator_snr_scale_db must be positive")
    if not 1 <= args.ddim_steps <= NUM_TIMESTEPS:
        raise ValueError("--ddim_steps must be between 1 and 1000")
    if args.classification_temperature <= 0.0:
        raise ValueError("--classification_temperature must be positive")
    if args.budget_loss_weight < 0.0:
        raise ValueError("--budget_loss_weight must be nonnegative")
    if args.ordinal_loss_weight < 0.0:
        raise ValueError("--ordinal_loss_weight must be nonnegative")
    if args.power_loss_weight < 0.0:
        raise ValueError("--power_loss_weight must be nonnegative")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive")

    args.data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    device, distributed, rank, world_size = initialize_distributed(
        args.device,
        args.local_rank,
    )
    is_main_process = rank == 0
    result_dir = (
        Path("checkpoints_timestep_estimator")
        / args.dataset
        / args.exp_name
    )
    if is_main_process:
        result_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = build_logger(result_dir / "train.log", is_main_process)
    checkpoint_path = result_dir / Path(args.save_path).name
    best_checkpoint_path = result_dir / "best.pth"

    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(process_seed)

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        subfolder="vae",
        revision=VAE_MODEL_REVISION,
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    train_loader, val_loader = build_dataloaders(
        args.dataset,
        args.data_path,
        args.batch,
        args.image_size,
        distributed,
        args.seed,
        args.num_workers,
    )
    latent_channels = int(vae.config.latent_channels)
    power_allocator = None
    power_allocator_config = None
    allocator_temperature = 1.0
    allocator_snr_scale_db = 1.0
    if args.power_map_source == "allocator":
        (
            power_allocator,
            power_allocator_config,
            args.allocator_checkpoint,
        ) = load_frozen_power_allocator(
            args.allocator_checkpoint,
            latent_channels,
            device,
        )
        allocator_dataset = power_allocator_config.get("dataset")
        if (
            allocator_dataset is not None
            and allocator_dataset != args.dataset
        ):
            raise ValueError(
                "Power allocator checkpoint uses dataset "
                f"{allocator_dataset}, but estimator uses {args.dataset}"
            )
        allocator_temperature = (
            float(args.allocator_temperature)
            if args.allocator_temperature is not None
            else float(power_allocator_config.get("temperature", 1.0))
        )
        allocator_snr_scale_db = (
            float(args.allocator_snr_scale_db)
            if args.allocator_snr_scale_db is not None
            else float(
                power_allocator_config.get(
                    "allocator_snr_scale_db",
                    1.0,
                )
            )
        )
        if allocator_temperature <= 0.0:
            raise ValueError(
                "Allocator checkpoint temperature must be positive"
            )
        if allocator_snr_scale_db <= 0.0:
            raise ValueError(
                "Allocator checkpoint SNR scale must be positive"
            )

    estimator = PatchTimestepEstimator(
        latent_channels,
        args.patch_height,
        args.patch_width,
        args.hidden_dim,
        args.prediction_mode,
        args.ddim_steps,
        args.statistic_mode,
        args.input_mode,
    ).to(device)

    start_epoch = 0
    global_step = 0
    best_val_objective = float("inf")
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        checkpoint_config = checkpoint["config"]
        if checkpoint_config.get("stage") != "timestep_estimator":
            raise ValueError("Resume checkpoint is not a timestep estimator")
        if checkpoint_config.get("estimator_version") != ESTIMATOR_VERSION:
            raise ValueError(
                "Resume checkpoint uses the old latent-input estimator; "
                "start a new experiment"
            )
        if checkpoint_config.get("dataset") != args.dataset:
            raise ValueError("Resume checkpoint uses a different dataset")
        if checkpoint_config.get("prediction_mode", "alpha_bar") != (
            args.prediction_mode
        ):
            raise ValueError("Resume checkpoint uses a different prediction mode")
        if int(checkpoint_config.get("ddim_steps", 25)) != args.ddim_steps:
            raise ValueError("Resume checkpoint uses a different DDIM grid")
        if checkpoint_config.get("statistic_mode", "basic") != (
            args.statistic_mode
        ):
            raise ValueError("Resume checkpoint uses a different statistic mode")
        if checkpoint_config.get("input_mode", "statistics") != (
            args.input_mode
        ):
            raise ValueError("Resume checkpoint uses a different input mode")
        if checkpoint_config.get("power_map_source", "random") != (
            args.power_map_source
        ):
            raise ValueError(
                "Resume checkpoint uses a different power map source"
            )
        if (
            args.prediction_mode == "ddim_classification"
            and checkpoint_config.get("classification_loss_version")
            != CLASSIFICATION_LOSS_VERSION
        ):
            raise ValueError(
                "Resume checkpoint uses an older classification loss or "
                "projected-logit scale; "
                "start a new classification experiment"
            )
        if (
            args.prediction_mode == "projected_power_classification"
            and checkpoint_config.get(
                "projected_power_classification_version"
            )
            != PROJECTED_POWER_CLASSIFICATION_VERSION
        ):
            raise ValueError(
                "Resume checkpoint uses an older projected-power "
                "classification mode; start a new experiment"
            )
        if (
            args.prediction_mode == "projected_power_classification"
            and not math.isclose(
                float(
                    checkpoint_config.get(
                        "classification_temperature",
                        0.5,
                    )
                ),
                args.classification_temperature,
            )
        ):
            raise ValueError(
                "Resume checkpoint uses a different classification "
                "temperature"
            )
        if checkpoint_config.get("patch_shape") != [
            args.patch_height,
            args.patch_width,
        ]:
            raise ValueError("Resume checkpoint uses a different patch size")
        estimator.load_state_dict(checkpoint["estimator"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val_objective = float(
            checkpoint.get(
                "best_val_objective",
                checkpoint.get("best_val_alpha_bar_mse", float("inf")),
            )
        )

    if distributed:
        ddp_options = {}
        if device.type == "cuda":
            ddp_options = {
                "device_ids": [args.local_rank],
                "output_device": args.local_rank,
            }
        estimator = DistributedDataParallel(estimator, **ddp_options)

    parameters = [
        parameter
        for parameter in estimator.parameters()
        if parameter.requires_grad
    ]
    optimizer = optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if args.prediction_mode == "alpha_bar":
        estimator_name = "global_patch_statistic_alpha_bar_cnn"
        estimator_output = (
            "raw_alpha_bar_and_log_patch_energy_then_budget_projection"
        )
        loss_description = "mse_on_budget_projected_alpha_bar_only"
        classification_inference_mode = None
        classification_budget_constraint = None
    elif args.prediction_mode == "log_lambda":
        estimator_name = "global_patch_statistic_log_lambda_cnn"
        estimator_output = (
            "normalized_log_lambda_and_log_patch_energy_then_budget_"
            "projection"
        )
        loss_description = (
            "mse_on_budget_projected_normalized_log_lambda_only"
        )
        classification_inference_mode = None
        classification_budget_constraint = None
    elif args.prediction_mode == "ddim_classification":
        estimator_name = "global_patch_statistic_ddim_classifier"
        estimator_output = "raw_ddim_class_logits"
        loss_description = (
            "raw_cross_entropy_plus_log_frame_budget_loss_"
            "plus_ordinal_regularization"
        )
        classification_inference_mode = "raw_logits"
        classification_budget_constraint = (
            "soft_log_frame_energy_loss_using_true_training_patch_energy"
        )
    else:
        estimator_name = (
            "global_patch_statistic_projected_power_classifier"
        )
        estimator_output = (
            "raw_log_power_and_log_patch_energy_then_hard_budget_"
            "projection_then_ddim_class_logits"
        )
        loss_description = (
            "cross_entropy_plus_squared_ordinal_plus_smooth_l1_log_power_"
            "on_budget_projected_power_classes"
        )
        classification_inference_mode = "projected_power"
        classification_budget_constraint = (
            "hard_projection_using_predicted_patch_energy"
        )

    selected_statistic_names = (
        statistic_channel_names(args.statistic_mode, latent_channels)
        if args.input_mode == "statistics"
        else ()
    )
    if args.input_mode == "raw":
        estimator_name = estimator_name.replace(
            "patch_statistic",
            "raw_received_latent",
        )
    estimator_input_description = (
        "patch_statistics(y/sqrt(N0/2))"
        if args.input_mode == "statistics"
        else "raw_y/sqrt(N0/2)"
    )
    config = vars(args).copy()
    config.update(
        {
            "stage": "timestep_estimator",
            "estimator_version": ESTIMATOR_VERSION,
            "dataset": args.dataset,
            "estimator": estimator_name,
            "estimator_input": (
                "patch_statistic_map"
                if args.input_mode == "statistics"
                else "raw_normalized_received_latent"
            ),
            "input_mode": args.input_mode,
            "statistic_mode": args.statistic_mode,
            "patch_statistics": list(selected_statistic_names),
            "statistic_transform": (
                (
                    "basic_signed_log1p_and_log1p"
                    if args.statistic_mode == "basic"
                    else (
                        "signed_log1p_moments_and_covariance; bounded_"
                        "correlations; log1p_nonnegative; log_frame_ratios"
                    )
                )
                if args.input_mode == "statistics"
                else None
            ),
            "statistic_channels": len(selected_statistic_names),
            "estimator_output": estimator_output,
            "loss": loss_description,
            "regression_target": (
                "normalized_log_lambda_over_sd15_schedule"
                if args.prediction_mode == "log_lambda"
                else (
                    "alpha_bar"
                    if args.prediction_mode == "alpha_bar"
                    else None
                )
            ),
            "classification_loss_version": (
                CLASSIFICATION_LOSS_VERSION
                if args.prediction_mode == "ddim_classification"
                else None
            ),
            "projected_power_classification_version": (
                PROJECTED_POWER_CLASSIFICATION_VERSION
                if args.prediction_mode
                == "projected_power_classification"
                else None
            ),
            "projected_logit_version": (
                PROJECTED_LOGIT_VERSION
                if args.prediction_mode
                == "projected_power_classification"
                else None
            ),
            "classification_inference_mode": (
                classification_inference_mode
            ),
            "classification_power_representative": (
                "log_noise_variance_midpoint_per_ddim_class"
                if args.prediction_mode == "ddim_classification"
                else None
            ),
            "ddim_timestep_spacing": "trailing",
            "ddim_schedule_timesteps_ascending": (
                ddim_schedule_timesteps(args.ddim_steps).tolist()
            ),
            "classification_budget_constraint": (
                classification_budget_constraint
            ),
            "power_auxiliary_loss": (
                "smooth_l1_on_log_projected_power"
                if args.prediction_mode
                == "projected_power_classification"
                else None
            ),
            "classification_temperature_units": (
                "ddim_class"
                if args.prediction_mode
                == "projected_power_classification"
                else None
            ),
            "ordinal_distance": (
                "squared_class_distance"
                if args.prediction_mode
                == "projected_power_classification"
                else (
                    "absolute_class_distance"
                    if args.prediction_mode == "ddim_classification"
                    else None
                )
            ),
            "num_timesteps": NUM_TIMESTEPS,
            "patch_shape": [args.patch_height, args.patch_width],
            "latent_channels": latent_channels,
            "transmission_model": "y=a*z+n",
            "gain_phase": "positive_real",
            "gain_source": (
                "frozen_power_allocator_with_exact_frame_energy_constraint"
                if args.power_map_source == "allocator"
                else (
                    "allocator_independent_random_power_map_with_exact_"
                    "frame_energy_constraint"
                )
            ),
            "power_budget": "sum_g power_g * latent_energy_g = N_payload",
            "power_map_distribution": (
                "frozen_allocator_softmax_weights"
                if args.power_map_source == "allocator"
                else "iid_log_normal_per_patch"
            ),
            "noise_model": "complex_awgn_packed_as_real_channels",
            "noise_real_component_variance": "N0/2",
            "oracle_noise_variance": "N0/(2*a^2)",
            "oracle_timestep_mapping": (
                "searchsorted([0, (1-alpha_bar)/alpha_bar], variance)-1"
            ),
            "pilots_per_group": 0,
            "uses_power_allocator": (
                args.power_map_source == "allocator"
            ),
            "power_allocator_checkpoint": (
                args.allocator_checkpoint
                if args.power_map_source == "allocator"
                else None
            ),
            "power_allocator_stage": (
                power_allocator_config.get("stage")
                if power_allocator_config is not None
                else None
            ),
            "power_allocator_num_groups": (
                int(power_allocator_config["num_groups"])
                if power_allocator_config is not None
                else None
            ),
            "power_allocator_hidden_dim": (
                int(power_allocator_config.get("hidden_dim", 64))
                if power_allocator_config is not None
                else None
            ),
            "allocator_temperature_effective": (
                allocator_temperature
                if args.power_map_source == "allocator"
                else None
            ),
            "allocator_snr_scale_db_effective": (
                allocator_snr_scale_db
                if args.power_map_source == "allocator"
                else None
            ),
            "uses_task_model": False,
            "uses_reverse_diffusion": False,
            "vae_frozen": True,
            "vae_model_id": VAE_MODEL_ID,
            "vae_model_revision": VAE_MODEL_REVISION,
            "vae_scale": VAE_SCALE,
            "checkpoint_path": str(checkpoint_path),
            "distributed": distributed,
            "world_size": world_size,
            "global_batch_size": args.batch * world_size,
        }
    )
    if is_main_process:
        save_config(result_dir / "config.json", config)
    logger.info(
        "experiment=%s dataset=%s result_dir=%s device=%s world_size=%d",
        args.exp_name,
        args.dataset,
        result_dir,
        device,
        world_size,
    )
    if args.prediction_mode == "projected_power_classification":
        logger.info(
            "task=patch_estimation prediction_mode=%s "
            "channel=y=a*z+n estimator_input=%s "
            "patch=%dx%d max_timestep=%d ddim_steps=%d "
            "snr_db=[%.2f,%.2f] hidden_dim=%d statistic_mode=%s "
            "temperature=%.3f ordinal_loss_weight=%.3f "
            "power_loss_weight=%.3f hard_budget_projection=true "
            "pilots=0 power_map_source=%s allocator_frozen=%s "
            "vae_frozen=true",
            args.prediction_mode,
            estimator_input_description,
            args.patch_height,
            args.patch_width,
            args.max_timestep,
            args.ddim_steps,
            args.snr_db_min,
            args.snr_db_max,
            args.hidden_dim,
            args.statistic_mode,
            args.classification_temperature,
            args.ordinal_loss_weight,
            args.power_loss_weight,
            args.power_map_source,
            args.power_map_source == "allocator",
        )
    else:
        logger.info(
            "task=patch_estimation prediction_mode=%s "
            "channel=y=a*z+n estimator_input=%s "
            "patch=%dx%d max_timestep=%d ddim_steps=%d "
            "snr_db=[%.2f,%.2f] hidden_dim=%d "
            "budget_loss_weight=%.3f ordinal_loss_weight=%.3f "
            "pilots=0 power_map_source=%s allocator_frozen=%s "
            "vae_frozen=true",
            args.prediction_mode,
            estimator_input_description,
            args.patch_height,
            args.patch_width,
            args.max_timestep,
            args.ddim_steps,
            args.snr_db_min,
            args.snr_db_max,
            args.hidden_dim,
            args.budget_loss_weight,
            args.ordinal_loss_weight,
            args.power_map_source,
            args.power_map_source == "allocator",
        )
    if args.power_map_source == "allocator":
        logger.info(
            "allocator_checkpoint=%s allocator_temperature=%.4f "
            "allocator_snr_scale_db=%.4f groups=%d",
            args.allocator_checkpoint,
            allocator_temperature,
            allocator_snr_scale_db,
            int(power_allocator_config["num_groups"]),
        )

    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        estimator.train()
        train_totals = torch.zeros(17, device=device, dtype=torch.float64)
        logger.info(
            "epoch=%d/%d train_start batches=%d",
            epoch + 1,
            args.epochs,
            len(train_loader),
        )
        for batch_index, (images, _labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            snr_db = sample_uniform_snr_db(
                images.shape[0],
                device,
                args.snr_db_min,
                args.snr_db_max,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                result = estimator_step(
                    images,
                    vae,
                    estimator,
                    snr_db,
                    args.patch_height,
                    args.patch_width,
                    args.max_timestep,
                    args.power_log_std_min,
                    args.power_log_std_max,
                    args.prediction_mode,
                    args.ddim_steps,
                    args.classification_temperature,
                    args.budget_loss_weight,
                    args.ordinal_loss_weight,
                    statistic_mode=args.statistic_mode,
                    power_loss_weight=args.power_loss_weight,
                    power_map_source=args.power_map_source,
                    allocator=power_allocator,
                    allocator_temperature=allocator_temperature,
                    allocator_snr_scale_db=allocator_snr_scale_db,
                    input_mode=args.input_mode,
                )

            finite_loss = torch.isfinite(result["loss"]).to(torch.int32)
            if distributed:
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if not finite_loss.item():
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            add_batch_metrics(train_totals, result)

            if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                batch_patches = result["patch_count"].item()
                batch_mae = result["absolute_error_sum"].item() / batch_patches
                if args.prediction_mode == "ddim_classification":
                    batch_accuracy = (
                        result["same_ddim_class_sum"].item()
                        / batch_patches
                    )
                    logger.info(
                        "epoch=%d/%d batch=%d/%d loss=%.6f "
                        "ce=%.6f budget_loss=%.6f ordinal_loss=%.4f "
                        "acc=%.4f "
                        "timestep_mae=%.2f budget_error=%.4f "
                        "frame_snr_db=%.2f",
                        epoch + 1,
                        args.epochs,
                        batch_index + 1,
                        len(train_loader),
                        result["loss"].item(),
                        result["classification_ce_loss"].item(),
                        result["budget_loss"].item(),
                        result["ordinal_loss"].item(),
                        batch_accuracy,
                        batch_mae,
                        (
                            result["budget_relative_error_sum"].item()
                            / result["frame_count"].item()
                        ),
                        snr_db.mean().item(),
                    )
                elif (
                    args.prediction_mode
                    == "projected_power_classification"
                ):
                    batch_accuracy = (
                        result["same_ddim_class_sum"].item()
                        / batch_patches
                    )
                    logger.info(
                        "epoch=%d/%d batch=%d/%d loss=%.6f "
                        "ce=%.6f ordinal_loss=%.4f power_loss=%.4f "
                        "acc=%.4f timestep_mae=%.2f budget_error=%.4f "
                        "frame_snr_db=%.2f",
                        epoch + 1,
                        args.epochs,
                        batch_index + 1,
                        len(train_loader),
                        result["loss"].item(),
                        result["classification_ce_loss"].item(),
                        result["ordinal_loss"].item(),
                        result["power_loss"].item(),
                        batch_accuracy,
                        batch_mae,
                        (
                            result["budget_relative_error_sum"].item()
                            / result["frame_count"].item()
                        ),
                        snr_db.mean().item(),
                    )
                else:
                    logger.info(
                        "epoch=%d/%d batch=%d/%d objective=%.6f "
                        "timestep_mae=%.2f target=[%d,%d] "
                        "gain=[%.6f,%.6f] budget_error=%.4f "
                        "frame_snr_db=%.2f",
                        epoch + 1,
                        args.epochs,
                        batch_index + 1,
                        len(train_loader),
                        result["loss"].item(),
                        batch_mae,
                        result["target_min"].item(),
                        result["target_max"].item(),
                        result["gain_min"].item(),
                        result["gain_max"].item(),
                        (
                            result["budget_relative_error_sum"].item()
                            / result["frame_count"].item()
                        ),
                        snr_db.mean().item(),
                    )

        train_metrics = reduce_metrics(train_totals, distributed)

        estimator.eval()
        validation_totals = torch.zeros(
            17,
            device=device,
            dtype=torch.float64,
        )
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(args.seed + rank + 1_000_000)
        with torch.no_grad():
            for images, _labels in val_loader:
                images = images.to(device, non_blocking=True)
                snr_db = sample_uniform_snr_db(
                    images.shape[0],
                    device,
                    args.snr_db_min,
                    args.snr_db_max,
                    validation_generator,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    result = estimator_step(
                        images,
                        vae,
                        estimator,
                        snr_db,
                        args.patch_height,
                        args.patch_width,
                        args.max_timestep,
                        args.power_log_std_min,
                        args.power_log_std_max,
                        args.prediction_mode,
                        args.ddim_steps,
                        args.classification_temperature,
                        args.budget_loss_weight,
                        args.ordinal_loss_weight,
                        validation_generator,
                        statistic_mode=args.statistic_mode,
                        power_loss_weight=args.power_loss_weight,
                        power_map_source=args.power_map_source,
                        allocator=power_allocator,
                        allocator_temperature=allocator_temperature,
                        allocator_snr_scale_db=allocator_snr_scale_db,
                        input_mode=args.input_mode,
                    )
                add_batch_metrics(validation_totals, result)

        validation_metrics = reduce_metrics(
            validation_totals,
            distributed,
        )
        scheduler.step()
        if args.prediction_mode == "ddim_classification":
            logger.info(
                "epoch=%d/%d train_loss=%.6f train_ce=%.6f "
                "train_budget_loss=%.6f train_ordinal_loss=%.4f "
                "train_acc=%.4f "
                "train_timestep_mae=%.2f train_budget_error=%.4f "
                "val_loss=%.6f val_ce=%.6f val_budget_loss=%.6f "
                "val_ordinal_loss=%.4f "
                "val_acc=%.4f val_timestep_mae=%.2f "
                "val_budget_error=%.4f",
                epoch + 1,
                args.epochs,
                train_metrics["loss"],
                train_metrics["classification_ce_loss"],
                train_metrics["budget_loss"],
                train_metrics["ordinal_loss"],
                train_metrics["accuracy"],
                train_metrics["timestep_mae"],
                train_metrics["budget_relative_error"],
                validation_metrics["loss"],
                validation_metrics["classification_ce_loss"],
                validation_metrics["budget_loss"],
                validation_metrics["ordinal_loss"],
                validation_metrics["accuracy"],
                validation_metrics["timestep_mae"],
                validation_metrics["budget_relative_error"],
            )
        elif args.prediction_mode == "projected_power_classification":
            logger.info(
                "epoch=%d/%d train_loss=%.6f train_ce=%.6f "
                "train_ordinal_loss=%.4f train_power_loss=%.4f "
                "train_acc=%.4f train_timestep_mae=%.2f "
                "train_budget_error=%.4f val_loss=%.6f val_ce=%.6f "
                "val_ordinal_loss=%.4f val_power_loss=%.4f val_acc=%.4f "
                "val_timestep_mae=%.2f val_budget_error=%.4f",
                epoch + 1,
                args.epochs,
                train_metrics["loss"],
                train_metrics["classification_ce_loss"],
                train_metrics["ordinal_loss"],
                train_metrics["power_loss"],
                train_metrics["accuracy"],
                train_metrics["timestep_mae"],
                train_metrics["budget_relative_error"],
                validation_metrics["loss"],
                validation_metrics["classification_ce_loss"],
                validation_metrics["ordinal_loss"],
                validation_metrics["power_loss"],
                validation_metrics["accuracy"],
                validation_metrics["timestep_mae"],
                validation_metrics["budget_relative_error"],
            )
        else:
            logger.info(
                "epoch=%d/%d train_objective=%.6f "
                "train_timestep_rmse=%.2f train_timestep_mae=%.2f "
                "train_same_ddim=%.4f train_within_one_ddim=%.4f "
                "val_objective=%.6f val_timestep_rmse=%.2f "
                "val_timestep_mae=%.2f val_same_ddim=%.4f "
                "val_within_one_ddim=%.4f",
                epoch + 1,
                args.epochs,
                train_metrics["objective_loss"],
                train_metrics["timestep_rmse"],
                train_metrics["timestep_mae"],
                train_metrics["same_ddim_class"],
                train_metrics["within_one_ddim_class"],
                validation_metrics["objective_loss"],
                validation_metrics["timestep_rmse"],
                validation_metrics["timestep_mae"],
                validation_metrics["same_ddim_class"],
                validation_metrics["within_one_ddim_class"],
            )

        is_best = (
            validation_metrics["objective_loss"]
            < best_val_objective
        )
        if is_best:
            best_val_objective = validation_metrics["objective_loss"]
        if is_main_process:
            checkpoint_state = {
                "estimator": unwrap_model(estimator).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_objective": best_val_objective,
                "config": config,
                "metrics": {
                    "train": train_metrics,
                    "validation": validation_metrics,
                },
            }
            torch.save(checkpoint_state, checkpoint_path)
            if is_best:
                torch.save(checkpoint_state, best_checkpoint_path)
            logger.info(
                "checkpoint_saved path=%s best=%s",
                checkpoint_path,
                is_best,
            )
        if distributed:
            dist.barrier()


def main() -> None:
    install_termination_handlers()
    try:
        run_training()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
