import argparse
import json
import logging
import math
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.data_processing import construct_dataloaders
from power_utils import (
    LegacyPowerAllocator,
    build_latent_classifier,
    classifier_forward,
)
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, MEAN_REAL_COMPONENT_POWER,
    NOISE_REAL_COMPONENT_VARIANCE, RAYLEIGH_CHANNEL_MODEL,
    BlockPowerAllocator, OrthogonalPatchTransform, block_map, patchify,
    common_channel_magnitude, gain_estimator_receive, sample_raw_channel,
    sample_rayleigh_fading,
)


VAE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
VAE_SCALE = 0.18215


class WarmupCosineLR(optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by cosine decay, counted in optimizer updates.

    Each parameter group's initial LR is its peak; the warmup and minimum
    factors are relative to that group's peak. Step zero is the first update's
    LR. Advancing after the last planned update leaves the LR at min_lr_ratio.
    """

    def __init__(self, optimizer, total_steps, warmup_steps=0,
                 warmup_start_factor=0.1, min_lr_ratio=0.5):
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("Require total_steps > 0 and 0 <= warmup_steps < total_steps")
        if not math.isfinite(warmup_start_factor) or not 0 <= warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in [0,1]")
        if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        step = min(max(self.last_epoch, 0), self.total_steps)
        if step < self.warmup_steps:
            factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        return [base_lr * factor for base_lr in self.base_lrs]


def build_training_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0.5,
                             warmup_start_factor=0.1, min_lr_ratio=0.5, checkpoint=None):
    """Create a step schedule, or restore the checkpoint's original schedule.

    Old epoch-cosine checkpoints retain their update interval and T_max.
    Optimizer state is loaded after scheduler construction so warmup does not
    overwrite the checkpoint's next-update LR.
    """
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs and training batches per epoch must be positive")
    state = checkpoint["scheduler"] if checkpoint is not None else None
    if state is not None and "T_max" in state:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif state is not None:
        scheduler = WarmupCosineLR(
            optimizer, state["total_steps"], state["warmup_steps"],
            state["warmup_start_factor"], state["min_lr_ratio"],
        )
    else:
        if not math.isfinite(warmup_epochs) or not 0 <= warmup_epochs < epochs:
            raise ValueError("warmup_epochs must be in [0,epochs)")
        scheduler = WarmupCosineLR(
            optimizer, epochs * steps_per_epoch, round(warmup_epochs * steps_per_epoch),
            warmup_start_factor, min_lr_ratio,
        )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(state)
    return scheduler


def training_schedule_config(scheduler):
    if isinstance(scheduler, WarmupCosineLR):
        return {
            "type": "warmup_cosine", "interval": "optimizer_step",
            "total_steps": scheduler.total_steps, "warmup_steps": scheduler.warmup_steps,
            "warmup_start_factor": scheduler.warmup_start_factor, "min_lr_ratio": scheduler.min_lr_ratio,
            "peak_lrs": scheduler.base_lrs,
        }
    return {"type": "cosine", "interval": "epoch", "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min, "peak_lrs": scheduler.base_lrs}


def optimizer_step_with_schedule(optimizer, scheduler, scaler, parameters, grad_clip):
    """Advance the step schedule only when AMP actually applies the update."""
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(parameters, grad_clip)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= scale_before
    if updated and isinstance(scheduler, WarmupCosineLR):
        scheduler.step()
    return updated


class PowerAllocator(nn.Module):
    """Block-MLP allocator with jointly learned Q; legacy CNN without Q."""

    def __init__(self, latent_channels, num_groups, hidden_dim, sampling=False,
                 orthogonal_transform=False, block_dim=256, group_grid=None):
        super().__init__()
        if orthogonal_transform and group_grid is None and num_groups != 64:
            raise ValueError("The orthogonal ImageNet transmitter requires 64 blocks")
        if group_grid is not None and (
            not orthogonal_transform or len(group_grid) != 2
            or any(not isinstance(value, int) or value <= 0 for value in group_grid)
            or group_grid[0] * group_grid[1] != num_groups
        ):
            raise ValueError("group_grid must be a positive (rows,cols) grid matching orthogonal num_groups")
        self.sampling = sampling
        self.block_dim = block_dim
        if orthogonal_transform:
            if sampling:
                raise ValueError("The block power allocator emits deterministic logits")
            self.group_rows, self.group_cols = (8, 8) if group_grid is None else group_grid
            self.allocator_architecture = ALLOCATOR_ARCHITECTURE
            self.orthogonal = OrthogonalPatchTransform(self.group_rows, self.group_cols)
            self.block_allocator = BlockPowerAllocator(block_dim, num_groups, hidden_dim)
        else:
            # Preserve existing state-dict names for the legacy path.
            legacy = LegacyPowerAllocator(latent_channels, num_groups, hidden_dim, sampling)
            self.group_rows, self.group_cols = legacy.group_rows, legacy.group_cols
            self.allocator_architecture = "legacy_cnn"
            self.orthogonal = None
            self.group_pool, self.backbone = legacy.group_pool, legacy.backbone
            if sampling:
                self.log_sigma_head = legacy.log_sigma_head
            else:
                self.mu_head = legacy.mu_head

    def forward(self, z0, conditioning, return_transformed=False):
        if self.orthogonal is not None:
            transformed, q = self.orthogonal(z0)
            logits = self.block_allocator(patchify(transformed, self.group_rows, self.group_cols), conditioning)
        else:
            transformed, q = z0, None
            logits = LegacyPowerAllocator.forward(self, z0, conditioning)
        if return_transformed:
            return logits, transformed, q
        return logits


def restore_spatial_latent(received, allocation_info, channel_info=None):
    """Apply Q after gain correction; propagate noise variance into space."""
    q = allocation_info.get("orthogonal_matrix")
    if q is None:
        return received, channel_info
    rows, cols = allocation_info["group_grid"]
    restored = OrthogonalPatchTransform.apply_matrix(received, q, rows, cols)
    if channel_info is None:
        return restored, None
    info = dict(channel_info)
    with torch.autocast(device_type=received.device.type, enabled=False):
        variance = torch.matmul(q.float().square(), channel_info["noise_variance"].float().T).T
        clean = allocation_info.get("source_latent")
        signal_power = patchify(clean.float() if clean is not None else restored, rows, cols).square().mean(-1)
        if clean is None:
            signal_power = (signal_power - variance).clamp_min(1e-8)
        snr = (signal_power / variance.clamp_min(1e-12)).clamp_min(1e-8)
    info.update(noise_variance=variance, signal_power=signal_power, snr=snr,
                log_snr=snr.log(), snr_map=block_map(snr.log(), restored, rows, cols))
    return restored, info


def build_logger(log_path: Path, enabled: bool = True) -> logging.Logger:
    logger = logging.getLogger("train_power_allocator")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def initialize_distributed(
    device_name: str,
    local_rank: int,
) -> tuple[torch.device, bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    requested_device = torch.device(device_name)

    if distributed:
        backend = "nccl" if requested_device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        if requested_device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = requested_device
    else:
        rank = 0
        device = requested_device

    return device, distributed, rank, world_size


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def install_termination_handlers() -> None:
    def handle_termination(signum, _frame) -> None:
        signal.signal(signum, signal.SIG_DFL)
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), handle_termination)


def save_config(config_path: Path, config: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def save_epoch_checkpoints(state, checkpoint_path, save_every=5, total_epochs=None):
    """Update latest and retain periodic/final epochs as independent files."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed_epoch = int(state["epoch"]) + 1
    paths = [checkpoint_path]
    if save_every > 0 and (completed_epoch % save_every == 0 or completed_epoch == total_epochs):
        paths.append(checkpoint_path.parent / f"epoch_{completed_epoch:03d}.pth")
    for path in paths:
        temporary_path = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return paths


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    predictions = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return predictions.eq(labels[:, None]).any(dim=1).float().mean()


def encode_latents(
    images: torch.Tensor,
    vae: AutoencoderKL,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    vae_images = (images * 2.0 - 1.0).float()
    with torch.no_grad():
        posterior = vae.encode(vae_images).latent_dist
        z0 = posterior.sample(generator=generator) * VAE_SCALE
    return z0.float()


def _split_spatial_groups(
    latent: torch.Tensor,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, ...]:
    row_groups = torch.tensor_split(latent, rows, dim=2)
    return tuple(
        group
        for row_group in row_groups
        for group in torch.tensor_split(row_group, cols, dim=3)
    )


def _merge_spatial_groups(
    groups: tuple[torch.Tensor, ...] | list[torch.Tensor],
    rows: int,
    cols: int,
) -> torch.Tensor:
    row_tensors = [
        torch.cat(groups[row * cols:(row + 1) * cols], dim=3)
        for row in range(rows)
    ]
    return torch.cat(row_tensors, dim=2)


def sample_snr_db(
    batch_size: int,
    device: torch.device,
    snr_db_values: list[float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    values = torch.tensor(snr_db_values, device=device, dtype=torch.float32)
    if values.numel() == 1:
        return torch.full(
            (batch_size,),
            values.item(),
            device=device,
        )
    indices = torch.randint(
        values.numel(),
        (batch_size,),
        device=device,
        generator=generator,
    )
    return values[indices]


def allocate_power(
    z0: torch.Tensor,
    snr_db: torch.Tensor,
    allocator: nn.Module,
    channel: "QuantizedAWGNChannel",
    temperature: float = 1.0,
    channel_coefficient: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Allocate gains under the exact frame-energy constraint."""
    if isinstance(channel, RayleighFadingChannel) and channel_coefficient is None:
        raise ValueError("Sample the common Rayleigh h before allocation to provide transmitter CSI")
    allocator_snr_db = snr_db
    if channel_coefficient is not None:
        magnitude = common_channel_magnitude(channel_coefficient, z0)
        allocator_snr_db = snr_db + 20.0 * magnitude.log10()
    allocator_module = getattr(allocator, "module", allocator)
    log_gamma = allocator_snr_db * (math.log(10.0) / 10.0)
    conditioning = (log_gamma if getattr(allocator_module, "allocator_architecture", None)
                    == ALLOCATOR_ARCHITECTURE else allocator_snr_db)
    if isinstance(allocator_module, PowerAllocator):
        allocator_output, transformed_latent, q = allocator(z0, conditioning, return_transformed=True)
    else:
        allocator_output, transformed_latent, q = allocator(z0, conditioning), z0, None
    transmitted_latent = channel.prepare_latent(transformed_latent)
    transmitted_groups = _split_spatial_groups(
        transmitted_latent,
        allocator_module.group_rows,
        allocator_module.group_cols,
    )
    payload_energy = torch.stack(
        [
            group.square().sum(dim=(1, 2, 3))
            / channel.kappa_m_squared
            for group in transmitted_groups
        ],
        dim=1,
    )
    unscaled_block_energy = payload_energy + channel.pilots_per_group
    payload_symbols = z0[0].numel() // 2
    pilot_symbols = len(transmitted_groups) * channel.pilots_per_group
    frame_channel_uses = payload_symbols + pilot_symbols
    # Rayleigh P=1 is defined per real component. Complex channel-use
    # counts stay separate: 16,384 real values occupy 8,192 I/Q symbols.
    target_frame_energy = float(
        z0[0].numel() * MEAN_REAL_COMPONENT_POWER
        if isinstance(channel, RayleighFadingChannel) else frame_channel_uses
    )

    weights = torch.softmax(allocator_output.float() / temperature, dim=1)
    weights = weights.clamp_min(1e-12)
    normalizer = (
        weights * unscaled_block_energy
    ).sum(dim=1, keepdim=True)
    group_power = target_frame_energy * weights / normalizer
    block_transmit_energy = group_power * unscaled_block_energy
    frame_energy = block_transmit_energy.sum(dim=1)
    allocation_ratio = block_transmit_energy / target_frame_energy
    return allocation_ratio, group_power, {
        "channel_coefficient": channel_coefficient,
        "allocator_snr_db": allocator_snr_db,
        "log_gamma": log_gamma,
        "source_latent": z0,
        "channel_input": transformed_latent,
        "orthogonal_matrix": q,
        "group_grid": (allocator_module.group_rows, allocator_module.group_cols),
        "transmitted_latent": transmitted_latent,
        "payload_energy": payload_energy,
        "unscaled_block_energy": unscaled_block_energy,
        "frame_energy": frame_energy,
        "target_frame_energy": target_frame_energy,
        "frame_channel_uses": frame_channel_uses,
        "mean_symbol_power": frame_energy / frame_channel_uses,
        "mean_real_component_power": frame_energy / (2 * frame_channel_uses),
    }


class QuantizedAWGNChannel(nn.Module):
    """Patch-wise power-scaled AWGN with optional fixed quantizer."""

    def __init__(
        self,
        group_rows: int,
        group_cols: int,
        qam_order: int = 256,
        quantization_amplitude: float = 3.0,
        default_snr_db: float = 10.0,
        channel_gain_abs: torch.Tensor | list[float] | float = 1.0,
        pilots_per_group: int = 4,
        transmission_mode: str = "qam",
    ):
        super().__init__()
        real_levels = math.isqrt(qam_order)
        if real_levels ** 2 != qam_order:
            raise ValueError("qam_order must be a square QAM order")
        if quantization_amplitude <= 0.0:
            raise ValueError("quantization_amplitude must be positive")
        if pilots_per_group < 0:
            raise ValueError("pilots_per_group must be non-negative")
        default_channel_gains = torch.as_tensor(
            channel_gain_abs,
            dtype=torch.float32,
        )
        if (default_channel_gains <= 0.0).any():
            raise ValueError("channel_gain_abs must be positive")

        self.group_rows = group_rows
        self.group_cols = group_cols
        self.transmission_mode = transmission_mode
        self.qam_order = qam_order
        self.real_levels = real_levels
        self.quantization_amplitude = quantization_amplitude
        self.default_snr_db = default_snr_db
        self.pilots_per_group = pilots_per_group
        self.gain_power_threshold = 1e-12
        self.delta = (
            2.0 * quantization_amplitude / (real_levels - 1)
        )
        self.qam_normalizer = math.sqrt(2.0 * (qam_order - 1) / 3.0)
        self.kappa_m = (
            quantization_amplitude / (real_levels - 1) * self.qam_normalizer
            if transmission_mode == "qam"
            else 1.0
        )
        self.kappa_m_squared = self.kappa_m ** 2
        self.register_buffer(
            "default_channel_gain_abs",
            default_channel_gains,
        )

    def prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.transmission_mode == "analog":
            return latent
        return self.quantize(latent)

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        amplitude = self.quantization_amplitude
        clipped = latent.clamp(min=-amplitude, max=amplitude)
        symbol_indices = torch.floor(
            (clipped + amplitude) / self.delta + 0.5
        )
        symbol_indices = symbol_indices.clamp(
            min=0,
            max=self.real_levels - 1,
        )
        return -amplitude + symbol_indices * self.delta

    def _noise_spectral_density(
        self,
        reference: torch.Tensor,
        snr_db: torch.Tensor | float | None,
    ) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.default_snr_db
        snr_db = torch.as_tensor(
            snr_db,
            device=reference.device,
            dtype=torch.float32,
        )
        if snr_db.ndim == 0:
            snr_db = snr_db.expand(reference.shape[0])
        if snr_db.shape != (reference.shape[0],):
            raise ValueError("snr_db must be scalar or have shape [batch_size]")
        return torch.pow(10.0, -snr_db / 10.0)

    def sample_epsilon(
        self,
        reference: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def _channel_gains(
        self,
        group_power: torch.Tensor,
        channel_gain_abs: torch.Tensor | float | None,
    ) -> torch.Tensor:
        if channel_gain_abs is None:
            gains = self.default_channel_gain_abs.to(
                device=group_power.device,
                dtype=group_power.dtype,
            )
        else:
            gains = torch.as_tensor(
                channel_gain_abs,
                device=group_power.device,
                dtype=group_power.dtype,
            )
        if gains.ndim == 0 or gains.numel() == 1:
            return gains.expand_as(group_power)
        if gains.ndim == 1 and gains.shape[0] == group_power.shape[1]:
            return gains.unsqueeze(0).expand_as(group_power)
        if gains.shape != group_power.shape:
            raise ValueError(
                "channel_gain_abs must be scalar, [num_groups], or "
                "[batch_size, num_groups]"
            )
        return gains

    def forward(
        self,
        z0: torch.Tensor,
        group_power: torch.Tensor,
        generator: torch.Generator | None = None,
        epsilon: torch.Tensor | None = None,
        channel_gain_abs: torch.Tensor | float | None = None,
        snr_db: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        groups = _split_spatial_groups(
            z0,
            self.group_rows,
            self.group_cols,
        )
        if group_power.shape != (z0.shape[0], len(groups)):
            raise ValueError(
                "group_power must have shape "
                f"({z0.shape[0]}, {len(groups)})"
            )
        if epsilon is None:
            epsilon = self.sample_epsilon(z0, generator)
        epsilon_groups = _split_spatial_groups(
            epsilon,
            self.group_rows,
            self.group_cols,
        )
        channel_gains = self._channel_gains(
            group_power,
            channel_gain_abs,
        ).abs().clamp_min(1e-12)
        noise_spectral_density = self._noise_spectral_density(
            z0,
            snr_db,
        )

        received_groups = []
        signal_powers = []
        noise_variances = []
        snrs = []
        for group_index, (group, group_epsilon) in enumerate(
            zip(groups, epsilon_groups)
        ):
            transmitted_group = self.prepare_latent(group)
            power = group_power[:, group_index].clamp_min(
                self.gain_power_threshold
            )
            amplitude_gain = torch.sqrt(power)
            fading_gain = channel_gains[:, group_index]
            noise_variance = (
                self.kappa_m_squared
                * noise_spectral_density
                / (
                    2.0
                    * fading_gain.square()
                    * amplitude_gain.square()
                )
            )
            received_group = transmitted_group + torch.sqrt(
                noise_variance
            )[:, None, None, None] * group_epsilon
            received_groups.append(received_group)

            signal_power = transmitted_group.square().mean(dim=(1, 2, 3))
            snr = (signal_power / noise_variance).clamp_min(1e-8)
            signal_powers.append(signal_power)
            noise_variances.append(noise_variance)
            snrs.append(snr)

        received_latent = _merge_spatial_groups(
            received_groups,
            self.group_rows,
            self.group_cols,
        )
        signal_power = torch.stack(signal_powers, dim=1)
        noise_variance = torch.stack(noise_variances, dim=1)
        snr = torch.stack(snrs, dim=1)
        log_snr = torch.log(snr)
        snr_groups = [
            log_snr[:, group_index, None, None, None].expand_as(group)
            for group_index, group in enumerate(groups)
        ]
        snr_map = _merge_spatial_groups(
            snr_groups,
            self.group_rows,
            self.group_cols,
        )
        return received_latent, {
            "group_power": group_power,
            "amplitude_gain": torch.sqrt(
                group_power.clamp_min(self.gain_power_threshold)
            ),
            "channel_gain_abs": channel_gains,
            "signal_power": signal_power,
            "noise_variance": noise_variance,
            "noise_spectral_density": noise_spectral_density,
            "snr": snr,
            "log_snr": log_snr,
            "snr_map": snr_map,
            "quantization_delta": (
                self.delta if self.transmission_mode == "qam" else None
            ),
            "kappa_m": self.kappa_m,
        }


class RayleighFadingChannel(QuantizedAWGNChannel):
    """Pilot-free block fading with one complex h per image and perfect CSI."""

    def __init__(self, group_rows=8, group_cols=8, default_snr_db=10.0):
        super().__init__(group_rows, group_cols, default_snr_db=default_snr_db,
                         pilots_per_group=0, transmission_mode="analog")

    @staticmethod
    def sample_channel_coefficient(reference, generator=None):
        return sample_rayleigh_fading(reference.shape[0], reference.device, generator)

    def forward(self, z0, group_power, generator=None, epsilon=None,
                channel_coefficient=None, snr_db=None):
        h = (self.sample_channel_coefficient(z0, generator)
             if channel_coefficient is None else channel_coefficient)
        n0 = self._noise_spectral_density(z0, snr_db)
        raw = sample_raw_channel(z0, group_power, n0, h, generator,
                                 self.group_rows, self.group_cols, epsilon)
        received = gain_estimator_receive(raw, 0.5 * group_power.log(), n0, h,
                                          self.group_rows, self.group_cols)
        signal_power = patchify(z0, self.group_rows, self.group_cols).square().mean(-1)
        snr = (signal_power / received["noise_variance"]).clamp_min(1e-8)
        return received["equalized_latent"], {
            "group_power": group_power, "amplitude_gain": group_power.sqrt(),
            "channel_coefficient": h, "channel_gain_abs": h.abs(),
            "raw_received_latent": raw, "signal_power": signal_power,
            "noise_variance": received["noise_variance"], "noise_spectral_density": n0,
            "snr": snr, "log_snr": snr.log(),
            "snr_map": block_map(snr.log(), z0, self.group_rows, self.group_cols),
            "quantization_delta": None, "kappa_m": 1.0,
        }


def training_step(
    images: torch.Tensor,
    labels: torch.Tensor,
    vae: AutoencoderKL,
    allocator: nn.Module,
    classifier: nn.Module,
    channel: QuantizedAWGNChannel,
    snr_db: torch.Tensor,
    temperature: float = 1.0,
    condition_classifier_on_snr: bool = True,
) -> dict:
    z0 = encode_latents(images, vae)
    h = channel.sample_channel_coefficient(z0) if isinstance(channel, RayleighFadingChannel) else None
    allocation_ratio, group_power, allocation_info = allocate_power(
        z0,
        snr_db,
        allocator,
        channel,
        temperature,
        channel_coefficient=h,
    )
    received_latent, channel_info = channel(
        allocation_info["channel_input"],
        group_power,
        snr_db=snr_db,
        **({"channel_coefficient": h} if h is not None else {}),
    )
    received_latent, channel_info = restore_spatial_latent(
        received_latent, allocation_info, channel_info
    )
    logits = classifier_forward(
        classifier,
        received_latent,
        channel_info,
        condition_classifier_on_snr,
    )

    task_loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return {
        "loss": task_loss,
        "task_loss": task_loss.detach(),
        "mean_symbol_power": allocation_info[
            "mean_symbol_power"
        ].detach(),
        "frame_energy": allocation_info["frame_energy"].detach(),
        "mean_real_component_power": allocation_info["mean_real_component_power"].detach(),
        "top1_accuracy": accuracy.detach(),
        "allocation_ratio": allocation_ratio.detach(),
        "group_power": group_power.detach(),
        "snr": channel_info["snr"].detach(),
    }


def validation_step(
    images: torch.Tensor,
    labels: torch.Tensor,
    vae: AutoencoderKL,
    allocator: nn.Module,
    classifier: nn.Module,
    channel: QuantizedAWGNChannel,
    snr_db: torch.Tensor,
    generator: torch.Generator,
    temperature: float,
    condition_classifier_on_snr: bool,
) -> dict:
    z0 = encode_latents(images, vae, generator)
    h = channel.sample_channel_coefficient(z0, generator) if isinstance(channel, RayleighFadingChannel) else None
    _, group_power, allocation_info = allocate_power(
        z0,
        snr_db,
        allocator,
        channel,
        temperature,
        channel_coefficient=h,
    )
    received_latent, channel_info = channel(
        allocation_info["channel_input"],
        group_power,
        generator=generator,
        snr_db=snr_db,
        **({"channel_coefficient": h} if h is not None else {}),
    )
    received_latent, channel_info = restore_spatial_latent(
        received_latent, allocation_info, channel_info
    )
    logits = classifier_forward(
        classifier,
        received_latent,
        channel_info,
        condition_classifier_on_snr,
    )
    task_loss = F.cross_entropy(logits, labels)
    return {
        "loss": task_loss,
        "task_loss": task_loss,
        "top1_accuracy": topk_accuracy(logits, labels, 1),
        "top5_accuracy": topk_accuracy(logits, labels, 5),
        "mean_symbol_power": allocation_info[
            "mean_symbol_power"
        ].mean(),
        "mean_real_component_power": allocation_info["mean_real_component_power"].mean(),
        "mean_latent_snr_db": (
            10.0 * torch.log10(channel_info["snr"])
        ).mean(),
    }


def run_training() -> None:
    parser = argparse.ArgumentParser(
        description="Train a patch-wise latent power allocator"
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset", default="ImageNet")
    parser.add_argument("--data_path", default="/dataset/ImageNet")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--allocator_lr", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=float, default=0.5,
                        help="Linear warmup duration for new training, measured in epochs")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Warmup initial LR / peak LR, applied separately to each parameter group")
    parser.add_argument("--min_lr_ratio", type=float, default=0.5,
                        help="Final cosine LR / peak LR; scheduler updates after each optimizer step")
    parser.add_argument("--save_path", default="latest.pth")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Keep a separate checkpoint every N epochs and at the final epoch; 0 keeps latest only")
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--orthogonal_transform", action=argparse.BooleanOptionalAction,
                        default=True, help="Jointly train Q=exp(L-L^T) over 64 blocks")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Block MLP feature/head width; v4 defaults to 256->128->64 and 138->64->1")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--snr_db",
        type=float,
        nargs="+",
        default=[10.0],
        help="Frame SNR values; the paper sweep is -5 0 5 10 15 20",
    )
    parser.add_argument(
        "--qam_order",
        type=int,
        choices=(64, 256),
        default=256,
    )
    parser.add_argument(
        "--transmission_mode",
        choices=("analog", "qam"),
        default="analog",
        help="Analog latent transmission by default; qam preserves the old link",
    )
    parser.add_argument("--quantization_amplitude", type=float, default=3.0)
    parser.add_argument("--pilots_per_group", type=int, default=0)
    parser.add_argument(
        "--channel_gain_abs",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fixed channel magnitudes for legacy AWGN; Rayleigh h is sampled automatically",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--condition_classifier_on_snr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not math.isfinite(args.warmup_epochs) or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup_epochs must be in [0,epochs)")
    if not math.isfinite(args.warmup_start_factor) or not 0 <= args.warmup_start_factor <= 1:
        raise ValueError("--warmup_start_factor must be in [0,1]")
    if not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min_lr_ratio must be in [0,1]")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative")
    if args.orthogonal_transform and (
        args.dataset != "ImageNet" or args.num_groups != 64
        or args.transmission_mode != "analog" or args.pilots_per_group != 0
    ):
        raise ValueError("Orthogonal training requires ImageNet, 64 blocks, analog transmission, and no pilots")
    if args.orthogonal_transform and args.image_size % 64:
        raise ValueError("Orthogonal training requires --image_size divisible by 64")
    if args.orthogonal_transform and args.channel_gain_abs != [1.0]:
        raise ValueError("Rayleigh transmission samples a common complex h; remove fixed --channel_gain_abs")
    if len(args.channel_gain_abs) not in (1, args.num_groups):
        raise ValueError(
            "--channel_gain_abs requires either one value or num_groups "
            "values"
        )

    device, distributed, rank, world_size = initialize_distributed(
        args.device,
        args.local_rank,
    )
    is_main_process = rank == 0
    target_name = "snr_" + "-".join(str(value) for value in args.snr_db)
    result_dir = (
        Path("checkpoints_power")
        / args.dataset
        / target_name
        / args.exp_name
    )
    if is_main_process:
        result_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = build_logger(result_dir / "train.log", is_main_process)
    checkpoint_path = result_dir / Path(args.save_path).name

    process_seed = args.seed + rank
    torch.manual_seed(process_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(process_seed)

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        subfolder="vae",
        revision=VAE_MODEL_REVISION,
        torch_dtype=torch.float32,
    ).to(device=device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    trainloader, validloader = construct_dataloaders(
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch,
        normalize=False,
        image_size=args.image_size,
        distributed=distributed,
        seed=args.seed,
    )

    latent_channels = int(vae.config.latent_channels)
    allocator = PowerAllocator(
        latent_channels,
        args.num_groups,
        args.hidden_dim,
        sampling=False,
        orthogonal_transform=args.orthogonal_transform,
        block_dim=latent_channels * (args.image_size // 64) ** 2 if args.orthogonal_transform else 256,
    ).to(device)
    channel_model = RAYLEIGH_CHANNEL_MODEL if args.orthogonal_transform else "awgn"
    if args.orthogonal_transform:
        channel = RayleighFadingChannel(
            allocator.group_rows, allocator.group_cols, args.snr_db[0]
        ).to(device)
    else:
        channel = QuantizedAWGNChannel(
            allocator.group_rows, allocator.group_cols, args.qam_order,
            args.quantization_amplitude, args.snr_db[0], args.channel_gain_abs,
            args.pilots_per_group, args.transmission_mode,
        ).to(device)
    classifier_input_channels = latent_channels * (
        2 if args.condition_classifier_on_snr else 1
    )
    classifier = build_latent_classifier(
        args.model,
        classifier_input_channels,
        args.num_classes,
    ).to(device)

    start_epoch = 0
    global_step = 0
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        checkpoint_mode = checkpoint["config"].get(
            "transmission_mode",
            "qam",
        )
        if checkpoint_mode != args.transmission_mode:
            raise ValueError(
                f"resume checkpoint uses {checkpoint_mode}, "
                f"but --transmission_mode is {args.transmission_mode}"
            )
        if bool(checkpoint["config"].get("orthogonal_transform", False)) != args.orthogonal_transform:
            raise ValueError("Resume checkpoint uses a different orthogonal-transform setting; start a new experiment")
        if args.orthogonal_transform and checkpoint["config"].get("allocator_architecture") != ALLOCATOR_ARCHITECTURE:
            raise ValueError("Resume checkpoint does not use the v4 block-MLP allocator; start a new experiment")
        if checkpoint["config"].get("channel_model", "awgn") != channel_model:
            raise ValueError("Resume checkpoint uses a different channel model; start a new Rayleigh experiment")
        if args.orthogonal_transform and checkpoint["config"].get(
            "noise_real_component_variance", "N0/2"
        ) != NOISE_REAL_COMPONENT_VARIANCE:
            raise ValueError("Resume checkpoint uses a different noise variance; start a new N0-noise experiment")
        if args.orthogonal_transform and checkpoint["config"].get(
            "mean_real_component_power", 0.5
        ) != MEAN_REAL_COMPONENT_POWER:
            raise ValueError("Resume checkpoint uses a different power definition; start a new per-real-component P=1 experiment")
        allocator.load_state_dict(checkpoint["allocator"])
        classifier.load_state_dict(checkpoint["classifier"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    if distributed:
        ddp_options = {}
        if device.type == "cuda":
            ddp_options = {
                "device_ids": [args.local_rank],
                "output_device": args.local_rank,
            }
        allocator = DistributedDataParallel(allocator, **ddp_options)
        classifier = DistributedDataParallel(classifier, **ddp_options)

    allocator_parameters = list(allocator.parameters())
    classifier_parameters = list(classifier.parameters())
    parameters = allocator_parameters + classifier_parameters
    allocator_lr = (
        args.lr if args.allocator_lr is None else args.allocator_lr
    )
    optimizer = optim.AdamW(
        [
            {
                "params": allocator_parameters,
                "lr": allocator_lr,
            },
            {"params": classifier_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = build_training_scheduler(
        optimizer, args.epochs, len(trainloader), args.warmup_epochs,
        args.warmup_start_factor, args.min_lr_ratio, checkpoint,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if checkpoint is not None:
        if amp_enabled and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

    config = vars(args).copy()
    config.update(
        {
            "stage": "power_allocator",
            "channel_model": channel_model,
            "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE if args.orthogonal_transform else "N0/2",
            "mean_real_component_power": MEAN_REAL_COMPONENT_POWER if args.orthogonal_transform else 0.5,
            "allocator_architecture": unwrap_model(allocator).allocator_architecture,
            "allocator_block_dim": unwrap_model(allocator).block_dim,
            "allocator_embedding_dim": 8 if args.orthogonal_transform else None,
            "allocator_parameter_count": (
                sum(p.numel() for p in unwrap_model(allocator).block_allocator.parameters())
                if args.orthogonal_transform else sum(p.numel() for p in allocator.parameters())
            ),
            "perfect_csi": True,
            "fading_granularity": "one_complex_h_per_image_shared_by_all_blocks" if args.orthogonal_transform else None,
            "fading_distribution": "(randn(B)+1j*randn(B))/sqrt(2)" if args.orthogonal_transform else None,
            "allocator_snr_conditioning": "log_gamma=ln(abs(h)^2*P/N0)" if args.orthogonal_transform else "snr_db",
            "channel_gain_abs": None if args.orthogonal_transform else args.channel_gain_abs,
            "latent_channels": latent_channels,
            "vae_latent_method": "sample",
            "orthogonal_parameterization": "Q=exp(L-L^T), L strictly upper triangular" if args.orthogonal_transform else None,
            "transmitter_order": "patchify -> Q^T -> CSI_conditioned_allocator -> Rayleigh_channel" if args.orthogonal_transform else "allocator -> channel",
            "task_model_input": "Q_applied_to_gain_corrected_received_latent" if args.orthogonal_transform else "received_latent",
            "training_uses_diffusion": False,
            "training_receiver_mode": "oracle_gain",
            "transmission_mode": args.transmission_mode,
            "vae_model_id": VAE_MODEL_ID,
            "vae_model_revision": VAE_MODEL_REVISION,
            "vae_scale": VAE_SCALE,
            "checkpoint_path": str(checkpoint_path),
            "distributed": distributed,
            "world_size": world_size,
            "global_batch_size": args.batch * world_size,
            "lr_schedule": training_schedule_config(scheduler),
            "lr_steps_per_epoch": len(trainloader),
            "power_definition": (
                "sum_b a_b^2 ||U_b||^2 = N_real * P, P=1"
                if args.orthogonal_transform else "sum_g a_g^2 (||u_g||^2 + n_p) = N_f"
            ),
            "real_quantization_levels": (
                channel.real_levels if args.transmission_mode == "qam" else None
            ),
            "quantization_delta": (
                channel.delta if args.transmission_mode == "qam" else None
            ),
            "qam_normalizer": (
                channel.qam_normalizer
                if args.transmission_mode == "qam"
                else None
            ),
            "kappa_m": channel.kappa_m,
            "quantization_alphabet": (
                "-A + 2Al/(L-1), l=0,...,L-1"
                if args.transmission_mode == "qam"
                else None
            ),
        }
    )
    if is_main_process:
        save_config(result_dir / "config.json", config)
    logger.info(
        "experiment=%s result_dir=%s device=%s world_size=%d",
        args.exp_name,
        result_dir,
        device,
        world_size,
    )
    logger.info("lr_schedule=%s resumed=%s next_lr_allocator=%.8g next_lr_classifier=%.8g",
                config["lr_schedule"], checkpoint is not None,
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
    logger.info(
        "channel_model=%s transmission_mode=%s snr_db=%s N0=%s groups=%d M=%s L=%s A=%s "
        "delta=%s kappa_m=%.6f pilots_per_group=%d "
        "fixed_channel_gain_abs=%s",
        channel_model,
        args.transmission_mode,
        args.snr_db,
        [10.0 ** (-value / 10.0) for value in args.snr_db],
        args.num_groups,
        args.qam_order if args.transmission_mode == "qam" else None,
        channel.real_levels if args.transmission_mode == "qam" else None,
        (
            args.quantization_amplitude
            if args.transmission_mode == "qam"
            else None
        ),
        channel.delta if args.transmission_mode == "qam" else None,
        channel.kappa_m,
        args.pilots_per_group,
        config["channel_gain_abs"],
    )
    if args.orthogonal_transform:
        logger.info("perfect_csi=true h~CN(0,1) one_h_per_image_shared_by_64_blocks "
                    "allocator_condition=log_gamma=ln(abs(h)^2*P/N0) receiver_equalization=complex_h")

    for epoch in range(start_epoch, args.epochs):
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)
        allocator.train()
        classifier.train()
        train_totals = torch.zeros(5, device=device, dtype=torch.float64)
        power_sum = torch.zeros(args.num_groups, device=device)
        allocation_sum = torch.zeros(args.num_groups, device=device)

        logger.info(
            "epoch=%d/%d train_start batches=%d",
            epoch + 1,
            args.epochs,
            len(trainloader),
        )
        for batch_index, (images, labels) in enumerate(trainloader):
            batch_lrs = [group["lr"] for group in optimizer.param_groups]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            snr_db = sample_snr_db(
                images.shape[0],
                device,
                args.snr_db,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                result = training_step(
                    images,
                    labels,
                    vae,
                    allocator,
                    classifier,
                    channel,
                    snr_db,
                    args.temperature,
                    args.condition_classifier_on_snr,
                )

            finite_loss = torch.isfinite(result["loss"]).to(torch.int32)
            if distributed:
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if not finite_loss.item():
                raise FloatingPointError("non-finite training loss")
            scaler.scale(result["loss"]).backward()
            optimizer_updated = optimizer_step_with_schedule(
                optimizer, scheduler, scaler, parameters, args.grad_clip,
            )

            batch_size = images.shape[0]
            global_step += 1
            train_totals += torch.tensor(
                [
                    result["loss"].item() * batch_size,
                    result["task_loss"].item() * batch_size,
                    result["mean_symbol_power"].mean().item() * batch_size,
                    result["top1_accuracy"].item() * batch_size,
                    batch_size,
                ],
                device=device,
                dtype=torch.float64,
            )
            power_sum += result["group_power"].sum(dim=0)
            allocation_sum += result["allocation_ratio"].sum(dim=0)

            if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                logger.info(
                    "epoch=%d/%d batch=%d/%d loss=%.4f top1=%.4f "
                    "frame_snr_db=%.2f N0=%.6f mean_symbol_power=%.6f mean_real_component_power=%.6f "
                    "frame_energy=%.2f gain_power_min=%.6f "
                    "gain_power_max=%.6f latent_snr_db=%.2f "
                    "lr_allocator=%.8g lr_classifier=%.8g optimizer_updated=%s",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(trainloader),
                    result["loss"].item(),
                    result["top1_accuracy"].item(),
                    snr_db.mean().item(),
                    torch.pow(10.0, -snr_db / 10.0).mean().item(),
                    result["mean_symbol_power"].mean().item(),
                    result["mean_real_component_power"].mean().item(),
                    result["frame_energy"].mean().item(),
                    result["group_power"].min().item(),
                    result["group_power"].max().item(),
                    (10.0 * torch.log10(result["snr"])).mean().item(),
                    batch_lrs[0], batch_lrs[1], optimizer_updated,
                )

        if distributed:
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(power_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(allocation_sum, op=dist.ReduceOp.SUM)
        train_samples = int(train_totals[4].item())

        allocator.eval()
        classifier.eval()
        validation_totals = torch.zeros(7, device=device, dtype=torch.float64)
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(process_seed)
        with torch.no_grad():
            for images, labels in validloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                snr_db = sample_snr_db(
                    images.shape[0],
                    device,
                    args.snr_db,
                    validation_generator,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    result = validation_step(
                        images,
                        labels,
                        vae,
                        allocator,
                        classifier,
                        channel,
                        snr_db,
                        validation_generator,
                        args.temperature,
                        args.condition_classifier_on_snr,
                    )
                batch_size = images.shape[0]
                validation_totals += torch.tensor(
                    [
                        result["loss"].item() * batch_size,
                        result["task_loss"].item() * batch_size,
                        result["top1_accuracy"].item() * batch_size,
                        result["top5_accuracy"].item() * batch_size,
                        result["mean_symbol_power"].item() * batch_size,
                        result["mean_latent_snr_db"].item() * batch_size,
                        batch_size,
                    ],
                    device=device,
                    dtype=torch.float64,
                )

        if distributed:
            dist.all_reduce(validation_totals, op=dist.ReduceOp.SUM)
        validation_samples = int(validation_totals[-1].item())
        if not isinstance(scheduler, WarmupCosineLR):
            scheduler.step()

        train_mean = train_totals[:4] / train_samples
        validation_mean = validation_totals[:6] / validation_samples
        logger.info(
            "epoch=%d/%d train_loss=%.4f train_top1=%.4f "
            "train_symbol_power=%.6f val_loss=%.4f val_top1=%.4f "
            "val_top5=%.4f val_symbol_power=%.6f "
            "val_latent_snr_db=%.2f",
            epoch + 1,
            args.epochs,
            train_mean[0].item(),
            train_mean[3].item(),
            train_mean[2].item(),
            validation_mean[0].item(),
            validation_mean[2].item(),
            validation_mean[3].item(),
            validation_mean[4].item(),
            validation_mean[5].item(),
        )
        logger.info(
            "epoch=%d mean_gain_power_min=%.6f "
            "mean_gain_power_max=%.6f allocation_min=%.6f "
            "allocation_max=%.6f",
            epoch + 1,
            (power_sum / train_samples).min().item(),
            (power_sum / train_samples).max().item(),
            (allocation_sum / train_samples).min().item(),
            (allocation_sum / train_samples).max().item(),
        )

        if is_main_process:
            checkpoint_state = {
                "allocator": unwrap_model(allocator).state_dict(),
                "classifier": unwrap_model(classifier).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
            }
            for saved_path in save_epoch_checkpoints(
                checkpoint_state, checkpoint_path, args.save_every, args.epochs
            ):
                logger.info("checkpoint_saved path=%s", saved_path)
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
