import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from datasets.data_processing import construct_dataloaders
from inference import StaggeredReverseDiffusion
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    VAE_SCALE,
    PowerAllocator,
    QuantizedAWGNChannel,
    _merge_spatial_groups,
    _split_spatial_groups,
    allocate_power,
    build_latent_classifier,
    classifier_forward,
    encode_latents,
)
from power_utils import sd15_alpha_bars
from reconstruction_metrics import LPIPS_NETWORK, image_quality, load_lpips


def allocate_power_for_inference(
    z0, snr_db, allocator, channel, temperature=1.0,
    channel_coefficient=None, min_gain=0.0, max_timestep=None,
):
    """Reserve gain floors, then distribute residual power as before.

    max_timestep bounds the SD1.5-equivalent observation noise, not the DPS
    sampling schedule. An infeasible timestep floor uses uniform gain at the
    same frame budget and is recorded in gain_floor_fallback.
    """
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and non-negative")
    if max_timestep is not None and (
        not isinstance(max_timestep, int) or not 0 <= max_timestep <= 999
    ):
        raise ValueError("max_timestep must be an integer in [0,999]")
    ratio, power, info = allocate_power(
        z0, snr_db, allocator, channel, temperature,
        channel_coefficient=channel_coefficient,
    )
    if min_gain == 0 and max_timestep is None:
        return ratio, power, info
    energy = info["unscaled_block_energy"]
    budget = info["target_frame_energy"]
    total_energy = energy.sum(dim=1, keepdim=True)
    if (min_gain**2 * total_energy > budget).any():
        maximum = (budget / energy.sum(dim=1)).sqrt().min().item()
        raise ValueError(
            f"--min_gain {min_gain:g} exceeds the frame-energy budget; "
            f"use a value <= {maximum:.6g} for this batch"
        )
    minimum_power = torch.full_like(power, min_gain**2)
    if max_timestep is not None:
        alpha = sd15_alpha_bars().to(device=power.device)[max_timestep]
        maximum_variance = (1.0 - alpha) / alpha
        n0 = channel._noise_spectral_density(z0, snr_db)
        if channel_coefficient is not None:
            # Current perfect-CSI Rayleigh/AWGN link: real noise variance N0.
            variance_before_gain = n0[:, None] / channel_coefficient.abs().square()[:, None]
        else:
            # Legacy transport: N0/2, with the original quantizer scaling.
            h = channel._channel_gains(power, None)
            variance_before_gain = channel.kappa_m_squared * n0[:, None] / (2.0 * h.abs().square())
        minimum_power = torch.maximum(minimum_power, variance_before_gain / maximum_variance)
    required = (minimum_power * energy).sum(dim=1, keepdim=True)
    fallback = required > budget
    uniform_power = (budget / total_energy).expand_as(power)
    minimum_power = torch.where(fallback, uniform_power, minimum_power)
    remaining = (budget - (minimum_power * energy).sum(dim=1, keepdim=True)).clamp_min(0)
    # power is a_b^2 and already proportional to the allocator's softmax.
    power = minimum_power + (remaining / budget) * power
    power = torch.where(fallback, uniform_power, power)
    frame_energy = (power * energy).sum(dim=1)
    info.update(
        frame_energy=frame_energy,
        mean_symbol_power=frame_energy / info["frame_channel_uses"],
        mean_real_component_power=frame_energy / (2 * info["frame_channel_uses"]),
        gain_floor_fallback=fallback.squeeze(1),
        max_timestep=max_timestep,
    )
    return power * energy / budget, power, info


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _decode_latent(
    vae: AutoencoderKL,
    latent: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    vae_parameter = next(vae.parameters())
    latent = latent.to(
        device=vae_parameter.device,
        dtype=vae_parameter.dtype,
    )
    decoded = vae.decode(latent / scaling_factor).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def _save_labeled_row(
    images: list[torch.Tensor],
    labels: list[str],
    output_path: Path,
) -> None:
    pil_images = [
        to_pil_image(image.detach().float().cpu().clamp(0.0, 1.0))
        for image in images
    ]
    panel_width, panel_height = pil_images[0].size
    label_height = 24
    canvas = Image.new(
        "RGB",
        (panel_width * len(pil_images), panel_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(pil_images, labels)):
        left = index * panel_width
        canvas.paste(image, (left, label_height))
        draw.text((left + 4, 5), label, fill="black")
    canvas.save(output_path)


def _load_image(
    args: argparse.Namespace,
    config: dict,
) -> tuple[torch.Tensor, int | None]:
    image_size = int(config.get("image_size", 512))
    if args.input_image is not None:
        image = Image.open(args.input_image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        return transform(image).unsqueeze(0), None

    _, validloader = construct_dataloaders(
        dataset=config["dataset"],
        data_path=args.data_path or config["data_path"],
        batch_size=1,
        normalize=False,
        image_size=image_size,
    )
    image, label = validloader.dataset[args.sample_index]
    return image.unsqueeze(0), int(label)


def _task_result(
    logits: torch.Tensor,
    label: int | None,
) -> dict:
    probabilities = logits.softmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(5, dim=1)
    result = {
        "top5_indices": top_indices[0].cpu().tolist(),
        "top5_probabilities": top_probabilities[0].cpu().tolist(),
    }
    if label is not None:
        top1_correct = bool(top_indices[0, 0].item() == label)
        top5_correct = bool((top_indices[0] == label).any().item())
        result.update(
            {
                "label": label,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "top1_accuracy": float(top1_correct),
                "top5_accuracy": float(top5_correct),
            }
        )
    return result


class ActualPowerTransport:
    def __init__(
        self,
        channel: QuantizedAWGNChannel,
        receiver_mode: str,
    ):
        if receiver_mode == "pilot" and channel.pilots_per_group == 0:
            raise ValueError("pilot receiver requires at least one pilot")
        self.channel = channel
        self.receiver_mode = receiver_mode

    def _pilots(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        base = torch.tensor(
            [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j],
            device=device,
            dtype=torch.complex64,
        )
        repeats = (self.channel.pilots_per_group + 3) // 4
        return base.repeat(repeats)[:self.channel.pilots_per_group]

    @staticmethod
    def _complex_noise(
        shape: tuple[int, ...],
        noise_spectral_density: torch.Tensor,
        reference: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = torch.randn(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        imaginary = torch.randn(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        standard_complex = torch.complex(real, imaginary)
        standard_deviation = torch.sqrt(
            noise_spectral_density / 2.0
        )
        return standard_deviation[:, None] * standard_complex

    def __call__(
        self,
        z0: torch.Tensor,
        group_power: torch.Tensor,
        snr_db: torch.Tensor,
        generator: torch.Generator | None,
        pool_pilots: bool = False,
    ) -> tuple[torch.Tensor, dict, dict]:
        transmitted_latent = self.channel.prepare_latent(z0)
        transmitted_groups = _split_spatial_groups(
            transmitted_latent,
            self.channel.group_rows,
            self.channel.group_cols,
        )
        channel_gains = self.channel._channel_gains(
            group_power,
            None,
        )
        noise_spectral_density = self.channel._noise_spectral_density(
            z0,
            snr_db,
        )
        pilots = self._pilots(z0.device)

        received_payloads = []
        true_effective_gains = []
        per_patch_gain_estimates = []
        block_energies = []
        transmitted_blocks = []
        received_blocks = []

        for group_index, group in enumerate(transmitted_groups):
            flattened = group.reshape(group.shape[0], -1)
            payload = torch.complex(
                flattened[:, 0::2],
                flattened[:, 1::2],
            ) / self.channel.kappa_m
            amplitude_gain = torch.sqrt(group_power[:, group_index])
            fading_gain = channel_gains[:, group_index]
            effective_gain = torch.complex(
                amplitude_gain * fading_gain,
                torch.zeros_like(amplitude_gain),
            )

            transmitted_pilots = amplitude_gain[:, None] * pilots[None]
            transmitted_payload = amplitude_gain[:, None] * payload
            block_energies.append(
                transmitted_pilots.abs().square().sum(dim=1)
                + transmitted_payload.abs().square().sum(dim=1)
            )

            pilot_noise = self._complex_noise(
                tuple(transmitted_pilots.shape),
                noise_spectral_density,
                z0,
                generator,
            )
            payload_noise = self._complex_noise(
                tuple(transmitted_payload.shape),
                noise_spectral_density,
                z0,
                generator,
            )
            received_pilots = (
                fading_gain[:, None] * transmitted_pilots + pilot_noise
            )
            received_payload = (
                fading_gain[:, None] * transmitted_payload + payload_noise
            )
            transmitted_blocks.append(
                torch.cat([transmitted_pilots, transmitted_payload], dim=1)
            )
            received_blocks.append(
                torch.cat([received_pilots, received_payload], dim=1)
            )
            if self.channel.pilots_per_group == 0:
                estimated_gain = effective_gain
            else:
                estimated_gain = (
                    pilots.conj()[None] * received_pilots
                ).mean(dim=1)

            received_payloads.append(received_payload)
            true_effective_gains.append(effective_gain)
            per_patch_gain_estimates.append(estimated_gain)

        true_effective_gain = torch.stack(true_effective_gains, dim=1)
        per_patch_gain_estimate = torch.stack(
            per_patch_gain_estimates,
            dim=1,
        )
        if self.receiver_mode == "oracle":
            equalization_gain = true_effective_gain
            gain_power = equalization_gain.abs().square()
            pilot_pooling = "oracle_gain"
            pilot_count_for_estimate = 0
        elif pool_pilots:
            if not torch.allclose(
                true_effective_gain,
                true_effective_gain[:, :1].expand_as(true_effective_gain),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "pilot pooling requires one common effective gain"
                )
            pooled_gain = per_patch_gain_estimate.mean(
                dim=1,
                keepdim=True,
            )
            equalization_gain = pooled_gain.expand_as(true_effective_gain)
            gain_power = equalization_gain.abs().square()
            pilot_pooling = "all_frame_pilots"
            pilot_count_for_estimate = (
                len(transmitted_groups) * self.channel.pilots_per_group
            )
        else:
            equalization_gain = per_patch_gain_estimate
            gain_power = equalization_gain.abs().square()
            pilot_pooling = "per_patch"
            pilot_count_for_estimate = self.channel.pilots_per_group

        numerical_reliability = gain_power > 1e-12
        if self.receiver_mode == "oracle":
            pilot_estimation_snr = None
        else:
            noise_power = noise_spectral_density[:, None]
            pilot_estimation_snr = (
                pilot_count_for_estimate * gain_power / noise_power
            )
        reliability = numerical_reliability
        if not reliability.all():
            failed_index = (~reliability).nonzero()[0]
            failed_sample = int(failed_index[0].item())
            failed_patch = int(failed_index[1].item())
            raise RuntimeError(
                "unreliable gain estimate in "
                f"sample {failed_sample}, patch {failed_patch}"
            )

        received_groups = []
        signal_powers = []
        empirical_mse = []
        for group_index, (group, received_payload) in enumerate(
            zip(transmitted_groups, received_payloads)
        ):
            flattened = group.reshape(group.shape[0], -1)
            group_equalization_gain = equalization_gain[:, group_index]
            equalized_payload = (
                received_payload / group_equalization_gain[:, None]
            )
            recovered_flattened = torch.empty_like(flattened)
            recovered_flattened[:, 0::2] = (
                self.channel.kappa_m * equalized_payload.real
            )
            recovered_flattened[:, 1::2] = (
                self.channel.kappa_m * equalized_payload.imag
            )
            received_group = recovered_flattened.reshape_as(group)
            received_groups.append(received_group)
            signal_powers.append(group.square().mean(dim=(1, 2, 3)))
            empirical_mse.append(
                (received_group - group).square().mean(dim=(1, 2, 3))
            )

        received_payload_power = torch.stack(
            [value.abs().square().mean(dim=1) for value in received_payloads],
            dim=1,
        )
        if self.receiver_mode == "pilot" and pool_pilots:
            payload_symbol_count = sum(
                value.shape[1] for value in received_payloads
            )
            frame_received_payload_power = sum(
                value.abs().square().sum(dim=1)
                for value in received_payloads
            ) / payload_symbol_count
            pooled_power_estimate = (
                (frame_received_payload_power - noise_spectral_density)
                / gain_power[:, 0]
            ).clamp_min(0.0)
            payload_power_estimate = pooled_power_estimate[:, None].expand_as(
                gain_power
            )
            total_pilots = (
                len(transmitted_groups) * self.channel.pilots_per_group
            )
            noise_variance = (
                self.channel.kappa_m_squared
                * noise_spectral_density[:, None]
                / (2.0 * gain_power)
                * (1.0 + payload_power_estimate / total_pilots)
            )
        else:
            payload_power_estimate = (
                (received_payload_power - noise_spectral_density[:, None])
                / gain_power
            ).clamp_min(0.0)
            noise_variance = (
                self.channel.kappa_m_squared
                * noise_spectral_density[:, None]
                / (2.0 * gain_power)
            )
            if self.receiver_mode == "pilot":
                noise_variance = noise_variance * (
                    1.0
                    + payload_power_estimate
                    / self.channel.pilots_per_group
                )

        received_latent = _merge_spatial_groups(
            received_groups,
            self.channel.group_rows,
            self.channel.group_cols,
        )
        signal_power = torch.stack(signal_powers, dim=1)
        if (
            self.receiver_mode == "pilot"
            and self.channel.transmission_mode == "analog"
        ):
            local_payload_power_estimate = (
                (received_payload_power - noise_spectral_density[:, None])
                / gain_power
            ).clamp_min(0.0)
            signal_power = (
                self.channel.kappa_m_squared
                * local_payload_power_estimate
                / 2.0
            )
        snr = (signal_power / noise_variance).clamp_min(1e-8)
        log_snr = torch.log(snr)
        snr_groups = [
            log_snr[:, group_index, None, None, None].expand_as(group)
            for group_index, group in enumerate(transmitted_groups)
        ]
        snr_map = _merge_spatial_groups(
            snr_groups,
            self.channel.group_rows,
            self.channel.group_cols,
        )
        block_energy = torch.stack(block_energies, dim=1)
        transmitted_frame = torch.cat(transmitted_blocks, dim=1)
        received_frame = torch.cat(received_blocks, dim=1)
        channel_info = {
            "group_power": group_power,
            "amplitude_gain": torch.sqrt(group_power),
            "channel_gain_abs": channel_gains,
            "signal_power": signal_power,
            "noise_variance": noise_variance,
            "noise_spectral_density": noise_spectral_density,
            "snr": snr,
            "log_snr": log_snr,
            "snr_map": snr_map,
            "quantization_delta": (
                self.channel.delta
                if self.channel.transmission_mode == "qam"
                else None
            ),
            "kappa_m": self.channel.kappa_m,
        }
        transport_info = {
            "receiver_mode": self.receiver_mode,
            "pilot_pooling": pilot_pooling,
            "frame_channel_uses": transmitted_frame.shape[1],
            "payload_symbols_per_group": payload.shape[1],
            "pilot_symbols_per_group": self.channel.pilots_per_group,
            "pilot_symbols_used_for_estimate": pilot_count_for_estimate,
            "pilot_estimation_snr": pilot_estimation_snr,
            "pilot_reliability_threshold": None,
            "pilot_reliability_enforced": False,
            "transmit_frame_energy": block_energy.sum(dim=1),
            "transmit_block_energy": block_energy,
            "true_effective_gain_abs": true_effective_gain.abs(),
            "estimated_effective_gain_abs": equalization_gain.abs(),
            "per_patch_pilot_estimate_abs": (
                per_patch_gain_estimate.abs()
            ),
            "payload_power_estimate": payload_power_estimate,
            "empirical_latent_mse": torch.stack(empirical_mse, dim=1),
            "pilot_reliability": reliability,
            "transmitted_frame": transmitted_frame,
            "received_frame": received_frame,
            "transmitted_latent": transmitted_latent,
            "received_latent": received_latent,
        }
        return received_latent, channel_info, transport_info


def _strategy_result(
    group_power: torch.Tensor,
    allocation_ratio: torch.Tensor,
    frame_energy: torch.Tensor,
    frame_channel_uses: int,
    channel_info: dict,
    transport_info: dict,
    logits: torch.Tensor,
    label: int | None,
) -> dict:
    latent_snr_db = 10.0 * torch.log10(channel_info["snr"])
    return {
        "task": _task_result(logits, label),
        "frame_energy": frame_energy.item(),
        "mean_symbol_power": (
            frame_energy / frame_channel_uses
        ).item(),
        "gain_power_min": group_power.min().item(),
        "gain_power_max": group_power.max().item(),
        "gain_power": group_power[0].cpu().tolist(),
        "amplitude_gain": torch.sqrt(group_power[0]).cpu().tolist(),
        "energy_allocation_ratio": allocation_ratio[0].cpu().tolist(),
        "noise_variance": channel_info["noise_variance"][0].cpu().tolist(),
        "latent_snr_db_mean": latent_snr_db.mean().item(),
        "latent_snr_db_min": latent_snr_db.min().item(),
        "latent_snr_db_max": latent_snr_db.max().item(),
        "latent_snr_db": latent_snr_db[0].cpu().tolist(),
        "transport": {
            "receiver_mode": transport_info["receiver_mode"],
            "pilot_pooling": transport_info["pilot_pooling"],
            "frame_channel_uses": transport_info["frame_channel_uses"],
            "payload_symbols_per_group": transport_info[
                "payload_symbols_per_group"
            ],
            "pilot_symbols_per_group": transport_info[
                "pilot_symbols_per_group"
            ],
            "pilot_symbols_used_for_estimate": transport_info[
                "pilot_symbols_used_for_estimate"
            ],
            "pilot_estimation_snr": (
                None
                if transport_info["pilot_estimation_snr"] is None
                else transport_info["pilot_estimation_snr"][0]
                .cpu()
                .tolist()
            ),
            "pilot_reliability_threshold": transport_info[
                "pilot_reliability_threshold"
            ],
            "pilot_reliability_enforced": transport_info[
                "pilot_reliability_enforced"
            ],
            "transmit_frame_energy": transport_info[
                "transmit_frame_energy"
            ].item(),
            "transmit_block_energy": transport_info[
                "transmit_block_energy"
            ][0].cpu().tolist(),
            "true_effective_gain_abs": transport_info[
                "true_effective_gain_abs"
            ][0].cpu().tolist(),
            "estimated_effective_gain_abs": transport_info[
                "estimated_effective_gain_abs"
            ][0].cpu().tolist(),
            "per_patch_pilot_estimate_abs": transport_info[
                "per_patch_pilot_estimate_abs"
            ][0].cpu().tolist(),
            "payload_power_estimate": transport_info[
                "payload_power_estimate"
            ][0].cpu().tolist(),
            "empirical_latent_mse": transport_info[
                "empirical_latent_mse"
            ][0].cpu().tolist(),
            "pilot_reliability": transport_info[
                "pilot_reliability"
            ][0].cpu().tolist(),
        },
    }


def _match_diffusion_start(
    received_latent: torch.Tensor,
    noise_variance: torch.Tensor,
    channel: QuantizedAWGNChannel,
    matching_epsilon: torch.Tensor,
    available_timesteps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict, dict]:
    library_alpha_bars = sd15_alpha_bars().to(
        device=received_latent.device,
        dtype=received_latent.dtype,
    )
    alpha_bars = torch.cat(
        [library_alpha_bars.new_ones(1), library_alpha_bars]
    )
    lambda_schedule = (1.0 - alpha_bars) / alpha_bars
    unsupported = noise_variance > lambda_schedule[-1]
    bounded_noise_variance = noise_variance.clamp_max(
        lambda_schedule[-1]
    )

    raw_mathematical_timesteps = torch.searchsorted(
        lambda_schedule,
        bounded_noise_variance.contiguous(),
        right=False,
    )
    raw_schedule_timesteps = raw_mathematical_timesteps - 1
    if available_timesteps is None:
        schedule_timesteps = raw_schedule_timesteps
        mathematical_timesteps = raw_mathematical_timesteps
    else:
        available_timesteps = torch.unique(
            available_timesteps.to(
                device=received_latent.device,
                dtype=torch.long,
            ),
            sorted=True,
        )
        if available_timesteps.numel() == 0:
            raise ValueError("available_timesteps must not be empty")
        # Use the closest DDIM grid point that is not less noisy than the
        # observation. Mapping downward would require removing channel noise.
        insertion_indices = torch.searchsorted(
            available_timesteps,
            raw_schedule_timesteps.clamp_min(0).contiguous(),
            right=False,
        )
        unavailable = insertion_indices >= available_timesteps.numel()
        insertion_indices = insertion_indices.clamp_max(
            available_timesteps.numel() - 1
        )
        schedule_timesteps = available_timesteps[insertion_indices]
        mathematical_timesteps = schedule_timesteps + 1
        unsupported = unsupported | unavailable
    selected_alpha_bars = alpha_bars[mathematical_timesteps]
    selected_lambdas = lambda_schedule[mathematical_timesteps]
    matching_variance = (
        selected_lambdas - bounded_noise_variance
    ).clamp_min(0.0)

    received_groups = _split_spatial_groups(
        received_latent,
        channel.group_rows,
        channel.group_cols,
    )
    epsilon_groups = _split_spatial_groups(
        matching_epsilon,
        channel.group_rows,
        channel.group_cols,
    )
    matched_groups = []
    for group_index, (group, epsilon_group) in enumerate(
        zip(received_groups, epsilon_groups)
    ):
        alpha = selected_alpha_bars[:, group_index, None, None, None]
        extra_variance = matching_variance[
            :, group_index, None, None, None
        ]
        matched_group = (
            torch.sqrt(alpha)
            * (group + torch.sqrt(extra_variance) * epsilon_group)
        )
        # If the observation is noisier than t=999, normalize it using its
        # actual variance and start from t=999. This keeps the received signal
        # instead of dropping the patch, while saturating only the timestep.
        actual_variance = noise_variance[
            :, group_index, None, None, None
        ]
        saturated_group = group * torch.rsqrt(1.0 + actual_variance)
        matched_groups.append(
            torch.where(
                unsupported[:, group_index, None, None, None],
                saturated_group,
                matched_group,
            )
        )
    matched_latent = _merge_spatial_groups(
        matched_groups,
        channel.group_rows,
        channel.group_cols,
    )

    reverse_context = {
        "target_timesteps": schedule_timesteps,
        "effective_alpha_bar": selected_alpha_bars,
        "group_rows": channel.group_rows,
        "group_cols": channel.group_cols,
    }
    matching_info = {
        "unmapped_mathematical_timestep": (
            raw_mathematical_timesteps[0].cpu().tolist()
        ),
        "unmapped_schedule_timestep": (
            raw_schedule_timesteps[0].cpu().tolist()
        ),
        "mathematical_timestep": mathematical_timesteps[0].cpu().tolist(),
        "unet_timestep": schedule_timesteps[0].cpu().tolist(),
        "schedule_timestep": schedule_timesteps[0].cpu().tolist(),
        "alpha_bar": selected_alpha_bars[0].cpu().tolist(),
        "matching_noise_variance": matching_variance[0].cpu().tolist(),
        "unsupported_patch_mask": unsupported[0].cpu().tolist(),
        "unsupported_patch_count": int(unsupported.sum().item()),
        "timestep_min": schedule_timesteps.min().item(),
        "timestep_max": schedule_timesteps.max().item(),
    }
    return matched_latent, reverse_context, matching_info


def main() -> None:
    from inference_orthogonal_power import build_parser, run_inference, validate_arguments

    parser = build_parser(include_legacy=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path /= "latest.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if args.restoration_method is None:
        args.restoration_method = "ddrm" if config.get("orthogonal_transform", False) else "dps"
    validate_arguments(args, parser)
    if config.get("orthogonal_transform", False):
        if args.snr_db is None:
            args.snr_db = 10.
        run_inference(args, checkpoint_path, checkpoint)
        return
    if args.restoration_method == "ddrm":
        parser.error("DDRM requires an orthogonal analog checkpoint; use the Q receiver path")
    if args.receiver_mode is None:
        args.receiver_mode = "pilot"
    if args.receiver_mode == "gain_estimator":
        parser.error("gain_estimator receiver requires an orthogonal ImageNet checkpoint")
    if args.uniform_receiver_mode is not None:
        parser.error("--uniform_receiver_mode requires an orthogonal ImageNet checkpoint")
    lpips_model = load_lpips(device, enabled=args.lpips, local_files_only=args.local_files_only)
    transmission_mode = config.get("transmission_mode", "qam")
    configured_snr = config.get("snr_db", [10.0])
    if isinstance(configured_snr, (int, float)):
        configured_snr = [float(configured_snr)]
    snr_db = args.snr_db if args.snr_db is not None else configured_snr[0]
    channel_gain_abs = (
        args.channel_gain_abs
        if args.channel_gain_abs is not None
        else config.get("channel_gain_abs", [1.0])
    )

    output_dir = Path(
        args.output_dir
        or checkpoint_path.parent
        / "inference"
        / (
            f"sample_{args.sample_index:06d}_snr_{snr_db:g}_"
            f"{args.receiver_mode}"
            + (f"_mingain_{args.min_gain:g}" if args.min_gain > 0 else "")
            + (f"_maxt_{args.max_timestep}" if args.max_timestep is not None else "")
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
    ).to(device=device)
    _freeze(vae)

    latent_channels = int(vae.config.latent_channels)
    allocator = PowerAllocator(
        latent_channels,
        config["num_groups"],
        config["hidden_dim"],
        sampling=False,
    ).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    _freeze(allocator)

    condition_on_snr = bool(config.get("condition_classifier_on_snr", True))
    classifier_input_channels = latent_channels * (
        2 if condition_on_snr else 1
    )
    classifier = build_latent_classifier(
        config.get("model", "ResNet50"),
        classifier_input_channels,
        config.get("num_classes", 1000),
    ).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    _freeze(classifier)

    channel = QuantizedAWGNChannel(
        allocator.group_rows,
        allocator.group_cols,
        config.get("qam_order", 256),
        config.get("quantization_amplitude", 3.0),
        snr_db,
        channel_gain_abs,
        config.get("pilots_per_group", 4),
        transmission_mode,
    ).to(device)
    transport = ActualPowerTransport(channel, args.receiver_mode)

    image, label = _load_image(args, config)
    image = image.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    vae_scale = float(config.get("vae_scale", VAE_SCALE))

    with torch.inference_mode():
        z0 = encode_latents(image, vae, generator)
        vae_reconstruction = _decode_latent(vae, z0, vae_scale)
        snr = torch.full((1,), snr_db, device=device)
        allocated_ratio, allocated_power, allocation_info = allocate_power_for_inference(
            z0,
            snr,
            allocator,
            channel,
            config.get("temperature", 1.0),
            min_gain=args.min_gain,
            max_timestep=args.max_timestep,
        )

        unscaled_energy = allocation_info["unscaled_block_energy"]
        frame_channel_uses = int(allocation_info["frame_channel_uses"])
        uniform_value = frame_channel_uses / unscaled_energy.sum(
            dim=1,
            keepdim=True,
        )
        uniform_power = uniform_value.expand_as(allocated_power)
        uniform_block_energy = uniform_power * unscaled_energy
        uniform_ratio = uniform_block_energy / frame_channel_uses

        channel_generator_state = generator.get_state()
        (
            allocated_latent,
            allocated_channel_info,
            allocated_transport_info,
        ) = transport(
            z0,
            allocated_power,
            snr,
            generator,
        )
        generator.set_state(channel_generator_state)
        (
            uniform_latent,
            uniform_channel_info,
            uniform_transport_info,
        ) = transport(
            z0,
            uniform_power,
            snr,
            generator,
            pool_pilots=True,
        )

        matching_generator = torch.Generator(device=device)
        matching_generator.manual_seed(args.seed + 1)
        matching_epsilon = torch.randn(
            z0.shape,
            device=device,
            dtype=z0.dtype,
            generator=matching_generator,
        )
        allocated_start, allocated_context, allocated_matching = (
            _match_diffusion_start(
                allocated_latent,
                allocated_channel_info["noise_variance"],
                channel,
                matching_epsilon,
            )
        )
        uniform_start, uniform_context, uniform_matching = (
            _match_diffusion_start(
                uniform_latent,
                uniform_channel_info["noise_variance"],
                channel,
                matching_epsilon,
            )
        )
        allocated_logits = classifier_forward(
            classifier,
            allocated_latent,
            allocated_channel_info,
            condition_on_snr,
        )
        uniform_logits = classifier_forward(
            classifier,
            uniform_latent,
            uniform_channel_info,
            condition_on_snr,
        )

        allocated_direct_image = _decode_latent(
            vae,
            allocated_latent,
            vae_scale,
        )
        uniform_direct_image = _decode_latent(
            vae,
            uniform_latent,
            vae_scale,
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
        )
        diffusion_generator = torch.Generator(device=device)
        diffusion_generator.manual_seed(args.seed + 2)
        diffusion_generator_state = diffusion_generator.get_state()
        allocated_reconstructed_latent = reverse_model(
            allocated_start,
            allocated_context,
            diffusion_generator,
        )
        diffusion_generator.set_state(diffusion_generator_state)
        uniform_reconstructed_latent = reverse_model(
            uniform_start,
            uniform_context,
            diffusion_generator,
        )
        allocated_image = _decode_latent(
            vae,
            allocated_reconstructed_latent,
            vae_scale,
        )
        uniform_image = _decode_latent(
            vae,
            uniform_reconstructed_latent,
            vae_scale,
        )
        result = {
            "checkpoint": str(checkpoint_path),
            "epoch": int(checkpoint["epoch"]) + 1,
            "sample_index": args.sample_index,
            "input_image": args.input_image,
            "input_image_size": list(image.shape[-2:]),
            "snr_db": snr_db,
            "noise_spectral_density": 10.0 ** (-snr_db / 10.0),
            "channel_gain_abs": channel_gain_abs,
            "transmission_mode": transmission_mode,
            "qam_order": (
                channel.qam_order if transmission_mode == "qam" else None
            ),
            "quantization_amplitude": (
                channel.quantization_amplitude
                if transmission_mode == "qam"
                else None
            ),
            "quantization_delta": (
                channel.delta if transmission_mode == "qam" else None
            ),
            "kappa_m": channel.kappa_m,
            "pilots_per_group": channel.pilots_per_group,
            "frame_channel_uses": frame_channel_uses,
            "same_noise_for_strategies": True,
            "channel_simulation": (
                "quantized_QAM_pilots_complex_AWGN_soft_equalization"
                if transmission_mode == "qam"
                else "analog_IQ_pilots_complex_AWGN_soft_equalization"
            ),
            "receiver_mode": args.receiver_mode,
            "min_gain": args.min_gain,
            "max_timestep": args.max_timestep,
            "gain_floor_infeasible_images": int(allocation_info.get(
                "gain_floor_fallback", torch.zeros(1, dtype=torch.bool)
            ).sum().item()),
            "task_input": "received_latent_before_diffusion",
            "diffusion_model": diffusion_model,
            "strategies": {
                "allocated": _strategy_result(
                    allocated_power,
                    allocated_ratio,
                    allocated_transport_info["transmit_frame_energy"],
                    frame_channel_uses,
                    allocated_channel_info,
                    allocated_transport_info,
                    allocated_logits,
                    label,
                ),
                "uniform": _strategy_result(
                    uniform_power,
                    uniform_ratio,
                    uniform_transport_info["transmit_frame_energy"],
                    frame_channel_uses,
                    uniform_channel_info,
                    uniform_transport_info,
                    uniform_logits,
                    label,
                ),
            },
        }
        result["strategies"]["allocated"]["diffusion"] = (
            allocated_matching
        )
        result["strategies"]["uniform"]["diffusion"] = uniform_matching
        result["lpips_network"] = LPIPS_NETWORK if lpips_model is not None else None
        result["vae_vs_input"] = image_quality(vae_reconstruction, image, lpips_model)
        for name, direct, restored in (
            ("allocated", allocated_direct_image, allocated_image),
            ("uniform", uniform_direct_image, uniform_image),
        ):
            result["strategies"][name]["direct_vs_input"] = image_quality(direct, image, lpips_model)
            result["strategies"][name]["restored_vs_input"] = image_quality(restored, image, lpips_model)
        if lpips_model is not None:
            print(f"LPIPS: VAE={result['vae_vs_input']['lpips']:.5f}, "
                  f"uniform={result['strategies']['uniform']['restored_vs_input']['lpips']:.5f}, "
                  f"allocated={result['strategies']['allocated']['restored_vs_input']['lpips']:.5f}", flush=True)

    to_pil_image(allocated_direct_image[0].float().cpu()).save(
        output_dir / "direct_reconstructed.png"
    )
    to_pil_image(uniform_direct_image[0].float().cpu()).save(
        output_dir / "uniform_direct_reconstructed.png"
    )
    to_pil_image(allocated_image[0].float().cpu()).save(
        output_dir / "reconstructed.png"
    )
    to_pil_image(uniform_image[0].float().cpu()).save(
        output_dir / "uniform_reconstructed.png"
    )
    torch.save(
        {
            "transmitted_frame": allocated_transport_info[
                "transmitted_frame"
            ].cpu(),
            "received_frame": allocated_transport_info[
                "received_frame"
            ].cpu(),
            "transmitted_latent": allocated_transport_info[
                "transmitted_latent"
            ].cpu(),
            "received_latent": allocated_transport_info[
                "received_latent"
            ].cpu(),
            "group_power": allocated_power.cpu(),
        },
        output_dir / "allocated_baseband.pt",
    )
    torch.save(
        {
            "transmitted_frame": uniform_transport_info[
                "transmitted_frame"
            ].cpu(),
            "received_frame": uniform_transport_info[
                "received_frame"
            ].cpu(),
            "transmitted_latent": uniform_transport_info[
                "transmitted_latent"
            ].cpu(),
            "received_latent": uniform_transport_info[
                "received_latent"
            ].cpu(),
            "group_power": uniform_power.cpu(),
        },
        output_dir / "uniform_baseband.pt",
    )
    _save_labeled_row(
        [
            image[0].float().cpu(),
            allocated_image[0].float().cpu(),
            uniform_image[0].float().cpu(),
        ],
        ["Original", "Allocated power", "Uniform power"],
        output_dir / "comparison.png",
    )
    _save_labeled_row(
        [
            image[0].float().cpu(),
            allocated_direct_image[0].float().cpu(),
            uniform_direct_image[0].float().cpu(),
        ],
        ["Original", "Allocated direct", "Uniform direct"],
        output_dir / "direct_comparison.png",
    )
    with open(output_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Saved inference output to {output_dir}")


if __name__ == "__main__":
    main()
