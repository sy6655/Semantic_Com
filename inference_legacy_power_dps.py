"""Restore a pre-Q ImageNet power-allocator checkpoint with the current DPS.

The learned CNN allocator and legacy analog AWGN transport are retained.
DPS receives the gain-corrected observation, its exact noise precision,
and Q=I. No gain estimator or classifier is loaded.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from inference import StaggeredReverseDiffusion
from inference_oracle_gain import image_metrics, load_input, save_image
from inference_power import (
    ActualPowerTransport, _decode_latent, _freeze, _save_labeled_row,
    allocate_power_for_inference,
)
from orthogonal_power import observation_loss
from posterior_sampling import OrthogonalPosteriorSampling
from reconstruction_metrics import LPIPS_NETWORK, load_lpips
from train_power_allocator import (
    VAE_MODEL_ID, VAE_MODEL_REVISION, VAE_SCALE,
    PowerAllocator, QuantizedAWGNChannel, encode_latents,
)


class LegacyPowerAllocator(PowerAllocator):
    """Keep the SNR input normalization recorded by the old checkpoint."""

    def __init__(self, latent_channels, config):
        super().__init__(latent_channels, config["num_groups"], config["hidden_dim"],
                         sampling=False, orthogonal_transform=False)
        self.snr_offset = float(config.get("allocator_snr_offset", 0.0))
        self.snr_scale = float(config.get("allocator_snr_scale", 1.0))
        if not math.isfinite(self.snr_scale) or self.snr_scale <= 0:
            raise ValueError("allocator_snr_scale must be finite and positive")

    def forward(self, latent, snr_db, return_transformed=False):
        normalized_snr = (snr_db - self.snr_offset) / self.snr_scale
        return super().forward(latent, normalized_snr, return_transformed)


def load_legacy_transmitter(checkpoint_path, device, snr_db, channel_gain_abs=None):
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        if (path / "latest.pth").is_file():
            path /= "latest.pth"
        else:
            candidates = sorted(path.glob("*/latest.pth"))
            if len(candidates) != 1:
                raise ValueError("Specify a checkpoint file or a directory containing one latest.pth")
            path = candidates[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("dataset") != "ImageNet" or config.get("orthogonal_transform", False):
        raise ValueError("Expected a pre-Q ImageNet power allocator checkpoint")
    if config.get("stage") != "power_allocator" or config.get("transmission_mode") != "analog":
        raise ValueError("This comparison requires an analog power allocator, not a rate allocator or QAM codec")
    latent_channels = int(checkpoint["allocator"]["backbone.0.weight"].shape[1]) - 1
    allocator = LegacyPowerAllocator(latent_channels, config).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    _freeze(allocator)
    channel = QuantizedAWGNChannel(
        allocator.group_rows, allocator.group_cols,
        config.get("qam_order", 256), config.get("quantization_amplitude", 3.0),
        snr_db, channel_gain_abs if channel_gain_abs is not None else config.get("channel_gain_abs", [1.0]),
        config.get("pilots_per_group", 4), "analog",
    ).to(device)
    return allocator, channel, config, path, int(checkpoint.get("epoch", -1)) + 1


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; run this command on an allocated GPU")
    allocator, channel, config, checkpoint_path, epoch = load_legacy_transmitter(
        args.checkpoint, device, args.snr_db, args.channel_gain_abs
    )
    image, label, image_path = load_input(args, config)
    image = image.to(device)
    vae_kwargs = {"torch_dtype": torch.float32, "local_files_only": args.local_files_only}
    if args.vae_path is None:
        vae_kwargs.update(subfolder="vae", revision=config.get("vae_model_revision", VAE_MODEL_REVISION))
    vae = _freeze(AutoencoderKL.from_pretrained(
        args.vae_path or config.get("vae_model_id", VAE_MODEL_ID), **vae_kwargs
    ).to(device))
    vae_scale = float(config.get("vae_scale", VAE_SCALE))
    lpips_model = load_lpips(device, enabled=getattr(args, "lpips", False), local_files_only=args.local_files_only)
    vae_generator = torch.Generator(device=device).manual_seed(args.seed)
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    diffusion_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    tag = Path(image_path).stem if args.input_image else f"sample_{args.sample_index:06d}"
    min_gain = getattr(args, "min_gain", 0.0)
    max_timestep = getattr(args, "max_timestep", None)
    gain_suffix = f"_mingain_{min_gain:g}" if min_gain > 0 else ""
    gain_suffix += f"_maxt_{max_timestep}" if max_timestep is not None else ""
    output_dir = Path(args.output_dir or checkpoint_path.parent / "inference_dps_no_q" /
                      f"{tag}_snr_{args.snr_db:g}_seed_{args.seed}_steps_{args.diffusion_steps}_guide_{args.guidance_scale:g}{gain_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z0 = encode_latents(image, vae, vae_generator)
        snr = torch.full((z0.shape[0],), args.snr_db, device=device)
        allocated_ratio, allocated_power, info = allocate_power_for_inference(
            z0, snr, allocator, channel, float(config.get("temperature", 1.0)),
            min_gain=min_gain,
            max_timestep=max_timestep,
        )
        energy, budget = info["unscaled_block_energy"], info["target_frame_energy"]
        uniform_power = (budget / energy.sum(1, keepdim=True)).expand_as(allocated_power)
        identity = torch.eye(allocated_power.shape[1], device=device, dtype=torch.float32)
        clean_image = _decode_latent(vae, z0, vae_scale)
        save_image(image, output_dir / "input.png")
        save_image(clean_image, output_dir / "vae_reconstruction.png")

    print(f"Q 없는 allocator: {checkpoint_path} (epoch {epoch})", flush=True)
    print(f"AWGN, SNR={args.snr_db:g} dB, 실수 잡음 분산=N0/2, "
          f"pilot={channel.pilots_per_group}/block, Q=I_{identity.shape[0]}", flush=True)
    reverse_model = StaggeredReverseDiffusion(
        args.diffusion_model or config.get("vae_model_id", VAE_MODEL_ID),
        device, torch.float32, args.local_files_only,
        revision=config.get("vae_model_revision", VAE_MODEL_REVISION) if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    sampler = OrthogonalPosteriorSampling(reverse_model, args.guidance_scale,
                                          args.guidance_max_rms, args.posterior_eta)
    transport = ActualPowerTransport(channel, "oracle")
    channel_state, diffusion_state = channel_generator.get_state(), diffusion_generator.get_state()
    strategies = ("uniform", "allocated") if args.strategy == "both" else (args.strategy,)
    panels, labels = [image[0], clean_image[0]], ["Input", "VAE reconstruction"]
    results = {}
    for name in strategies:
        with torch.no_grad():
            power = allocated_power if name == "allocated" else uniform_power
            channel_generator.set_state(channel_state)
            observation, channel_info, transport_info = transport(z0, power, snr, channel_generator)
            variance = channel_info["noise_variance"].float()
            if not torch.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError("The saved channel settings produced invalid observation variances")
            precision = variance.reciprocal()
            direct_image = _decode_latent(vae, observation, vae_scale)
            save_image(direct_image, output_dir / f"{name}_direct.png")
        print(f"{name}: Q=I로 DPS 복원을 시작합니다.", flush=True)
        diffusion_generator.set_state(diffusion_state)
        restored = sampler(observation, precision, identity, diffusion_generator,
                           rows=allocator.group_rows, cols=allocator.group_cols, verbose=not args.quiet)
        with torch.no_grad():
            restored_image = _decode_latent(vae, restored, vae_scale)
            save_image(restored_image, output_dir / f"{name}_dps.png")
            direct_metrics = image_metrics(direct_image, image, lpips_model)
            dps_metrics = image_metrics(restored_image, image, lpips_model)
            results[name] = {
                "uses_orthogonal_transform": False, "group_power": power[0].cpu().tolist(),
                "energy_allocation_ratio": (allocated_ratio if name == "allocated" else power * energy / budget)[0].cpu().tolist(),
                "noise_variance": variance[0].cpu().tolist(), "precision": precision[0].cpu().tolist(),
                "frame_energy": transport_info["transmit_frame_energy"][0].item(),
                "frame_channel_uses": transport_info["frame_channel_uses"],
                "direct_vs_input": direct_metrics, "dps_vs_input": dps_metrics,
                "direct_latent_mse": (observation - z0).square().mean().item(),
                "dps_latent_mse": (restored - z0).square().mean().item(),
                "expected_direct_latent_mse": variance.mean().item(),
                "final_observation_loss": observation_loss(
                    restored, observation, precision, identity, allocator.group_rows, allocator.group_cols
                )[0].item(),
                "diffusion": sampler.last_diagnostics,
            }
            if not args.no_save_latents:
                torch.save({"Q": identity.cpu(), "uses_orthogonal_transform": False,
                            "clean_latent": z0.cpu(), "equalized_observation": observation.cpu(),
                            "restored_latent": restored.cpu(), "group_power": power.cpu(),
                            "noise_variance": variance.cpu(), "precision": precision.cpu(),
                            "transmitted_frame": transport_info["transmitted_frame"].cpu(),
                            "received_frame": transport_info["received_frame"].cpu()},
                           output_dir / f"{name}_dps_state.pt")
            panels.append(restored_image[0])
            labels.append(f"{name}, no Q: oracle DPS")
            print(f"{name}: 직접 복원 PSNR={direct_metrics['psnr_db']:.2f} dB, "
                  f"DPS PSNR={dps_metrics['psnr_db']:.2f} dB", flush=True)
            if lpips_model is not None:
                print(f"{name}: 직접 복원 LPIPS={direct_metrics['lpips']:.5f}, "
                      f"DPS LPIPS={dps_metrics['lpips']:.5f}", flush=True)
    _save_labeled_row(panels, labels, output_dir / "comparison.png")
    result = {"checkpoint": str(checkpoint_path), "epoch": epoch, "input_image": image_path,
              "sample_index": args.sample_index, "label": label, "snr_db": args.snr_db, "seed": args.seed,
              "uses_orthogonal_transform": False, "receiver_mode": "oracle", "channel_model": "awgn",
              "min_gain": min_gain,
              "max_timestep": max_timestep,
              "gain_floor_infeasible_images": int(info.get(
                  "gain_floor_fallback", torch.zeros(1, dtype=torch.bool)
              ).sum().item()),
              "noise_real_component_variance": "N0/2", "pilots_per_group": channel.pilots_per_group,
              "allocator_snr_offset": allocator.snr_offset, "allocator_snr_scale": allocator.snr_scale,
              "same_noise_for_strategies": True, "same_diffusion_noise_for_strategies": True,
              "lpips_network": LPIPS_NETWORK if lpips_model is not None else None,
              "vae_vs_input": image_metrics(clean_image, image, lpips_model), "strategies": results}
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Q 없는 DPS 결과를 저장했습니다: {output_dir}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_image", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--snr_db", type=float, default=0)
    parser.add_argument("--min_gain", type=float, default=0.0,
                        help="Minimum allocated amplitude a_b during inference; 0 disables the floor")
    parser.add_argument("--max_timestep", type=int, choices=range(1000), metavar="0..999", default=None,
                        help="SD1.5-equivalent observation-noise upper bound; does not change DPS steps")
    parser.add_argument("--channel_gain_abs", type=float, nargs="+", default=None)
    parser.add_argument("--strategy", choices=("allocated", "uniform", "both"), default="both")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=0.1)
    parser.add_argument("--guidance_max_rms", type=float, default=1.0)
    parser.add_argument("--posterior_eta", type=float, default=0.0)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_save_latents", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.diffusion_steps <= 1000 or not math.isfinite(args.snr_db):
        parser.error("diffusion_steps must be in [1,1000] and snr_db must be finite")
    OrthogonalPosteriorSampling(None, args.guidance_scale, args.guidance_max_rms, args.posterior_eta)
    run(args)


if __name__ == "__main__":
    main()
