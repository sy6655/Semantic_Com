"""Compare true and estimated SD1.5 noise timesteps of 64 learned Q blocks.

Only the frozen VAE, Q/allocator and gain estimator are loaded. Across SNRs,
reuse one image, VAE posterior sample, complex h and standard Gaussian noise.
AWGN with h=1 is the default; Rayleigh fading is opt-in.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from inference_oracle_gain import load_input, save_image
from inference_power import allocate_power_for_inference
from orthogonal_power import (
    allocator_fingerprint, gain_estimator_receive, load_gain_estimator,
    prepare_gain_estimator_input, sample_raw_channel,
)
from train_gain_estimator import load_transmitter
from train_power_allocator import encode_latents
from power_utils import sd15_alpha_bars
from visualization_utils import noise_variance_to_diffusion_timestep


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain_estimator_checkpoint", required=True,
                        help="Estimator checkpoint or its directory (latest.pth)")
    parser.add_argument("--allocator_checkpoint", default=None,
                        help="Defaults to the transmitter path stored in the estimator checkpoint")
    parser.add_argument("--input_image", default=None)
    parser.add_argument("--sample_index", type=int, default=0, help="Image index in single-image mode")
    parser.add_argument("--num_samples", "--max_samples", type=int, default=1,
                        help="1: selected image; >1: class-balanced random ImageNet validation subset")
    parser.add_argument("--batch_size", "--batch", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, nargs="+", default=[-5., 0., 5., 10., 15., 20.])
    parser.add_argument("--channel_model", choices=("awgn", "rayleigh"), default="awgn",
                        help="AWGN with h=1 (default), or one sampled Rayleigh h shared across SNRs")
    parser.add_argument("--max_timestep", type=int, default=None,
                        help="Optional allocation floor; default matches estimator training without a floor")
    parser.add_argument("--min_gain", type=float, default=0.)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--display", choices=("timestep", "gain"), default="timestep",
                        help="Plot equivalent SD1.5 noise timesteps (default) or amplitude gains")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample_index must be non-negative")
    if min(args.num_samples, args.batch_size, args.log_interval) <= 0 or args.num_workers < 0:
        parser.error("num_samples, batch_size and log_interval must be positive; num_workers must be non-negative")
    if args.num_samples > 1 and args.input_image is not None:
        parser.error("Multiple samples require an ImageNet validation dataset; remove --input_image")
    if any(not math.isfinite(value) for value in args.snr_db) or len(set(args.snr_db)) != len(args.snr_db):
        parser.error("--snr_db values must be finite and unique")
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        parser.error("--min_gain must be finite and non-negative")
    if args.max_timestep is not None and not 0 <= args.max_timestep <= 999:
        parser.error("--max_timestep must be between 0 and 999")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or run on a GPU host")
    return args


def _checkpoint_path(path):
    path = Path(path).expanduser()
    return path / "latest.pth" if path.is_dir() else path


def compare_gains(true_gain, predicted_log_gain):
    """Report raw log-regression error and the gains actually used by reception.

    The receiver exponentiates clamp(predicted_log_gain,-20,20). Preserve the
    raw prediction and report clamping separately so extreme errors stay visible.
    """
    truth = true_gain.detach().double().cpu()
    predicted = predicted_log_gain.detach().double().cpu()
    if truth.ndim != 1 or truth.shape != predicted.shape or not truth.numel():
        raise ValueError("Gain comparison requires matching nonempty one-dimensional block arrays")
    if not torch.isfinite(truth).all() or (truth <= 0).any() or not torch.isfinite(predicted).all():
        raise ValueError("True gains must be finite and positive; predicted log gains must be finite")
    target = truth.log()
    used_log_gain = predicted.clamp(-20, 20)
    estimated = used_log_gain.exp()
    log_error = predicted - target
    relative_error = estimated / truth - 1
    absolute_relative = relative_error.abs()
    precision_relative_error = (estimated / truth).square() - 1
    metrics = {
        "log_gain_mse": log_error.square().mean().item(),
        "log_gain_rmse": log_error.square().mean().sqrt().item(),
        "gain_relative_mae": absolute_relative.mean().item(),
        "gain_relative_bias": relative_error.mean().item(),
        "gain_relative_p90": torch.quantile(absolute_relative, .9).item(),
        "gain_relative_max": absolute_relative.max().item(),
        "precision_relative_mae": precision_relative_error.abs().mean().item(),
        "training_gain_relative_mae": (log_error.clamp(-20, 20).exp() - 1).abs().mean().item(),
        "clipped_prediction_blocks": int((used_log_gain != predicted).sum()),
    }
    blocks = {
        "block_id": list(range(truth.numel())), "true_gain": truth.tolist(),
        "estimated_gain": estimated.tolist(), "target_log_gain": target.tolist(),
        "predicted_log_gain": predicted.tolist(), "used_log_gain": used_log_gain.tolist(),
        "signed_relative_error_percent": (100 * relative_error).tolist(),
        "precision_relative_error_percent": (100 * precision_relative_error).tolist(),
    }
    return metrics, blocks


def compare_timesteps(true_precision, estimated_precision):
    """Convert 1/lambda=N0/(abs(h)^2*a^2) to the first matching SD1.5 t.

    Conversion describes the noise level after gain correction. Values beyond
    the schedule are displayed at the endpoint and flagged separately.
    """
    truth = true_precision.detach().double().cpu()
    estimated = estimated_precision.detach().double().cpu()
    if truth.ndim != 1 or truth.shape != estimated.shape or not truth.numel():
        raise ValueError("Timestep comparison requires matching nonempty one-dimensional precision arrays")
    if (not torch.isfinite(truth).all() or not torch.isfinite(estimated).all()
            or (truth <= 0).any() or (estimated <= 0).any()):
        raise ValueError("True and estimated precision must be finite and positive")
    true_variance, estimated_variance = truth.reciprocal(), estimated.reciprocal()
    true_t = noise_variance_to_diffusion_timestep(true_variance)
    estimated_t = noise_variance_to_diffusion_timestep(estimated_variance)
    difference = estimated_t - true_t
    error = difference.double()
    alpha = sd15_alpha_bars().double()
    variance_schedule = (1 - alpha) / alpha
    blocks = {
        "true_noise_variance": true_variance.tolist(),
        "estimated_noise_variance": estimated_variance.tolist(),
        "true_timestep": true_t.tolist(), "estimated_timestep": estimated_t.tolist(),
        "signed_timestep_error": difference.tolist(),
    }
    metrics = {
        "timestep_mae": error.abs().mean().item(),
        "timestep_rmse": error.square().mean().sqrt().item(),
        "timestep_bias": error.mean().item(),
        "timestep_p90": torch.quantile(error.abs(), .9).item(),
        "timestep_max_error": int(difference.abs().max()),
    }
    for name, variance in (("true", true_variance), ("estimated", estimated_variance)):
        above, below = variance > variance_schedule[-1], variance < variance_schedule[0]
        blocks[f"{name}_timestep_above_schedule"] = above.tolist()
        blocks[f"{name}_timestep_below_schedule"] = below.tolist()
        metrics[f"{name}_timestep_above_schedule_blocks"] = int(above.sum())
        metrics[f"{name}_timestep_below_schedule_blocks"] = int(below.sum())
    return metrics, blocks


def add_timestep_comparison(case):
    metrics, blocks = compare_timesteps(torch.tensor(case["blocks"]["true_precision"], dtype=torch.float64),
                                        torch.tensor(case["blocks"]["estimated_precision"], dtype=torch.float64))
    case["metrics"].update(metrics)
    case["blocks"].update(blocks)


def save_plots(cases, output_dir, h_abs, display="timestep"):
    """Use common scales across SNR cases; map transformed block IDs, not RGB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    timestep_view = display == "timestep"
    aggregated = any(case.get("aggregation") == "mean_over_images" for case in cases.values())
    sample_count = next(iter(cases.values())).get("num_samples", 1)
    true_key, estimated_key = (("true_timestep", "estimated_timestep") if timestep_view
                               else ("true_gain", "estimated_gain"))
    error_key = (("mean_absolute_timestep_error" if timestep_view else "mean_absolute_relative_error_percent")
                 if aggregated else ("signed_timestep_error" if timestep_view else "signed_relative_error_percent"))
    if timestep_view:
        minimum, maximum = 0., 999.
        value_norm = Normalize(vmin=minimum, vmax=maximum)
    else:
        gains = [value for case in cases.values() for key in (true_key, estimated_key)
                 for value in case["blocks"][key]]
        minimum, maximum = min(gains), max(gains)
        if minimum == maximum:
            minimum, maximum = minimum * .95, maximum * 1.05
        value_norm = LogNorm(vmin=minimum, vmax=maximum)
    error_limit = max(1., max(abs(value) for case in cases.values()
                             for value in case["blocks"][error_key]))
    error_norm = Normalize(vmin=0 if aggregated else -error_limit, vmax=error_limit)
    for case in cases.values():
        folder = output_dir / case["directory"]
        folder.mkdir(parents=True, exist_ok=True)
        blocks = case["blocks"]
        title = (f"SNR setting={case['snr_db']:g} dB; |h|={h_abs:.4g}; "
                 f"frame SNR={case['effective_snr_db']:.2f} dB\n"
                 "Q-transformed block IDs (row-major); shared color scales across SNRs")
        if aggregated:
            title = (f"{sample_count} images; SNR setting={case['snr_db']:g} dB; mean |h|={h_abs:.4g}; "
                     f"mean frame SNR={case['effective_snr_db']:.2f} dB\n"
                     "Q-transformed block IDs; values averaged over images; shared color scales across SNRs")
        true_title = "True noise timestep" if timestep_view else "True amplitude gain a"
        estimated_title = "Estimated noise timestep" if timestep_view else "Estimated amplitude gain a_hat"
        error_title = "Timestep error: estimated - true" if timestep_view else "Relative error: 100 * (a_hat / a - 1)"
        if aggregated:
            true_title, estimated_title = "Mean " + true_title.lower(), "Mean " + estimated_title.lower()
            error_title = "Mean absolute timestep error" if timestep_view else "Mean absolute relative error (%)"
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        try:
            for axis, key, label, norm, cmap in (
                (axes[0], true_key, true_title, value_norm, "viridis"),
                (axes[1], estimated_key, estimated_title, value_norm, "viridis"),
                (axes[2], error_key, error_title, error_norm, "magma" if aggregated else "RdBu_r"),
            ):
                values = torch.tensor(blocks[key], dtype=torch.float64).reshape(8, 8).numpy()
                rendered = axis.imshow(values, norm=norm, cmap=cmap)
                axis.set_title(label)
                axis.set_xticks(range(8))
                axis.set_yticks(range(8))
                for row in range(8):
                    for col in range(8):
                        value = float(values[row, col])
                        rgb = rendered.cmap(norm(value))
                        luminance = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
                        if timestep_view:
                            text = (f"{value:.1f}" if aggregated else
                                    (f"{value:+.0f}" if key == error_key else f"{value:.0f}"))
                            if key in (true_key, estimated_key):
                                source = "true" if key == true_key else "estimated"
                                index = row * 8 + col
                                suffix = "_count" if aggregated else ""
                                if (blocks[f"{source}_timestep_above_schedule{suffix}"][index]
                                        or blocks[f"{source}_timestep_below_schedule{suffix}"][index]):
                                    text += "*"
                        else:
                            text = (f"{value:.1f}%" if aggregated else f"{value:+.1f}%") if key.endswith("percent") else f"{value:.3g}"
                        axis.text(col, row, f"#{row * 8 + col}\n{text}", ha="center", va="center",
                                  fontsize=7, color="white" if luminance < .5 else "black")
                fig.colorbar(rendered, ax=axis, shrink=.8)
            if timestep_view:
                title += ("\n* at least one image outside the SD1.5 schedule (0..999)" if aggregated else
                          "\n* variance outside the SD1.5 schedule, displayed at t=0 or 999")
            fig.suptitle(title, fontsize=11)
            fig.savefig(folder / f"{display}_heatmaps.png", dpi=160)
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
        try:
            axes[0].plot(blocks["block_id"], blocks[true_key], "o-", label=true_title, markersize=3)
            axes[0].plot(blocks["block_id"], blocks[estimated_key], "x--", label=estimated_title, markersize=4)
            if timestep_view:
                axes[0].set_ylim(0, 999)
                axes[0].set_ylabel("SD1.5 noise timestep")
            else:
                axes[0].set_yscale("log")
                axes[0].set_ylim(minimum * .9, maximum * 1.1)
                axes[0].set_ylabel("Amplitude gain (log axis)")
            axes[0].legend()
            axes[0].grid(alpha=.25)
            errors = blocks[error_key]
            axes[1].bar(blocks["block_id"], errors, color=["tab:red" if value >= 0 else "tab:blue" for value in errors])
            axes[1].axhline(0, color="black", linewidth=.8)
            axes[1].set_ylim(0 if aggregated else -error_limit * 1.1, error_limit * 1.1)
            axes[1].set_ylabel(error_title if aggregated else ("Timestep error (steps)" if timestep_view else "Signed relative error (%)"))
            axes[1].set_xlabel("Q-transformed block ID")
            axes[1].set_xlim(-1, 64)
            axes[1].set_xticks(range(0, 64, 4))
            axes[1].grid(axis="y", alpha=.25)
            metrics = case["metrics"]
            count_label = f"{sample_count} images; " if aggregated else ""
            if timestep_view:
                fig.suptitle(f"{count_label}SNR={case['snr_db']:g} dB; timestep MAE={metrics['timestep_mae']:.2f}; "
                             f"RMSE={metrics['timestep_rmse']:.2f} steps")
            else:
                fig.suptitle(f"{count_label}SNR={case['snr_db']:g} dB; relative MAE={100*metrics['gain_relative_mae']:.2f}%; "
                             f"log-gain RMSE={metrics['log_gain_rmse']:.4f}")
            fig.savefig(folder / f"{display}_by_block.png", dpi=160)
        finally:
            plt.close(fig)
    ordered = sorted(cases.values(), key=lambda case: case["snr_db"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    try:
        summaries = (("timestep_mae", 1., "Timestep MAE (steps)"),
                     ("timestep_rmse", 1., "Timestep RMSE (steps)"),
                     ("timestep_max_error", 1., "Maximum timestep error (steps)")) if timestep_view else (
                     ("gain_relative_mae", 100., "Gain relative MAE (%)"),
                     ("log_gain_rmse", 1., "Log-gain RMSE"),
                     ("precision_relative_mae", 100., "Precision relative MAE (%)"))
        for axis, (key, factor, title) in zip(axes, summaries):
            axis.plot([case["snr_db"] for case in ordered],
                      [factor * case["metrics"][key] for case in ordered], "o-")
            axis.set_xlabel("SNR setting (dB)")
            axis.set_title(title)
            axis.grid(alpha=.25)
        fig.suptitle(f"{sample_count} images: same images / VAE draws / h / standard noise across SNRs; mean |h|={h_abs:.4g}"
                     if aggregated else f"One image / VAE sample / h / standard noise shared across SNRs; |h|={h_abs:.4g}")
        fig.savefig(output_dir / "snr_comparison.png", dpi=160)
    finally:
        plt.close(fig)


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    channel_model = getattr(args, "channel_model", "awgn")
    if channel_model not in ("awgn", "rayleigh"):
        raise ValueError("channel_model must be awgn or rayleigh")
    estimator_path = _checkpoint_path(args.gain_estimator_checkpoint)
    estimator_checkpoint = torch.load(estimator_path, map_location="cpu", weights_only=False)
    transmitter = args.allocator_checkpoint or estimator_checkpoint["config"].get("allocator_checkpoint")
    if transmitter is None:
        raise ValueError("Provide --allocator_checkpoint; no transmitter path is stored in the estimator checkpoint")
    allocator, channel, config, transmitter_path = load_transmitter(transmitter, device)
    # Diagnostic comparisons use the selected allocator's actual transmitted
    # gain as ground truth, including when its weights changed after training.
    estimator, estimator_config = load_gain_estimator(
        estimator_path, allocator, device, check_allocator_fingerprint=False,
    )
    current_fingerprint = allocator_fingerprint(allocator)
    trained_fingerprint = estimator_config.get("allocator_fingerprint")
    kwargs = {"local_files_only": args.local_files_only, "torch_dtype": torch.float32}
    if args.vae_path is None:
        kwargs.update(subfolder="vae", revision=config["vae_model_revision"])
    vae = AutoencoderKL.from_pretrained(args.vae_path or config["vae_model_id"], **kwargs).to(device)
    vae.eval().requires_grad_(False)
    if int(vae.config.latent_channels) != int(config.get("latent_channels", 4)):
        raise ValueError("VAE channels do not match the transmitter")
    if getattr(args, "num_samples", 1) > 1:
        metadata = {
            "gain_estimator_checkpoint": str(estimator_path), "allocator_checkpoint": str(transmitter_path),
            "allocator_fingerprint": current_fingerprint, "training_allocator_fingerprint": trained_fingerprint,
            "allocator_matches_training": current_fingerprint == trained_fingerprint,
            "estimator_epoch": int(estimator_checkpoint["epoch"]) + 1, "estimator_config": estimator_config,
        }
        return evaluate_samples(args, allocator, channel, estimator, vae, config, metadata)
    image, label, image_path = load_input(args, config)
    image = image.to(device)
    latent = encode_latents(image, vae, torch.Generator(device=device).manual_seed(args.seed))
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    h = channel.sample_channel_coefficient(latent, channel_generator)
    if channel_model == "awgn":
        h = torch.ones_like(h)
    epsilon = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=channel_generator)
    snr_tag = "-".join(f"{value:g}" for value in args.snr_db)
    floor_tag = f"maxt_{args.max_timestep}_mingain_{args.min_gain:g}"
    sample_tag = Path(image_path).stem if args.input_image else f"sample_{args.sample_index:06d}"
    output_dir = Path(args.output_dir or estimator_path.parent / "visualizations" /
                      f"{sample_tag}_snr_{snr_tag}_seed_{args.seed}_{channel_model}_{floor_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(image, output_dir / "input.png")
    result = {
        "gain_estimator_checkpoint": str(estimator_path), "allocator_checkpoint": str(transmitter_path),
        "allocator_fingerprint": current_fingerprint,
        "training_allocator_fingerprint": trained_fingerprint,
        "allocator_matches_training": current_fingerprint == trained_fingerprint,
        "estimator_epoch": int(estimator_checkpoint["epoch"]) + 1,
        "dataset": "ImageNet", "sample_index": args.sample_index, "image_path": image_path,
        "class_index": label, "seed": args.seed, "snr_db": args.snr_db,
        "max_timestep": args.max_timestep, "min_gain": args.min_gain,
        "allocation_policy_matches_training": args.max_timestep is None and args.min_gain == 0,
        "allocation_matches_training": (current_fingerprint == trained_fingerprint
                                        and args.max_timestep is None and args.min_gain == 0),
        "channel_model": "awgn" if channel_model == "awgn" else config["channel_model"],
        "transmitter_training_channel_model": config["channel_model"], "perfect_csi": True,
        "receiver_mode": "gain_estimator", "noise_real_component_variance": "N0",
        "mean_real_component_power": 1., "uses_reverse_diffusion": False,
        "same_image_latent_h_and_standard_noise_across_snrs": True,
        "h": {"real": h.real.item(), "imag": h.imag.item(), "abs": h.abs().item()},
        "num_groups": 64, "group_grid": [8, 8], "block_ids": "Q^T transformed, row-major; not spatial RGB patches",
        "gain_definition": "amplitude a_b, not a_b^2 or abs(h)*a_b",
        "receiver_gain_clamp_log_range": [-20, 20], "estimator_config": estimator_config,
        "display": args.display,
        "timestep_conversion": {
            "noise_variance": "1/precision = N0/(abs(h)^2*a_b^2)",
            "schedule": "SD1.5 scaled_linear beta, 1000 steps",
            "mapping": "first t with (1-alpha_bar[t])/alpha_bar[t] >= noise_variance; clamp to [0,999]",
            "error": "estimated_timestep - true_timestep",
        },
        "output_dir": str(output_dir), "cases": {},
    }
    state = {"latent": latent.cpu(), "h": h.cpu(), "standard_noise": epsilon.cpu(), "cases": {}}
    for snr_db in args.snr_db:
        snr = torch.full((latent.shape[0],), snr_db, device=device)
        ratio, power, info = allocate_power_for_inference(
            latent, snr, allocator, channel, config.get("temperature", 1.),
            channel_coefficient=h, min_gain=args.min_gain, max_timestep=args.max_timestep,
        )
        n0 = channel._noise_spectral_density(latent, snr)
        transformed = info["channel_input"]
        raw = sample_raw_channel(transformed, power, n0, h, epsilon=epsilon)
        inputs = prepare_gain_estimator_input(raw, n0, h)
        predicted_log_gain = estimator(inputs["channel_equalized_latent"],
                                       inputs["log_noise_variance"], inputs["log_power"])
        received = gain_estimator_receive(raw, predicted_log_gain, n0, h)
        metrics, blocks = compare_gains(power[0].sqrt(), predicted_log_gain[0])
        true_precision = h.abs().square()[:, None] * power / n0[:, None]
        blocks.update(true_precision=true_precision[0].cpu().tolist(),
                      estimated_precision=received["precision"][0].cpu().tolist(),
                      energy_allocation_ratio=ratio[0].cpu().tolist())
        timestep_metrics, timestep_blocks = compare_timesteps(true_precision[0], received["precision"][0])
        metrics.update(timestep_metrics)
        blocks.update(timestep_blocks)
        key = f"{snr_db:g}"
        folder = output_dir / f"snr_{key}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "blocks.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(blocks.keys())
            writer.writerows(zip(*blocks.values()))
        result["cases"][key] = {
            "snr_db": snr_db, "effective_snr_db": float(info["allocator_snr_db"][0]),
            "N0": n0.item(), "directory": folder.name, "metrics": metrics, "blocks": blocks,
            "frame_energy": float((power * info["unscaled_block_energy"]).sum()),
            "gain_floor_infeasible_images": int(info.get("gain_floor_fallback", torch.zeros(1)).sum()),
        }
        state["Q"] = info["orthogonal_matrix"].cpu()
        state["cases"][key] = {"group_power": power.cpu(), "raw_received_latent": raw.cpu(),
                               "predicted_log_gain": predicted_log_gain.cpu(),
                               "estimated_gain": received["amplitude_gain"].cpu(),
                               "true_precision": true_precision.cpu(), "estimated_precision": received["precision"].cpu(),
                               "N0": n0.cpu()}
        print(f"channel={channel_model} SNR={snr_db:g} dB effective={float(info['allocator_snr_db'][0]):.2f} dB "
              f"|h|={h.abs().item():.4g} relative_MAE={100*metrics['gain_relative_mae']:.2f}% "
              f"log_gain_RMSE={metrics['log_gain_rmse']:.4f} "
              f"timestep_MAE={metrics['timestep_mae']:.2f} "
              f"clipped_predictions={metrics['clipped_prediction_blocks']}", flush=True)
    save_plots(result["cases"], output_dir, h.abs().item(), display=args.display)
    torch.save(state, output_dir / "observations.pt")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved gain estimator comparison to {output_dir}", flush=True)
    return result


class GainEvaluationAccumulator:
    """Pool image/block errors; average maps only after measuring each image."""

    def __init__(self):
        self.parts = {name: [] for name in ("true_gain", "predicted_log_gain", "true_precision",
                                           "estimated_precision", "effective_snr_db", "frame_energy",
                                           "mean_real_component_power")}
        self.num_samples = 0
        self.timestep_absolute_error_sum = 0.

    def add(self, true_gain, predicted_log_gain, true_precision, estimated_precision,
            effective_snr_db, frame_energy, real_component_count):
        values = {
            "true_gain": true_gain, "predicted_log_gain": predicted_log_gain,
            "true_precision": true_precision, "estimated_precision": estimated_precision,
            "effective_snr_db": effective_snr_db, "frame_energy": frame_energy,
            "mean_real_component_power": frame_energy / real_component_count,
        }
        if true_gain.ndim != 2 or true_gain.shape[1] != 64:
            raise ValueError("Evaluation requires [B,64] block gains")
        for key, value in values.items():
            self.parts[key].append(value.detach().float().cpu())
        metrics, _ = compare_timesteps(true_precision.flatten(), estimated_precision.flatten())
        self.num_samples += true_gain.shape[0]
        self.timestep_absolute_error_sum += metrics["timestep_mae"] * true_gain.numel()

    @property
    def timestep_mae(self):
        return self.timestep_absolute_error_sum / (64 * self.num_samples)

    def finalize(self):
        if not self.num_samples:
            raise ValueError("No evaluated samples")
        data = {key: torch.cat(parts) for key, parts in self.parts.items()}
        gain_metrics, gain_blocks = compare_gains(data["true_gain"].flatten(), data["predicted_log_gain"].flatten())
        timestep_metrics, timestep_blocks = compare_timesteps(data["true_precision"].flatten(),
                                                             data["estimated_precision"].flatten())
        metrics = {**gain_metrics, **timestep_metrics, "num_images": self.num_samples,
                   "num_block_entries": self.num_samples * 64}
        blocks = {"block_id": list(range(64))}
        for key, values in {**gain_blocks, **timestep_blocks}.items():
            if key == "block_id":
                continue
            array = torch.tensor(values, dtype=torch.float64).reshape(self.num_samples, 64)
            if key.endswith("_schedule"):
                blocks[key + "_count"] = array.sum(0).long().tolist()
            else:
                blocks[key] = array.mean(0).tolist()
        gain_errors = torch.tensor(gain_blocks["signed_relative_error_percent"], dtype=torch.float64).reshape(self.num_samples, 64)
        step_errors = torch.tensor(timestep_blocks["signed_timestep_error"], dtype=torch.float64).reshape(self.num_samples, 64)
        log_errors = (data["predicted_log_gain"].double() - data["true_gain"].double().log())
        blocks.update(
            mean_absolute_relative_error_percent=gain_errors.abs().mean(0).tolist(),
            mean_absolute_timestep_error=step_errors.abs().mean(0).tolist(),
            timestep_rmse=step_errors.square().mean(0).sqrt().tolist(),
            true_precision=data["true_precision"].double().mean(0).tolist(),
            estimated_precision=data["estimated_precision"].double().mean(0).tolist(),
        )
        per_image = {
            "gain_relative_mae": gain_errors.abs().mean(1).div(100).tolist(),
            "log_gain_mse": log_errors.square().mean(1).tolist(),
            "timestep_mae": step_errors.abs().mean(1).tolist(),
            "timestep_rmse": step_errors.square().mean(1).sqrt().tolist(),
            "timestep_bias": step_errors.mean(1).tolist(),
            "timestep_max_error": step_errors.abs().max(1).values.long().tolist(),
            "effective_snr_db": data["effective_snr_db"].tolist(),
            "frame_energy": data["frame_energy"].tolist(),
            "mean_real_component_power": data["mean_real_component_power"].tolist(),
        }
        return metrics, blocks, per_image, data


@torch.no_grad()
def evaluate_samples(args, allocator, channel, estimator, vae, config, metadata):
    """Encode each selected batch once and evaluate all SNRs on the same draws."""
    from evaluate_power_maxvit import IndexedDataset, select_balanced_indices

    started = time.perf_counter()
    device = torch.device(args.device)
    data_path = args.data_path or config.get("data_path")
    if data_path is None:
        raise ValueError("Multiple samples require --data_path or a data_path in the allocator checkpoint")
    root = Path(data_path).expanduser()
    validation_root = root / "val" if (root / "val").is_dir() else root
    image_size = int(config.get("image_size", 512))
    dataset = datasets.ImageFolder(str(validation_root), transform=transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
    ]))
    indices = select_balanced_indices(dataset, args.num_samples, args.seed)
    loader = DataLoader(Subset(IndexedDataset(dataset), indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(args.seed + 3))
    channel_model = args.channel_model
    snr_tag = "-".join(f"{value:g}" for value in args.snr_db)
    output_dir = Path(args.output_dir or Path(metadata["gain_estimator_checkpoint"]).parent / "evaluation" /
                      f"gain_check_n{len(indices)}_snr_{snr_tag}_{channel_model}_seed_{args.seed}"
                      f"_maxt_{args.max_timestep}_mingain_{args.min_gain:g}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [{"index": index, "class_index": int(dataset.targets[index]),
                "path": str(Path(dataset.samples[index][0]).relative_to(validation_root))} for index in indices]
    selection = {"method": "class_balanced_random_without_replacement", "seed": args.seed,
                 "indices": indices, "samples": samples,
                 "class_counts": {dataset.classes[label]: sum(sample["class_index"] == label for sample in samples)
                                  for label in sorted({sample["class_index"] for sample in samples})}}
    result = {
        **metadata, "dataset": "ImageNet", "dataset_split": "val", "dataset_size": len(dataset),
        "validation_root": str(validation_root), "image_size": image_size,
        "num_samples": len(indices), "requested_num_samples": args.num_samples,
        "batch_size": args.batch_size, "seed": args.seed, "sample_selection": selection,
        "snr_db": args.snr_db, "channel_model": "awgn" if channel_model == "awgn" else config["channel_model"],
        "transmitter_training_channel_model": config["channel_model"], "perfect_csi": True,
        "h": {"real": 1., "imag": 0., "abs": 1.} if channel_model == "awgn" else None,
        "noise_real_component_variance": "N0", "mean_real_component_power": 1.,
        "max_timestep": args.max_timestep, "min_gain": args.min_gain,
        "receiver_mode": "gain_estimator", "uses_reverse_diffusion": False,
        "same_image_latent_h_and_standard_noise_across_snrs": True,
        "num_groups": 64, "group_grid": [8, 8], "display": args.display,
        "gain_definition": "amplitude a_b, not a_b^2 or abs(h)*a_b",
        "receiver_gain_clamp_log_range": [-20, 20],
        "timestep_conversion": "First SD1.5 t with (1-alpha_bar[t])/alpha_bar[t] >= 1/precision; clamp to [0,999]",
        "aggregation": "Metrics pooled over all image/block pairs; maps averaged over images; error maps show mean absolute error",
        "output_dir": str(output_dir), "cases": {},
    }
    accumulators = {f"{snr:g}": GainEvaluationAccumulator() for snr in args.snr_db}
    fallbacks = dict.fromkeys(accumulators, 0)
    vae_generator = torch.Generator(device=device).manual_seed(args.seed)
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    sampled_h, processed = [], 0
    print(f"Evaluating {len(indices)} ImageNet validation images; channel={channel_model}; batch={args.batch_size}; "
          "VAE/allocator/estimator only; aggregate plots", flush=True)
    for batch_index, (images, _, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        latent = encode_latents(images, vae, vae_generator)
        h = channel.sample_channel_coefficient(latent, channel_generator)
        if channel_model == "awgn":
            h = torch.ones_like(h)
        sampled_h.append(h.cpu())
        epsilon = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=channel_generator)
        for snr_db in args.snr_db:
            key = f"{snr_db:g}"
            snr = torch.full((latent.shape[0],), snr_db, device=device)
            _, power, info = allocate_power_for_inference(
                latent, snr, allocator, channel, config.get("temperature", 1.),
                channel_coefficient=h, min_gain=args.min_gain, max_timestep=args.max_timestep,
            )
            n0 = channel._noise_spectral_density(latent, snr)
            raw = sample_raw_channel(info["channel_input"], power, n0, h, epsilon=epsilon)
            inputs = prepare_gain_estimator_input(raw, n0, h)
            predicted_log_gain = estimator(inputs["channel_equalized_latent"],
                                           inputs["log_noise_variance"], inputs["log_power"])
            received = gain_estimator_receive(raw, predicted_log_gain, n0, h)
            true_precision = h.abs().square()[:, None] * power / n0[:, None]
            frame_energy = (power * info["unscaled_block_energy"]).sum(1)
            accumulators[key].add(power.sqrt(), predicted_log_gain, true_precision, received["precision"],
                                  info["allocator_snr_db"], frame_energy, latent[0].numel())
            fallbacks[key] += int(info.get("gain_floor_fallback", torch.zeros(1)).sum())
        processed += images.shape[0]
        if batch_index == 0 or (batch_index + 1) % args.log_interval == 0 or processed == len(indices):
            progress = " ".join(f"SNR={key}:t_MAE={accumulator.timestep_mae:.2f}" for key, accumulator in accumulators.items())
            print(f"samples={processed}/{len(indices)} elapsed={time.perf_counter()-started:.1f}s {progress}", flush=True)
    if processed != len(indices):
        raise RuntimeError("Evaluation did not process all selected images")
    all_h = torch.cat(sampled_h)
    mean_h = all_h.abs().mean().item()
    result["channel_statistics"] = {"mean_abs_h": mean_h, "min_abs_h": all_h.abs().min().item(),
                                    "max_abs_h": all_h.abs().max().item()}
    state = {"indices": torch.tensor(indices), "h": all_h, "cases": {}}
    per_image_rows = []
    for key, accumulator in accumulators.items():
        metrics, blocks, image_metrics, values = accumulator.finalize()
        folder = output_dir / f"snr_{key}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "blocks.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(blocks)
            writer.writerows(zip(*blocks.values()))
        result["cases"][key] = {
            "snr_db": float(key), "effective_snr_db": values["effective_snr_db"].mean().item(),
            "directory": folder.name, "num_samples": processed, "aggregation": "mean_over_images",
            "metrics": metrics, "blocks": blocks, "gain_floor_infeasible_images": fallbacks[key],
            "mean_frame_energy": values["frame_energy"].mean().item(),
            "mean_real_component_power": values["mean_real_component_power"].mean().item(),
        }
        state["cases"][key] = values
        for position, sample in enumerate(samples):
            per_image_rows.append({**sample, "snr_db": float(key), "h_abs": all_h[position].abs().item(),
                                   **{name: items[position] for name, items in image_metrics.items()}})
        print(f"SNR={key} images={processed} timestep_MAE={metrics['timestep_mae']:.2f} "
              f"RMSE={metrics['timestep_rmse']:.2f} relative_MAE={100*metrics['gain_relative_mae']:.2f}%", flush=True)
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    save_plots(result["cases"], output_dir, mean_h, display=args.display)
    torch.save(state, output_dir / "sample_values.pt")
    result["evaluation_seconds"] = time.perf_counter() - started
    result["complete"] = True
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved aggregate gain estimator evaluation: {output_dir} "
          f"(evaluation {result['evaluation_seconds']:.1f}s)", flush=True)
    return result


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()
