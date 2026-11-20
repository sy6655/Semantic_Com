"""Unified ImageNet Q inference with oracle or estimated gain correction.

Defaults: AWGN, DDRM, uniform Q=I, allocated Q from the supplied checkpoint.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from inference import StaggeredReverseDiffusion
from inference_power import (
    _decode_latent, _freeze, _save_labeled_row, _task_result,
    allocate_power_for_inference,
)
from orthogonal_power import (
    OrthogonalPatchTransform, allocator_fingerprint, apply_complex_channel, block_map,
    gain_estimator_receive, load_gain_estimator, observation_loss, patchify,
    prepare_gain_estimator_input, sample_raw_channel,
)
from posterior_sampling import add_restoration_arguments, build_restoration_sampler, resolve_max_timestep
from reconstruction_metrics import LPIPS_NETWORK, image_quality, load_lpips
from train_gain_estimator import load_transmitter
from train_power_allocator import (
    VAE_SCALE, build_latent_classifier, classifier_forward,
    encode_latents, restore_spatial_latent,
)
from visualization_utils import noise_variance_to_diffusion_timestep


def oracle_receive(raw_received, group_power, n0, h, rows=8, cols=8):
    """Use the true a_b directly: U_tilde=Y/(h*a_b), lambda=|h|^2*a_b^2/N0."""
    if group_power.shape != (raw_received.shape[0], rows * cols):
        raise ValueError("group_power must have shape [B,num_blocks]")
    if not torch.isfinite(group_power).all() or (group_power <= 0).any():
        raise ValueError("Oracle transmit powers must be finite and positive")
    if n0.shape != (raw_received.shape[0],) or (n0 <= 0).any():
        raise ValueError("N0 must have shape [B] and be positive")
    gain = group_power.float().sqrt()
    channel_equalized = apply_complex_channel(raw_received, h.reciprocal(), rows, cols)
    observation = channel_equalized / block_map(gain, raw_received, rows, cols)
    precision = h.abs().float().square()[:, None] * group_power.float() / n0[:, None]
    return {"observation": observation, "amplitude_gain": gain,
            "precision": precision, "noise_variance": precision.reciprocal()}


def load_input(args, config):
    image_size = int(config.get("image_size", 512))
    transform = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
    ])
    if args.input_image is not None:
        image_path = Path(args.input_image).expanduser()
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB"))
        return tensor.unsqueeze(0), None, str(image_path)
    data_path = args.data_path or config.get("data_path")
    if data_path is None:
        raise ValueError("Provide --input_image or --data_path")
    root = Path(data_path).expanduser()
    validation_root = root / "val" if (root / "val").is_dir() else root
    dataset = datasets.ImageFolder(str(validation_root), transform=transform)
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(f"sample_index must be between 0 and {len(dataset) - 1}")
    tensor, label = dataset[args.sample_index]
    return tensor.unsqueeze(0), int(label), dataset.samples[args.sample_index][0]


def image_metrics(reconstructed, reference, lpips_model=None):
    return image_quality(reconstructed, reference, lpips_model)


def save_image(tensor, path):
    to_pil_image(tensor[0].detach().float().cpu().clamp(0, 1)).save(path)


def save_power_allocation(strategies, path, rows=8, cols=8, *, display="power"):
    """Render power/energy or real/estimated diffusion timestep maps.

    Q-transformed blocks are shown by block ID, not overlaid on RGB patches.
    The power display remains available for segmentation callers.
    """
    if display not in ("power", "timestep"):
        raise ValueError("display must be power or timestep")
    timestep_display = display == "timestep"
    maps = [
        (name, torch.as_tensor(data["true_timestep"] if timestep_display else data["group_power"])
         .float().cpu().reshape(rows, cols),
         torch.as_tensor(data["estimated_timestep"]).float().cpu().reshape(rows, cols) if timestep_display else
         100 * torch.as_tensor(data["energy_allocation_ratio"]).float().cpu().reshape(rows, cols))
        for name, data in strategies.items()
    ]
    maxima = [999., 999.] if timestep_display else [max(float(entry[index].max()) for entry in maps) for index in (1, 2)]
    cell, margin, gap = 72, 24, 40
    grid_width, grid_height = cols * cell, rows * cell
    row_height = grid_height + 120
    width = 2 * grid_width + 2 * margin + gap
    canvas = Image.new("RGB", (width, 80 + len(maps) * row_height + 38), "white")
    draw = ImageDraw.Draw(canvas)

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    title_font, text_font, id_font = font(21), font(16), font(12)

    def color(fraction):
        low, high = (239, 248, 255), (8, 62, 119)
        return tuple(round(a + fraction * (b - a)) for a, b in zip(low, high))

    draw.text((margin, 12), "Diffusion timestep maps" if timestep_display else "Transmit power allocation",
              font=title_font, fill="black")
    draw.text((margin, 43), ("Common color scale: 0-999; cells show block ID and timestep." if timestep_display else
                           "Common color scale per column; cells show block ID and value."),
              font=text_font, fill="black")
    for index, (name, powers, shares) in enumerate(maps):
        top = 80 + index * row_height
        title = ("Allocated + Q: transformed block IDs (row-major)" if name == "allocated"
                 else "Uniform, no Q: spatial blocks (row-major)")
        draw.text((margin, top), title, font=title_font, fill="black")
        if timestep_display:
            estimated_label = ("Estimated diffusion timestep" if strategies[name]["receiver_mode"] == "gain_estimator"
                               else "Receiver timestep (oracle)")
            panels = ((powers, "Real diffusion timestep"), (shares, estimated_label))
        else:
            panels = ((powers, "Gain squared: a_b^2"), (shares, "Share of total transmit energy (%)"))
        for column, (values, label) in enumerate(panels):
            left = margin + column * (grid_width + gap)
            draw.text((left, top + 34), label, font=text_font, fill="black")
            grid_top = top + 60
            maximum = max(maxima[column], 1e-12)
            for row in range(rows):
                for col in range(cols):
                    value = float(values[row, col])
                    rgb = color(min(max(value / maximum, 0), 1))
                    foreground = "white" if sum(rgb) < 400 else "black"
                    x, y = left + col * cell, grid_top + row * cell
                    draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=rgb, outline="white")
                    draw.text((x + 5, y + 5), f"#{row * cols + col}", font=id_font, fill=foreground)
                    value_text = str(int(value)) if timestep_display else (f"{value:.2f}%" if column else f"{value:.3g}")
                    draw.text((x + 5, y + 30), value_text, font=text_font, fill=foreground)
            bar_top = grid_top + grid_height + 10
            for pixel in range(grid_width):
                draw.line((left + pixel, bar_top, left + pixel, bar_top + 10),
                          fill=color(pixel / (grid_width - 1)))
            draw.text((left, bar_top + 15), "0", font=id_font, fill="black")
            maximum_text = "999" if timestep_display else (f"{maxima[column]:.2f}%" if column else f"{maxima[column]:.3g}")
            draw.text((left + grid_width - 70, bar_top + 15), maximum_text, font=id_font, fill="black")
    draw.text((margin, canvas.height - 28),
              ("SD1.5 timesteps; noise beyond the schedule is displayed as 999." if timestep_display else
               "Uniform gain is constant; energy shares also depend on the original block energy."),
              font=text_font, fill="black")
    canvas.save(path)


def run_inference(args, checkpoint_path=None, checkpoint=None):
    method = getattr(args, "restoration_method", None) or "ddrm"
    args.restoration_method = method
    method_label = method.upper()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; run on a GPU host or specify --device cpu")
    checkpoint_path = Path(checkpoint_path or args.checkpoint).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path /= "latest.pth"
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    allocator, channel, config, _ = load_transmitter(checkpoint_path, device, checkpoint=checkpoint)
    estimator_path = getattr(args, "gain_estimator_checkpoint", None)
    receiver_mode = getattr(args, "receiver_mode", None) or ("gain_estimator" if estimator_path else "oracle")
    uniform_receiver_mode = getattr(args, "uniform_receiver_mode", None) or "oracle"
    receiver_modes = {"allocated": receiver_mode, "uniform": uniform_receiver_mode}
    strategy_names = ("uniform",) if getattr(args, "strategy", "both") == "uniform" else ("allocated", "uniform")
    active_receiver_modes = {receiver_modes[name] for name in strategy_names}
    channel_model = getattr(args, "channel_model", "awgn")
    if channel_model not in ("rayleigh", "awgn"):
        raise ValueError("channel_model must be rayleigh or awgn")
    if any(mode not in ("gain_estimator", "oracle") for mode in receiver_modes.values()):
        raise ValueError("Orthogonal transmission supports gain_estimator or oracle receiver mode")
    if "gain_estimator" in active_receiver_modes and estimator_path is None:
        raise ValueError("--gain_estimator_checkpoint is required when allocated or uniform uses gain_estimator")
    if not 1 <= args.diffusion_steps <= 1000:
        raise ValueError("diffusion_steps must be between 1 and 1000")
    snrs = config.get("snr_db", [10.0])
    snr_db = float(args.snr_db if args.snr_db is not None else snrs[0])
    if getattr(args, "channel_gain_abs", None) is not None:
        raise ValueError("Use --channel_model awgn for h=1 or --channel_model rayleigh for sampled h")
    lpips_model = load_lpips(device, enabled=getattr(args, "lpips", True), local_files_only=args.local_files_only)
    vae_kwargs = {"local_files_only": args.local_files_only, "torch_dtype": torch.float32}
    if args.vae_path is None:
        vae_kwargs.update(subfolder="vae", revision=config["vae_model_revision"])
    vae = _freeze(AutoencoderKL.from_pretrained(args.vae_path or config["vae_model_id"], **vae_kwargs).to(device))
    condition_on_snr = bool(config.get("condition_classifier_on_snr", False))
    classifier = None
    if "classifier" in checkpoint:
        channels = int(vae.config.latent_channels)
        classifier = build_latent_classifier(config["model"], channels * (2 if condition_on_snr else 1),
                                             config.get("num_classes", 1000)).to(device)
        classifier.load_state_dict(checkpoint["classifier"])
        _freeze(classifier)
    estimator, estimator_config = (None, None)
    current_fingerprint = allocator_fingerprint(allocator)
    trained_fingerprint = None
    if "gain_estimator" in active_receiver_modes:
        estimator, estimator_config = load_gain_estimator(
            Path(estimator_path) / "latest.pth" if Path(estimator_path).is_dir() else estimator_path,
            allocator, device, check_allocator_fingerprint=False,
        )
        trained_fingerprint = estimator_config.get("allocator_fingerprint")
    reverse_model = StaggeredReverseDiffusion(
        args.diffusion_model or config["vae_model_id"], device, torch.float32,
        args.local_files_only, revision=config["vae_model_revision"] if args.diffusion_model is None else None,
        num_inference_steps=args.diffusion_steps,
    )
    samplers = {mode: build_restoration_sampler(reverse_model, args, receiver_mode=mode)
                for mode in active_receiver_modes}
    min_gain = getattr(args, "min_gain", 0.0)
    max_timestep = resolve_max_timestep(args)
    gain_suffix = f"_mingain_{min_gain:g}" if min_gain > 0 else ""
    gain_suffix += f"_maxt_{max_timestep}" if max_timestep is not None else ""
    sampling_suffix = ("" if method == "dps" else
                       f"_eta_{args.ddrm_eta:g}_etab_{args.ddrm_eta_b:g}")
    channel_suffix = "_awgn" if channel_model == "awgn" else ""
    uniform_suffix = f"_uniform_{uniform_receiver_mode}" if uniform_receiver_mode != receiver_mode else ""
    output_dir = Path(args.output_dir or checkpoint_path.parent / "inference" /
                      f"sample_{args.sample_index:06d}_snr_{snr_db:g}_seed_{args.seed}_steps_{args.diffusion_steps}"
                      f"_{receiver_mode}{uniform_suffix}_{method}{sampling_suffix}{channel_suffix}{gain_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image, label, image_path = load_input(args, config)
    image = image.to(device)
    vae_generator = torch.Generator(device=device).manual_seed(args.seed)
    channel_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    diffusion_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    vae_scale = float(config.get("vae_scale", VAE_SCALE))
    with torch.no_grad():
        latent = encode_latents(image, vae, vae_generator)
        vae_image = _decode_latent(vae, latent, vae_scale)
        vae_metrics = image_quality(vae_image, image, lpips_model)
        print(f"VAE 복원 PSNR={vae_metrics['psnr_db']:.2f} dB입니다.", flush=True)
        to_pil_image(image[0].float().cpu()).save(output_dir / "input.png")
        to_pil_image(vae_image[0].float().cpu()).save(output_dir / "vae_reconstruction.png")
        snr = torch.full((image.shape[0],), snr_db, device=device)
        h = channel.sample_channel_coefficient(latent, channel_generator)
        # Consume the same h draws in both channel modes before drawing AWGN.
        if channel_model == "awgn":
            h = torch.ones_like(h)
        allocated_ratio, allocated_power, info = allocate_power_for_inference(
            latent, snr, allocator, channel, config.get("temperature", 1.0),
            channel_coefficient=h,
            min_gain=min_gain,
            max_timestep=max_timestep,
        )
        transformed, q = info["channel_input"], info["orthogonal_matrix"]
        energy = info["unscaled_block_energy"]
        spatial_energy = patchify(latent).square().sum(-1)
        budget = info["target_frame_energy"]
        frame_channel_uses = info["frame_channel_uses"]
        real_component_count = latent[0].numel()
        uniform_power = (budget / spatial_energy.sum(1, keepdim=True)).expand_as(allocated_power)
        identity = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
        n0 = channel._noise_spectral_density(latent, snr)
        epsilon = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=channel_generator)
        diffusion_state = diffusion_generator.get_state()
        images, direct_images, results = {}, {}, {}
        for name in strategy_names:
            strategy_receiver_mode = receiver_modes[name]
            sampler = samplers[strategy_receiver_mode]
            uses_q = name == "allocated"
            power = allocated_power if uses_q else uniform_power
            channel_input = transformed if uses_q else latent
            observation_q = q if uses_q else identity
            block_energy = energy if uses_q else spatial_energy
            raw = sample_raw_channel(channel_input, power, n0, h, epsilon=epsilon)
            if strategy_receiver_mode == "gain_estimator":
                estimator_input = prepare_gain_estimator_input(raw, n0, h)
                log_gain = estimator(estimator_input["channel_equalized_latent"],
                                     estimator_input["log_noise_variance"], estimator_input["log_power"])
                received = gain_estimator_receive(raw, log_gain, n0, h)
                observation = received["equalized_latent"]
            else:
                received = oracle_receive(raw, power, n0, h)
                observation = received["observation"]
            true_precision = h.abs().float().square()[:, None] * power.float() / n0[:, None]
            true_timestep = noise_variance_to_diffusion_timestep(true_precision.reciprocal())
            estimated_timestep = noise_variance_to_diffusion_timestep(received["noise_variance"])
            # U_tilde is in the actual transmitter's block coordinates.
            spatial, spatial_info = restore_spatial_latent(
                observation, {"orthogonal_matrix": observation_q, "group_grid": (8, 8)},
                {"noise_variance": received["noise_variance"]},
            )
            direct_task = restored_task = None
            if classifier is not None:
                direct_task = _task_result(classifier_forward(classifier, spatial, spatial_info, condition_on_snr), label)
            diffusion_generator.set_state(diffusion_state)
            try:
                restored = sampler(observation, received["precision"], observation_q, diffusion_generator,
                                   verbose=not getattr(args, "quiet", False))
            except ValueError as error:
                policy = getattr(sampler, "initialization_policy", None)
                raise ValueError(f"Restoration failed: strategy={name}, receiver_mode={strategy_receiver_mode}, "
                                 f"initialization_policy={policy}: {error}") from error
            if classifier is not None:
                restored_task = _task_result(classifier_forward(classifier, restored, spatial_info, condition_on_snr), label)
            direct_images[name] = _decode_latent(vae, spatial, vae_scale)
            images[name] = _decode_latent(vae, restored, vae_scale)
            frame_energy = (power * block_energy).sum(1)
            results[name] = {
                "receiver_mode": strategy_receiver_mode,
                "uses_orthogonal_transform": uses_q,
                "transmit_representation": "Q_transformed_blocks" if uses_q else "spatial_latent",
                "task_direct": direct_task,
                "task_diffusion": restored_task,
                "direct_vs_input": image_quality(direct_images[name], image, lpips_model),
                f"{method}_vs_input": image_quality(images[name], image, lpips_model),
                "group_power": power[0].cpu().tolist(),
                "true_amplitude_gain": power[0].sqrt().cpu().tolist(),
                "estimated_amplitude_gain": received["amplitude_gain"][0].cpu().tolist(),
                "true_timestep": true_timestep[0].cpu().tolist(),
                "estimated_timestep": estimated_timestep[0].cpu().tolist(),
                "precision": received["precision"][0].cpu().tolist(),
                "equalized_noise_variance": received["noise_variance"][0].cpu().tolist(),
                "energy_allocation_ratio": (allocated_ratio if uses_q else power * block_energy / budget)[0].cpu().tolist(),
                "frame_energy": frame_energy[0].item(),
                "target_frame_energy": budget,
                "frame_channel_uses": frame_channel_uses,
                "mean_symbol_power": (frame_energy[0] / frame_channel_uses).item(),
                "mean_real_component_power": (frame_energy[0] / real_component_count).item(),
                "direct_latent_mse": (spatial - latent).square().mean().item(),
                f"{method}_latent_mse": (restored - latent).square().mean().item(),
                "final_observation_loss": observation_loss(restored, observation, received["precision"], observation_q)[0].item(),
                "diffusion": sampler.last_diagnostics,
            }
            if strategy_receiver_mode == "oracle":
                results[name]["oracle_amplitude_gain"] = received["amplitude_gain"][0].cpu().tolist()
                results[name]["expected_direct_latent_mse"] = received["noise_variance"].mean().item()
            print(f"{name} ({strategy_receiver_mode}): 직접 복원 PSNR={results[name]['direct_vs_input']['psnr_db']:.2f} dB, "
                  f"{method_label} PSNR={results[name][f'{method}_vs_input']['psnr_db']:.2f} dB입니다.",
                  flush=True)
            if lpips_model is not None:
                print(f"{name} ({strategy_receiver_mode}): 직접 복원 LPIPS={results[name]['direct_vs_input']['lpips']:.5f}, "
                      f"{method_label} LPIPS={results[name][f'{method}_vs_input']['lpips']:.5f}", flush=True)
            payload = {"Q": observation_q.cpu(), "uses_orthogonal_transform": uses_q,
                        "receiver_mode": strategy_receiver_mode,
                        "clean_latent": latent.cpu(), "transformed_latent": channel_input.cpu(),
                        "raw_received_latent": raw.cpu(), "equalized_observation": observation.cpu(),
                        "direct_latent": spatial.cpu(), "spatial_received_latent": spatial.cpu(), "restored_latent": restored.cpu(),
                        "group_power": power.cpu(), "estimated_gain": received["amplitude_gain"].cpu(),
                        "precision": received["precision"].cpu(), "N0": n0.cpu(),
                        "true_timestep": true_timestep.cpu(), "estimated_timestep": estimated_timestep.cpu(),
                        "channel_coefficient": h.cpu(), "channel_gain_abs": h.abs().cpu(),
                        "epsilon": epsilon.cpu()}
            if strategy_receiver_mode == "oracle":
                payload["oracle_amplitude_gain"] = received["amplitude_gain"].cpu()
            if not getattr(args, "no_save_latents", False):
                torch.save(payload, output_dir / f"{name}_baseband.pt")
                if strategy_receiver_mode == "oracle":
                    tag = "allocated" if uses_q else "uniform_no_q"
                    torch.save(payload, output_dir / f"{tag}_oracle_state.pt")
    for name in images:
        filename = "reconstructed.png" if name == "allocated" else "uniform_reconstructed.png"
        direct_filename = "direct_reconstructed.png" if name == "allocated" else "uniform_direct_reconstructed.png"
        to_pil_image(images[name][0].float().cpu()).save(output_dir / filename)
        to_pil_image(direct_images[name][0].float().cpu()).save(output_dir / direct_filename)
        if receiver_modes[name] == "oracle":
            tag = "allocated" if name == "allocated" else "uniform_no_q"
            to_pil_image(images[name][0].float().cpu()).save(output_dir / f"{tag}_oracle_{method}.png")
            to_pil_image(direct_images[name][0].float().cpu()).save(output_dir / f"{tag}_oracle_direct.png")
    panels = [image[0]] + [images[name][0] for name in images]
    labels = ["Original"] + [f"{'Allocated + Q' if name == 'allocated' else 'Uniform, no Q'}: {receiver_modes[name]} {method_label}"
                              for name in images]
    _save_labeled_row(panels, labels, output_dir / "comparison.png")
    save_power_allocation(results, output_dir / "power_allocation.png", display="timestep")
    result = {"checkpoint": str(checkpoint_path), "epoch": int(checkpoint.get("epoch", -1)) + 1,
              "dataset": "ImageNet", "sample_index": args.sample_index, "input_image": args.input_image,
              "image_path": image_path, "label": label,
              "vae_latent_method": "sample",
              "min_gain": min_gain,
              "max_timestep": max_timestep,
              "gain_floor_infeasible_images": int(info.get(
                  "gain_floor_fallback", torch.zeros(1, dtype=torch.bool)
              ).sum().item()),
              "lpips_network": LPIPS_NETWORK if lpips_model is not None else None,
              "vae_vs_input": vae_metrics,
              "receiver_mode": receiver_mode,
              "uniform_receiver_mode": uniform_receiver_mode,
              "gain_estimator_checkpoint": estimator_path if estimator is not None else None,
              "uses_gain_estimator": estimator is not None, "uses_classifier": classifier is not None,
              "allocator_fingerprint": current_fingerprint,
              "training_allocator_fingerprint": trained_fingerprint,
              "allocator_matches_training": (current_fingerprint == trained_fingerprint
                                               if estimator is not None else None),
              "restoration_method": method,
              "gain_estimator_config": estimator_config, "snr_db": snr_db, "seed": args.seed,
              "channel_model": "awgn" if channel_model == "awgn" else config["channel_model"],
              "transmitter_training_channel_model": config["channel_model"], "perfect_csi": True,
              "noise_real_component_variance": config["noise_real_component_variance"], "N0": n0[0].item(),
              "mean_real_component_power": config["mean_real_component_power"],
              "allocator_architecture": config.get("allocator_architecture"),
              "channel_coefficient": {"real": h.real.cpu().tolist(), "imag": h.imag.cpu().tolist()},
              "effective_snr_db": info["allocator_snr_db"].cpu().tolist(),
              "orthogonal_transform": True, "group_grid": [8, 8], "num_groups": 64,
              "pilots_per_group": 0, "diffusion_steps": args.diffusion_steps,
              "same_channel_noise_for_strategies": True, "same_diffusion_noise_for_strategies": True,
              "same_channel_coefficient_for_strategies": True,
              "rng_streams": {"vae_seed": args.seed, "channel_seed": args.seed + 1,
                              "diffusion_seed": args.seed + 2},
              "strategies": results}
    with open(output_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Saved orthogonal ImageNet classification inference to {output_dir}")
    return result


def build_parser(*, include_legacy=False, default_receiver=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Transmitter checkpoint or directory containing latest.pth")
    parser.add_argument("--input_image", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, default=None if include_legacy else 10.)
    parser.add_argument("--receiver_mode", choices=("pilot", "oracle", "gain_estimator") if include_legacy
                        else ("oracle", "gain_estimator"), default=default_receiver,
                        help="Default: gain_estimator when supplied, otherwise oracle")
    parser.add_argument("--uniform_receiver_mode", choices=("oracle", "gain_estimator"),
                        default=None if include_legacy else "oracle",
                        help="Q uniform receiver (default: oracle)")
    parser.add_argument("--gain_estimator_checkpoint", default=None)
    parser.add_argument("--channel_model", choices=("awgn", "rayleigh"), default="awgn",
                        help="AWGN with h=1 (default), or one Rayleigh h per image")
    parser.add_argument("--strategy", choices=("both", "allocated", "uniform"), default="both",
                        help="allocated/both: allocated Q plus uniform Q=I; uniform: baseline only")
    parser.add_argument("--min_gain", type=float, default=0.)
    parser.add_argument("--max_timestep", type=int, choices=range(1000), metavar="0..999", default=None,
                        help="Allocation noise ceiling; default 999 for DDRM, opt-in for DPS")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    add_restoration_arguments(parser)
    parser.set_defaults(restoration_method=None if include_legacy else "ddrm")
    parser.add_argument("--guidance_scale", type=float, default=.1)
    parser.add_argument("--guidance_max_rms", "--guidance_max_norm", dest="guidance_max_rms", type=float, default=1.)
    parser.add_argument("--posterior_eta", type=float, default=0.)
    if include_legacy:
        parser.add_argument("--channel_gain_abs", type=float, nargs="+", default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_save_latents", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def validate_arguments(args, parser):
    if not 1 <= args.diffusion_steps <= 1000:
        parser.error("--diffusion_steps must be between 1 and 1000")
    if args.sample_index < 0 or (args.snr_db is not None and not math.isfinite(args.snr_db)):
        parser.error("sample_index must be non-negative and snr_db must be finite")
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        parser.error("--min_gain must be finite and non-negative")
    try:
        build_restoration_sampler(None, args)
    except ValueError as error:
        parser.error(str(error))


def run(args):
    return run_inference(args)


def main(argv=None, *, default_receiver=None):
    parser = build_parser(default_receiver=default_receiver)
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    return run(args)


if __name__ == "__main__":
    main()
