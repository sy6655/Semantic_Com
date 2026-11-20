import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from datasets.data_processing import construct_dataloaders
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    PowerAllocator,
    QuantizedAWGNChannel,
    RayleighFadingChannel,
    allocate_power,
    encode_latents,
)
from visualization_utils import (
    noise_variance_to_diffusion_timestep,
    save_diffusion_timestep_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
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
    snr_db = args.snr_db if args.snr_db is not None else configured_snr[0]

    output_dir = Path(
        args.output_dir
        or checkpoint_path.parent
        / "visualizations"
        / f"sample_{args.sample_index:06d}_snr_{snr_db:g}"
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
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    _, validloader = construct_dataloaders(
        dataset=config["dataset"],
        data_path=config["data_path"],
        batch_size=1,
        normalize=False,
        image_size=int(config.get("image_size", 512)),
    )
    image, _ = validloader.dataset[args.sample_index]
    image = image.unsqueeze(0).to(device)

    allocator = PowerAllocator(
        int(vae.config.latent_channels),
        config["num_groups"],
        config["hidden_dim"],
        sampling=False,
        orthogonal_transform=bool(config.get("orthogonal_transform", False)),
        block_dim=config.get("allocator_block_dim", 256),
    ).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    allocator.eval()

    if config.get("channel_model") == "rayleigh_block_fading":
        channel = RayleighFadingChannel(allocator.group_rows, allocator.group_cols, snr_db).to(device)
    else:
        channel = QuantizedAWGNChannel(
            allocator.group_rows, allocator.group_cols, config.get("qam_order", 256),
            config.get("quantization_amplitude", 3.0), snr_db,
            config.get("channel_gain_abs", [1.0]), config.get("pilots_per_group", 4),
            config.get("transmission_mode", "qam"),
        ).to(device)
    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(0)
        z0 = encode_latents(image, vae, generator)
        snr = torch.full((1,), snr_db, device=device)
        h = (channel.sample_channel_coefficient(z0, generator)
             if isinstance(channel, RayleighFadingChannel) else None)
        _, learned_group_power, allocation_info = allocate_power(
            z0,
            snr,
            allocator,
            channel,
            config.get("temperature", 1.0),
            channel_coefficient=h,
        )
        _, channel_info = channel(
            allocation_info["channel_input"],
            learned_group_power,
            epsilon=torch.zeros_like(z0),
            snr_db=snr,
            **({"channel_coefficient": h} if h is not None else {}),
        )
        timesteps = noise_variance_to_diffusion_timestep(
            channel_info["noise_variance"]
        )

    original = image[0].float().cpu()
    if config.get("orthogonal_transform", False):
        # Transformed block IDs have no spatial correspondence with RGB
        # patches. Render the 64 powers on a plain block grid instead.
        from PIL import Image, ImageDraw
        powers = learned_group_power[0].float().cpu().reshape(8, 8)
        normalized = (powers - powers.min()) / (powers.max() - powers.min()).clamp_min(1e-12)
        canvas = Image.new("RGB", (512, 550), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Q^T block power a_b^2 (block IDs, row-major)", fill="black")
        for row in range(8):
            for col in range(8):
                value = float(normalized[row, col])
                box = (col * 64, 38 + row * 64, (col + 1) * 64, 38 + (row + 1) * 64)
                draw.rectangle(box, fill=(int(255 * value), 80, int(255 * (1 - value))), outline="white")
                draw.text((box[0] + 3, box[1] + 4), f"{row * 8 + col}: {powers[row, col]:.2f}", fill="white")
        output_path = output_dir / "transformed_block_power.png"
        canvas.save(output_path)
        print(f"Saved transformed-block allocation to {output_path}")
        return
    output_path = output_dir / "diffusion_timestep_overlay.png"
    save_diffusion_timestep_overlay(
        original,
        timesteps[0],
        allocator.group_rows,
        allocator.group_cols,
        output_path,
    )

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
