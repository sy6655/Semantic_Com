import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from datasets.data_processing import CityscapesSemanticDataset
from inference_power import ActualPowerTransport
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    PowerAllocator,
    encode_latents,
)
from train_segmentation_power_allocator import (
    AnalogAWGNChannel,
    PILOTS_PER_GROUP,
    allocate_power_with_timestep_floor,
)
from visualization_utils import (
    noise_variance_to_diffusion_timestep,
    save_diffusion_timestep_overlay,
)


CITYSCAPES_PALETTE = torch.tensor(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=torch.uint8,
)


def _colorize_ground_truth(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().long().cpu()
    ignore = mask.eq(255)
    safe_mask = mask.clamp(min=0, max=len(CITYSCAPES_PALETTE) - 1)
    color = CITYSCAPES_PALETTE[safe_mask]
    color[ignore] = 0
    return color.permute(2, 0, 1)


def _decode_latent(
    vae: AutoencoderKL,
    latent: torch.Tensor,
) -> torch.Tensor:
    scaling_factor = float(vae.config.scaling_factor)
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
    for index, (panel, label) in enumerate(zip(pil_images, labels)):
        left = index * panel_width
        canvas.paste(panel, (left, label_height))
        draw.text((left + 4, 5), label, fill="black")
    canvas.save(output_path)


def legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--max_timestep", type=int, default=999)
    parser.add_argument("--vae_path", default=None)
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
    snr_db = args.snr_db if args.snr_db is not None else configured_snr[0]
    max_timestep = args.max_timestep

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

    dataset = CityscapesSemanticDataset(
        args.data_path or config["data_path"],
        "val",
    )
    image, ground_truth = dataset[args.sample_index]
    to_pil_image(_colorize_ground_truth(ground_truth)).save(
        output_dir / "ground_truth.png"
    )
    image = image.unsqueeze(0).to(device)

    allocator = PowerAllocator(
        int(vae.config.latent_channels),
        int(config["num_groups"]),
        int(config["hidden_dim"]),
        sampling=False,
    ).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    allocator.eval()

    channel = AnalogAWGNChannel(
        allocator.group_rows,
        allocator.group_cols,
        snr_db,
        config.get("channel_gain_abs", [1.0]),
        PILOTS_PER_GROUP,
    ).to(device)
    transport = ActualPowerTransport(channel, "oracle")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    with torch.inference_mode():
        z0 = encode_latents(image, vae)
        snr = torch.full((1,), snr_db, device=device)
        learned_ratio, learned_group_power, allocation_info = (
            allocate_power_with_timestep_floor(
                z0,
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

        # Use an equal allocation weight for every spatial group while
        # preserving exactly the same total frame-energy constraint.
        uniform_weights = torch.full_like(
            learned_ratio,
            1.0 / learned_ratio.shape[1],
        )
        uniform_normalizer = (
            uniform_weights * allocation_info["unscaled_block_energy"]
        ).sum(dim=1, keepdim=True)
        uniform_group_power = (
            allocation_info["frame_channel_uses"]
            * uniform_weights
            / uniform_normalizer
        )

        # Reuse the exact same payload noise for both strategies.
        channel_state = generator.get_state()
        learned_received, learned_channel_info, _ = transport(
            z0,
            learned_group_power,
            snr,
            generator,
        )
        generator.set_state(channel_state)
        uniform_received, _, _ = transport(
            z0,
            uniform_group_power,
            snr,
            generator,
        )
        clean_reconstruction = _decode_latent(vae, z0)[0]
        learned_reconstruction = _decode_latent(
            vae,
            learned_received,
        )[0]
        uniform_reconstruction = _decode_latent(
            vae,
            uniform_received,
        )[0]
        learned_timesteps = noise_variance_to_diffusion_timestep(
            learned_channel_info["noise_variance"]
        )

    output_path = output_dir / "diffusion_timestep_overlay.png"
    save_diffusion_timestep_overlay(
        image[0],
        learned_timesteps[0],
        allocator.group_rows,
        allocator.group_cols,
        output_path,
    )
    received_comparison_path = output_dir / "received_comparison.png"
    _save_labeled_row(
        [
            clean_reconstruction,
            learned_reconstruction,
            uniform_reconstruction,
        ],
        [
            "Clean VAE latent",
            f"Learned received ({snr_db:g} dB)",
            f"Uniform received ({snr_db:g} dB)",
        ],
        received_comparison_path,
    )
    print(f"Saved visualization to {output_dir}")


def main() -> None:
    from segmentation_orthogonal_power import dispatch
    dispatch("allocation", legacy_main)


if __name__ == "__main__":
    main()
