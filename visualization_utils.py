from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from power_utils import sd15_alpha_bars


def noise_variance_to_diffusion_timestep(
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    """Map additive-noise variance to the first matching SD1.5 timestep."""
    alpha_bars = sd15_alpha_bars().to(
        device=noise_variance.device,
        dtype=noise_variance.dtype,
    )
    lambda_schedule = (1.0 - alpha_bars) / alpha_bars
    timesteps = torch.searchsorted(
        lambda_schedule,
        noise_variance.contiguous().clamp_min(0.0),
        right=False,
    )
    return timesteps.clamp_max(alpha_bars.numel() - 1)


def effective_alpha_bar_to_diffusion_timestep(
    effective_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = effective_alpha_bar.clamp(1e-8, 1.0)
    noise_variance = (1.0 - alpha_bar) / alpha_bar
    return noise_variance_to_diffusion_timestep(noise_variance)


def save_diffusion_timestep_overlay(
    image: torch.Tensor,
    timesteps: torch.Tensor,
    group_rows: int,
    group_cols: int,
    output_path: Path,
) -> None:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    timestep_map = timesteps.detach().float().cpu().reshape(
        1,
        1,
        group_rows,
        group_cols,
    )
    timestep_min = timestep_map.min()
    timestep_max = timestep_map.max()
    if (timestep_max - timestep_min).item() > 0.0:
        normalized = (
            (timestep_map - timestep_min)
            / (timestep_max - timestep_min)
        )
    else:
        normalized = torch.full_like(timestep_map, 0.5)
    normalized = F.interpolate(
        normalized,
        size=image.shape[-2:],
        mode="nearest",
    )[0, 0]
    heatmap = torch.stack(
        (
            normalized,
            1.0 - (2.0 * normalized - 1.0).abs(),
            1.0 - normalized,
        ),
        dim=0,
    )
    overlay = 0.25 * image + 0.75 * heatmap
    overlay_image = to_pil_image(overlay.clamp(0.0, 1.0))

    label_height = 28
    width, height = overlay_image.size
    canvas = Image.new(
        "RGB",
        (width, height + label_height),
        color="white",
    )
    canvas.paste(overlay_image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 6),
        (
            "SD1.5 diffusion timestep: "
            f"blue={int(timestep_min.item())}, "
            f"red={int(timestep_max.item())}"
        ),
        fill="black",
    )
    for row in range(1, group_rows):
        y = label_height + round(row * height / group_rows)
        draw.line((0, y, width, y), fill="white", width=1)
    for col in range(1, group_cols):
        x = round(col * width / group_cols)
        draw.line(
            (x, label_height, x, label_height + height),
            fill="white",
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
