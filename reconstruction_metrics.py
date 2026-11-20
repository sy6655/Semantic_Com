"""Image quality metrics against the original, uncropped RGB image."""

import torch

LPIPS_NETWORK = "vgg_v0.1"


def load_lpips(device, enabled=True, local_files_only=False):
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("LPIPS requires the lpips package: python -m pip install lpips") from error
    if local_files_only:
        from pathlib import Path
        from urllib.parse import urlparse
        from torchvision.models import VGG16_Weights

        filename = Path(urlparse(VGG16_Weights.IMAGENET1K_V1.url).path).name
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not weights_path.is_file():
            raise FileNotFoundError(f"LPIPS VGG16 weights are not cached: {weights_path}")
    model = lpips.LPIPS(net="vgg", version="0.1", verbose=False)
    return model.to(
        device=device, dtype=torch.float32
    ).eval().requires_grad_(False)


@torch.no_grad()
def image_quality_batch(reconstructed, reference, lpips_model=None):
    if reconstructed.shape != reference.shape or reconstructed.ndim != 4 or reconstructed.shape[1] != 3:
        raise ValueError("Image metrics require matching [B,3,H,W] RGB tensors in [0,1]")
    reconstructed, reference = reconstructed.float(), reference.float()
    mse = (reconstructed - reference).square().flatten(1).mean(1)
    result = {"mse": mse, "psnr_db": -10 * mse.clamp_min(1e-12).log10()}
    if lpips_model is not None:
        # LPIPS v0.1 expects RGB in [-1,1], without ImageNet normalization.
        result["lpips"] = lpips_model(
            reconstructed * 2 - 1, reference * 2 - 1
        ).reshape(reconstructed.shape[0], -1).mean(1)
    return result


def image_quality(reconstructed, reference, lpips_model=None):
    return {name: values.mean().item() for name, values in
            image_quality_batch(reconstructed, reference, lpips_model).items()}
