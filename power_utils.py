"""Shared model, diffusion-schedule, and evaluation helpers for power allocation."""

import math

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, normalize, resize

from models.models import construct_model


def sd15_alpha_bars() -> torch.Tensor:
    betas = torch.linspace(
        0.00085 ** 0.5,
        0.012 ** 0.5,
        1000,
        dtype=torch.float32,
    ).square()
    return torch.cumprod(1.0 - betas, dim=0)


def _spatial_grid(num_groups: int) -> tuple[int, int]:
    rows = math.isqrt(num_groups)
    while num_groups % rows != 0:
        rows -= 1
    return rows, num_groups // rows


class LegacyPowerAllocator(nn.Module):
    """Legacy power CNN with unchanged checkpoint parameter names."""

    def __init__(
        self,
        latent_channels: int,
        num_groups: int,
        hidden_dim: int,
        sampling: bool = False,
    ):
        super().__init__()
        self.sampling = sampling
        self.group_rows, self.group_cols = _spatial_grid(num_groups)
        self.group_pool = nn.AdaptiveAvgPool2d(
            (self.group_rows, self.group_cols)
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(
                latent_channels + 1,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        if sampling:
            self.log_sigma_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.log_sigma_head.weight)
            nn.init.zeros_(self.log_sigma_head.bias)
        else:
            self.mu_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.mu_head.weight)
            nn.init.zeros_(self.mu_head.bias)

    def forward(
        self,
        z0: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        conditioning_map = conditioning[:, None, None, None].expand(
            -1,
            1,
            z0.shape[2],
            z0.shape[3],
        )
        features = self.backbone(torch.cat([z0, conditioning_map], dim=1))
        if self.sampling:
            return self.group_pool(
                self.log_sigma_head(features)
            ).flatten(1)
        return self.group_pool(self.mu_head(features)).flatten(1)


def build_latent_classifier(
    model_name: str,
    input_channels: int,
    num_classes: int,
) -> nn.Module:
    classifier = construct_model(
        model_name,
        num_classes=num_classes,
        pretrained=True,
    )
    if not hasattr(classifier, "conv1"):
        raise ValueError("latent classifier requires a ResNet model")

    classifier.conv1 = nn.Conv2d(
        input_channels,
        classifier.conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    classifier.maxpool = nn.Identity()
    return classifier


def classifier_forward(
    classifier: nn.Module,
    received_latent: torch.Tensor,
    codec_info: dict,
    condition_classifier_on_snr: bool,
) -> torch.Tensor:
    if condition_classifier_on_snr:
        received_latent = torch.cat(
            [received_latent, codec_info["snr_map"]],
            dim=1,
        )
    return classifier(received_latent)


def maxvit_input(images: torch.Tensor) -> torch.Tensor:
    resized = resize(
        images,
        235,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return normalize(
        center_crop(resized, [224, 224]),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
