"""Two-rank Cityscapes Q512 training smoke, with tiny local VAE/task models."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation_transport import SegmentationLayout, build_segmentation_allocator
from train_power_allocator import RayleighFadingChannel, cleanup_distributed, initialize_distributed
from train_segmentation_power_allocator import segmentation_step


class TinyVAE(torch.nn.Module):
    def encode(self, images):
        rgb = F.interpolate(images, (128, 256))
        mean = torch.cat((rgb, rgb[:, :1]), 1)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(
            torch.cat((mean, -2 * torch.ones_like(mean)), 1)))


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(4, 19, 1)

    def forward(self, latent, output_size):
        return F.interpolate(self.head(latent), output_size)


def assert_synchronized(module):
    values = torch.cat([parameter.detach().flatten() for parameter in module.parameters()])
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    for other in gathered:
        torch.testing.assert_close(values, other, atol=0, rtol=0)


def main():
    torch.set_num_threads(1)
    device, distributed, rank, world_size = initialize_distributed("cpu", int(os.environ["LOCAL_RANK"]))
    assert distributed and world_size == 2
    torch.manual_seed(12)
    layout = SegmentationLayout()
    allocator = build_segmentation_allocator(layout, 8)
    with torch.no_grad():
        allocator.orthogonal.upper.normal_(std=.01)
        allocator.block_allocator.logit_head[-1].weight.normal_(std=.1)
    allocator = DistributedDataParallel(allocator)
    segmenter = DistributedDataParallel(TinySegmenter())
    vae, channel = TinyVAE(), RayleighFadingChannel(16, 32)
    optimizer = torch.optim.SGD(list(allocator.parameters()) + list(segmenter.parameters()), lr=.01)
    torch.manual_seed(100 + rank)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = segmentation_step(torch.rand(1, 3, 16, 32), torch.randint(0, 19, (1, 16, 32)),
                                   vae, allocator, segmenter, channel, None, torch.tensor([10.]), 1., 999)
        result["loss"].backward()
        for parameter in (allocator.module.orthogonal.upper,
                          allocator.module.block_allocator.logit_head[-1].weight,
                          segmenter.module.head.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
        assert_synchronized(allocator.module)
        assert_synchronized(segmenter.module)
    dist.barrier()
    if rank == 0:
        print("Cityscapes DDP smoke passed: two ranks, Q512/allocator/segmenter gradients and synchronization", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
