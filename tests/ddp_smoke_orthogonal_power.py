"""Local DDP check with a sampled VAE posterior and no pretrained downloads.

torchrun --standalone --nproc_per_node=2 tests/ddp_smoke_orthogonal_power.py
Pass --device cuda to exercise NCCL on a host with two GPUs.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orthogonal_power import PatchGainEstimator
from train_gain_estimator import gain_estimator_step
from train_power_allocator import (
    PowerAllocator, RayleighFadingChannel, cleanup_distributed,
    initialize_distributed, training_step,
)


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)

    def encode(self, images):
        mean = self.encoder(F.avg_pool2d(images, 8))
        posterior = DiagonalGaussianDistribution(torch.cat((mean, -2 * torch.ones_like(mean)), 1))
        return type("EncodedLatent", (), {"latent_dist": posterior})()


def assert_synchronized(module):
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])
    gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, parameters)
    for other in gathered:
        torch.testing.assert_close(parameters, other, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device, distributed, rank, world_size = initialize_distributed(args.device, local_rank)
    if not distributed or world_size < 2:
        raise ValueError("Run this check with torchrun --nproc_per_node=2")
    torch.set_num_threads(1)
    torch.manual_seed(91)
    vae = TinyVAE().to(device).eval().requires_grad_(False)
    allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16).to(device)
    with torch.no_grad():
        allocator.orthogonal.upper.normal_(std=0.03)
        allocator.block_allocator.logit_head[-1].weight.normal_(std=0.3)
    classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1024, 7)).to(device)
    options = {"device_ids": [local_rank]} if device.type == "cuda" else {}
    allocator = DistributedDataParallel(allocator, **options)
    classifier = DistributedDataParallel(classifier, **options)
    channel = RayleighFadingChannel().to(device)
    optimizer = torch.optim.SGD(list(allocator.parameters()) + list(classifier.parameters()), lr=0.01)
    original_q_parameters = allocator.module.orthogonal.upper.detach().clone()
    torch.manual_seed(500 + rank)
    for _ in range(2):
        images = torch.rand(2, 3, 128, 128, device=device)
        labels = torch.tensor([rank % 7, (rank + 1) % 7], device=device)
        optimizer.zero_grad(set_to_none=True)
        result = training_step(images, labels, vae, allocator, classifier, channel,
                               torch.tensor([0., 5.], device=device), condition_classifier_on_snr=False)
        result["loss"].backward()
        for parameter in (allocator.module.orthogonal.upper, allocator.module.block_allocator.logit_head[-1].weight,
                          classifier.module[1].weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
        assert_synchronized(allocator.module)
        assert_synchronized(classifier.module)
    assert not torch.equal(original_q_parameters, allocator.module.orthogonal.upper)
    assert all(parameter.grad is None for parameter in vae.parameters())

    frozen_allocator = allocator.module.eval().requires_grad_(False)
    estimator = DistributedDataParallel(PatchGainEstimator(4, 2, 2, 8).to(device), **options)
    estimator_optimizer = torch.optim.SGD(estimator.parameters(), lr=0.01)
    for _ in range(2):
        images = torch.rand(2, 3, 128, 128, device=device)
        estimator_optimizer.zero_grad(set_to_none=True)
        result = gain_estimator_step(images, vae, frozen_allocator, channel, estimator,
                                     torch.tensor([0., 5.], device=device))
        result["loss"].backward()
        assert torch.isfinite(estimator.module.log_gain_head.weight.grad).all()
        assert estimator.module.log_gain_head.weight.grad.abs().sum() > 0
        estimator_optimizer.step()
        assert_synchronized(estimator.module)
    dist.barrier()
    if rank == 0:
        print(f"DDP smoke passed: world_size={world_size}, device={device.type}; "
              "Q/allocator/classifier and gain estimator stay synchronized with VAE.sample()", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
