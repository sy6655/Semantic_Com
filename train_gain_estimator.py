"""Train gains of the 64 Q^T-transformed ImageNet transmission blocks."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.data_processing import construct_dataloaders
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, GAIN_ESTIMATOR_ARCHITECTURE, GAIN_ESTIMATOR_VERSION,
    GAIN_STATISTIC_NAMES, MEAN_REAL_COMPONENT_POWER, NOISE_REAL_COMPONENT_VARIANCE,
    RAYLEIGH_CHANNEL_MODEL, PatchGainEstimator,
    allocator_fingerprint, prepare_gain_estimator_input, sample_raw_channel,
)
from train_power_allocator import (
    PowerAllocator, RayleighFadingChannel, allocate_power, encode_latents,
    build_logger, cleanup_distributed, initialize_distributed,
    install_termination_handlers, save_config, unwrap_model,
)


def load_transmitter(checkpoint_path, device, *, checkpoint=None):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path /= "latest.pth"
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config.get("dataset") != "ImageNet" or not config.get("orthogonal_transform")
            or config.get("num_groups") != 64 or config.get("transmission_mode") != "analog"
            or config.get("pilots_per_group") != 0
            or config.get("channel_model") != RAYLEIGH_CHANNEL_MODEL
            or config.get("noise_real_component_variance") != NOISE_REAL_COMPONENT_VARIANCE
            or config.get("mean_real_component_power") != MEAN_REAL_COMPONENT_POWER
            or config.get("allocator_architecture") != ALLOCATOR_ARCHITECTURE):
        raise ValueError("Expected a v4 block-MLP ImageNet Rayleigh checkpoint with real noise variance N0 and P=1")
    allocator = PowerAllocator(config.get("latent_channels", 4), 64, config["hidden_dim"],
                               orthogonal_transform=True,
                               block_dim=config["allocator_block_dim"]).to(device)
    allocator.load_state_dict(checkpoint["allocator"])
    allocator.eval().requires_grad_(False)
    channel = RayleighFadingChannel(8, 8).to(device)
    return allocator, channel, config, checkpoint_path


def gain_estimator_step(images, vae, allocator, channel, estimator, snr_db,
                        temperature=1.0, generator=None):
    # Only the estimator is trained. Q, allocator and VAE stay frozen.
    with torch.no_grad():
        latent = encode_latents(images, vae, generator)
        channel_gain = channel.sample_channel_coefficient(latent, generator)
        _, power, info = allocate_power(latent, snr_db, allocator, channel, temperature,
                                       channel_coefficient=channel_gain)
        n0 = channel._noise_spectral_density(latent, snr_db)
        raw = sample_raw_channel(info["channel_input"], power, n0, channel_gain, generator)
        estimator_input = prepare_gain_estimator_input(raw, n0, channel_gain)
        target_log_gain = 0.5 * power.log()
    predicted_log_gain = estimator(estimator_input["channel_equalized_latent"],
                                   estimator_input["log_noise_variance"], estimator_input["log_power"])
    loss = F.mse_loss(predicted_log_gain, target_log_gain)
    log_error = predicted_log_gain.detach() - target_log_gain
    return {"loss": loss, "log_gain_mse": log_error.square().mean(),
            "gain_relative_mae": (log_error.clamp(-20, 20).exp() - 1).abs().mean()}


def run_training():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator_checkpoint", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Global MLP hidden width; v4 default is 2946->256->64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--snr_db_min", type=float, default=-10)
    parser.add_argument("--snr_db_max", type=float, default=20)
    parser.add_argument("--grad_clip", type=float, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--local_rank", "--local-rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if min(args.batch, args.epochs, args.hidden_dim, args.log_interval) <= 0 or args.lr <= 0:
        parser.error("batch, epochs, hidden_dim, log_interval and lr must be positive")
    if args.snr_db_min > args.snr_db_max:
        parser.error("snr_db_min must not exceed snr_db_max")
    device, distributed, rank, world_size = initialize_distributed(args.device, args.local_rank)
    torch.manual_seed(args.seed + rank)
    allocator, channel, transmitter_config, transmitter_path = load_transmitter(args.allocator_checkpoint, device)
    image_size = int(transmitter_config.get("image_size", 512))
    if image_size % 64:
        raise ValueError("Image size must be divisible by 64 for an 8x8 VAE block grid")
    patch_side = image_size // 64
    latent_channels = int(transmitter_config.get("latent_channels", 4))
    vae = AutoencoderKL.from_pretrained(
        transmitter_config["vae_model_id"], subfolder="vae",
        revision=transmitter_config["vae_model_revision"], torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).to(device).eval().requires_grad_(False)
    if int(vae.config.latent_channels) != latent_channels:
        raise ValueError("VAE channels do not match transmitter")
    train_loader, val_loader = construct_dataloaders(
        "ImageNet", args.data_path or transmitter_config["data_path"], args.batch,
        normalize=False, image_size=image_size, distributed=distributed, seed=args.seed,
    )
    estimator = PatchGainEstimator(latent_channels, patch_side, patch_side, args.hidden_dim).to(device)
    if distributed:
        options = {"device_ids": [args.local_rank]} if device.type == "cuda" else {}
        estimator = DistributedDataParallel(estimator, **options)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path("checkpoints_gain_estimator") / "ImageNet" / args.exp_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = build_logger(output_dir / "train.log", rank == 0)
    config = vars(args).copy()
    config.update(stage="block_gain_estimator", estimator_version=GAIN_ESTIMATOR_VERSION, dataset="ImageNet",
                  estimator_architecture=GAIN_ESTIMATOR_ARCHITECTURE,
                  channel_model=RAYLEIGH_CHANNEL_MODEL, perfect_csi=True,
                  mean_real_component_power=MEAN_REAL_COMPONENT_POWER,
                  fading_granularity="one_complex_h_per_image_shared_by_all_blocks",
                  latent_channels=latent_channels, patch_shape=[patch_side, patch_side],
                  vae_latent_method="sample",
                  group_grid=[8, 8], num_groups=64, image_size=image_size,
                  allocator_checkpoint=str(transmitter_path.resolve()),
                  allocator_fingerprint=allocator_fingerprint(allocator),
                  estimator_input="R_h=Y/h,log(nu=N0/abs(h)^2),log(P)",
                  estimator_statistics=list(GAIN_STATISTIC_NAMES),
                  estimator_embedding_dim=8, estimator_conv_channels=[1, 16, 32],
                  estimator_conv_kernel_size=5,
                  estimator_global_input_dim=2946, estimator_head_dims=[2946, args.hidden_dim, 64],
                  estimator_parameter_count=sum(p.numel() for p in estimator.parameters()),
                  estimator_output="log_transmit_amplitude_gain", loss="log_gain_mse",
                  precision="abs(h)^2*a_hat^2/N0", noise_real_component_variance=NOISE_REAL_COMPONENT_VARIANCE,
                  uses_reverse_diffusion=False, world_size=world_size)
    start_epoch, global_step = 0, 0
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("allocator_fingerprint", "hidden_dim", "patch_shape", "channel_model", "estimator_version",
                    "noise_real_component_variance", "mean_real_component_power", "estimator_architecture"):
            if previous["config"].get(key) != config[key]:
                raise ValueError(f"Resume checkpoint mismatch: {key}")
        unwrap_model(estimator).load_state_dict(previous["estimator"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch, global_step = previous["epoch"] + 1, previous["global_step"]
    if rank == 0:
        save_config(output_dir / "config.json", config)
    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        metrics = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            training = phase == "train"
            estimator.train(training)
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            generator = torch.Generator(device=device).manual_seed(args.seed + rank)
            for batch_index, (images, _) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                snr = args.snr_db_min + (args.snr_db_max - args.snr_db_min) * torch.rand(
                    images.shape[0], device=device, generator=None if training else generator
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    result = gain_estimator_step(images, vae, allocator, channel, estimator, snr,
                                                 transmitter_config.get("temperature", 1.0),
                                                 None if training else generator)
                finite = torch.isfinite(result["loss"]).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite gain estimator loss")
                if training:
                    scaler.scale(result["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(estimator.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                size = images.shape[0]
                totals += torch.tensor([result["loss"].item() * size,
                                        result["gain_relative_mae"].item() * size, size], device=device)
                if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                    logger.info("epoch=%d phase=%s batch=%d/%d log_gain_mse=%.6f relative_mae=%.4f",
                                epoch + 1, phase, batch_index + 1, len(loader), result["loss"].item(),
                                result["gain_relative_mae"].item())
            if distributed:
                dist.all_reduce(totals)
            metrics[phase] = {"log_gain_mse": (totals[0] / totals[2]).item(),
                              "gain_relative_mae": (totals[1] / totals[2]).item()}
        scheduler.step()
        if rank == 0:
            torch.save({"estimator": unwrap_model(estimator).state_dict(), "config": config,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                        "metrics": metrics}, output_dir / "latest.pth")
            logger.info("epoch=%d metrics=%s checkpoint=%s", epoch + 1, metrics, output_dir / "latest.pth")
        if distributed:
            dist.barrier()


def main():
    install_termination_handlers()
    try:
        run_training()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
