"""Train gains in frozen ImageNet or Cityscapes Q-transformed block space."""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.data_processing import construct_cityscapes_dataloaders, construct_dataloaders
from inference_power import allocate_power_for_inference
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, GAIN_ESTIMATOR_ARCHITECTURE, GAIN_ESTIMATOR_VERSION,
    GAIN_STATISTIC_NAMES, MEAN_REAL_COMPONENT_POWER, NOISE_REAL_COMPONENT_VARIANCE,
    RAYLEIGH_CHANNEL_MODEL, PatchGainEstimator,
    allocator_fingerprint, prepare_gain_estimator_input, sample_raw_channel,
)
from train_power_allocator import (
    PowerAllocator, RayleighFadingChannel, WarmupCosineLR, allocate_power, encode_latents,
    build_logger, build_training_scheduler, cleanup_distributed, initialize_distributed,
    install_termination_handlers, optimizer_step_with_schedule, save_config,
    training_schedule_config, unwrap_model,
)


def load_transmitter(checkpoint_path, device, *, checkpoint=None, allow_segmentation=False):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path /= "latest.pth"
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if allow_segmentation and config.get("dataset") == "Cityscapes":
        from segmentation_transport import load_segmentation_transmitter
        allocator, channel, _, _, path = load_segmentation_transmitter(checkpoint_path, device, checkpoint=checkpoint)
        return allocator, channel, config, path
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
                        temperature=1.0, generator=None, *, max_timestep=None, min_gain=0.0):
    # Only the estimator is trained. Q, allocator and VAE stay frozen.
    module = unwrap_model(allocator)
    rows, cols = module.group_rows, module.group_cols
    with torch.no_grad():
        latent = encode_latents(images, vae, generator)
        channel_gain = channel.sample_channel_coefficient(latent, generator)
        if max_timestep is None and min_gain == 0:
            _, power, info = allocate_power(latent, snr_db, allocator, channel, temperature,
                                           channel_coefficient=channel_gain)
        else:
            _, power, info = allocate_power_for_inference(
                latent, snr_db, allocator, channel, temperature, channel_coefficient=channel_gain,
                max_timestep=max_timestep, min_gain=min_gain,
            )
        n0 = channel._noise_spectral_density(latent, snr_db)
        raw = sample_raw_channel(info["channel_input"], power, n0, channel_gain, generator, rows, cols)
        estimator_input = prepare_gain_estimator_input(raw, n0, channel_gain, rows=rows, cols=cols)
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
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch per GPU (default: ImageNet 32, Cityscapes 1)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers; omitted keeps each dataset's existing default")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Global MLP hidden width; input/output dimensions follow the checkpoint block grid")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=float, default=0.5,
                        help="Linear warmup duration for new training, measured in epochs")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Warmup initial LR / peak LR")
    parser.add_argument("--min_lr_ratio", type=float, default=0.5,
                        help="Final cosine LR / peak LR; updated after each optimizer step")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--snr_db_min", type=float, default=-10)
    parser.add_argument("--snr_db_max", type=float, default=20)
    parser.add_argument("--max_timestep", type=int, choices=range(1000), metavar="0..999", default=None,
                        help="Transmitter gain floor (Cityscapes: allocator checkpoint value, usually 999; ImageNet: off)")
    parser.add_argument("--min_gain", type=float, default=0., help="Minimum transmit amplitude during estimator training")
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
    if ((args.batch is not None and args.batch <= 0)
            or min(args.epochs, args.hidden_dim, args.log_interval) <= 0 or args.lr <= 0):
        parser.error("batch, epochs, hidden_dim, log_interval and lr must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("num_workers must be non-negative")
    if not math.isfinite(args.min_gain) or args.min_gain < 0:
        parser.error("min_gain must be finite and non-negative")
    if not math.isfinite(args.warmup_epochs) or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup_epochs must be in [0,epochs)")
    if not math.isfinite(args.warmup_start_factor) or not 0 <= args.warmup_start_factor <= 1:
        parser.error("warmup_start_factor must be in [0,1]")
    if not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1:
        parser.error("min_lr_ratio must be in [0,1]")
    if args.snr_db_min > args.snr_db_max:
        parser.error("snr_db_min must not exceed snr_db_max")
    device, distributed, rank, world_size = initialize_distributed(args.device, args.local_rank)
    torch.manual_seed(args.seed + rank)
    allocator, channel, transmitter_config, transmitter_path = load_transmitter(
        args.allocator_checkpoint, device, allow_segmentation=True,
    )
    dataset_name = transmitter_config.get("dataset", "ImageNet")
    args.batch = args.batch if args.batch is not None else (1 if dataset_name == "Cityscapes" else 32)
    if dataset_name == "Cityscapes":
        rows, cols = transmitter_config["group_grid"]
        patch_height, patch_width = transmitter_config["patch_shape"]
        input_shape = transmitter_config["input_shape"]
        latent_shape = transmitter_config["latent_shape"]
        image_size = input_shape[0]
        if args.max_timestep is None:
            args.max_timestep = int(transmitter_config.get("max_timestep", 999))
    else:
        image_size = int(transmitter_config.get("image_size", 512))
        if image_size % 64:
            raise ValueError("Image size must be divisible by 64 for an 8x8 VAE block grid")
        rows = cols = 8
        patch_height = patch_width = image_size // 64
        input_shape, latent_shape = [image_size] * 2, [image_size // 8] * 2
    num_groups = rows * cols
    latent_channels = int(transmitter_config.get("latent_channels", 4))
    vae = AutoencoderKL.from_pretrained(
        transmitter_config["vae_model_id"], subfolder="vae",
        revision=transmitter_config["vae_model_revision"], torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).to(device).eval().requires_grad_(False)
    if int(vae.config.latent_channels) != latent_channels:
        raise ValueError("VAE channels do not match transmitter")
    if dataset_name == "Cityscapes":
        train_loader, val_loader = construct_cityscapes_dataloaders(
            args.data_path or transmitter_config["data_path"], args.batch, distributed, args.seed,
            4 if args.num_workers is None else args.num_workers,
        )
    else:
        train_loader, val_loader = construct_dataloaders(
            "ImageNet", args.data_path or transmitter_config["data_path"], args.batch,
            normalize=False, image_size=image_size, distributed=distributed, seed=args.seed,
            num_workers=args.num_workers,
        )
    estimator = PatchGainEstimator(
        latent_channels, patch_height, patch_width, args.hidden_dim, group_grid=(rows, cols),
    )
    estimator = estimator.to(device)
    if distributed:
        options = {"device_ids": [args.local_rank]} if device.type == "cuda" else {}
        estimator = DistributedDataParallel(estimator, **options)
    parameters = list(estimator.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path("checkpoints_gain_estimator") / dataset_name / args.exp_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = build_logger(output_dir / "train.log", rank == 0)
    config = vars(args).copy()
    config.update(stage="block_gain_estimator", estimator_version=GAIN_ESTIMATOR_VERSION, dataset=dataset_name,
                  estimator_architecture=GAIN_ESTIMATOR_ARCHITECTURE,
                  channel_model=RAYLEIGH_CHANNEL_MODEL, perfect_csi=True,
                  mean_real_component_power=MEAN_REAL_COMPONENT_POWER,
                  fading_granularity="one_complex_h_per_image_shared_by_all_blocks",
                  latent_channels=latent_channels, patch_shape=[patch_height, patch_width],
                  vae_latent_method="sample",
                  group_grid=[rows, cols], num_groups=num_groups, image_size=image_size,
                  input_shape=input_shape, latent_shape=latent_shape,
                  allocator_checkpoint=str(transmitter_path.resolve()),
                  allocator_fingerprint=allocator_fingerprint(allocator),
                  estimator_input="R_h=Y/h,log(nu=N0/abs(h)^2),log(P)",
                  estimator_statistics=list(GAIN_STATISTIC_NAMES),
                  estimator_embedding_dim=8, estimator_conv_channels=[1, 16, 32],
                  estimator_conv_kernel_size=5,
                  estimator_global_input_dim=num_groups * 46 + 2,
                  estimator_head_dims=[num_groups * 46 + 2, args.hidden_dim, num_groups],
                  estimator_parameter_count=sum(p.numel() for p in estimator.parameters()),
                  estimator_output="log_transmit_amplitude_gain", loss="log_gain_mse",
                  precision="abs(h)^2*a_hat^2/N0", noise_real_component_variance=NOISE_REAL_COMPONENT_VARIANCE,
                  uses_reverse_diffusion=False, world_size=world_size)
    start_epoch, global_step, previous = 0, 0, None
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("allocator_fingerprint", "hidden_dim", "patch_shape", "channel_model", "estimator_version",
                    "noise_real_component_variance", "mean_real_component_power", "estimator_architecture",
                    "dataset", "group_grid", "num_groups", "latent_channels"):
            if previous["config"].get(key) != config[key]:
                raise ValueError(f"Resume checkpoint mismatch: {key}")
        unwrap_model(estimator).load_state_dict(previous["estimator"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch, global_step = previous["epoch"] + 1, previous["global_step"]
    scheduler = build_training_scheduler(
        optimizer, args.epochs, len(train_loader), args.warmup_epochs,
        args.warmup_start_factor, args.min_lr_ratio, previous,
    )
    config.update(lr_schedule=training_schedule_config(scheduler), lr_steps_per_epoch=len(train_loader))
    if rank == 0:
        save_config(output_dir / "config.json", config)
    logger.info("lr_schedule=%s resumed=%s next_lr_estimator=%.8g",
                config["lr_schedule"], previous is not None, optimizer.param_groups[0]["lr"])
    logger.info("dataset=%s frozen_transmitter=%s grid=%dx%d patch=%dx%d groups=%d "
                "snr_db_uniform=[%g,%g] max_timestep=%s min_gain=%g mean_real_component_power=1",
                dataset_name, transmitter_path, rows, cols, patch_height, patch_width, num_groups,
                args.snr_db_min, args.snr_db_max, args.max_timestep, args.min_gain)
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
                batch_lr = optimizer.param_groups[0]["lr"]
                images = images.to(device, non_blocking=True)
                snr = args.snr_db_min + (args.snr_db_max - args.snr_db_min) * torch.rand(
                    images.shape[0], device=device, generator=None if training else generator
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    floor_options = ({"max_timestep": args.max_timestep, "min_gain": args.min_gain}
                                     if args.max_timestep is not None or args.min_gain > 0 else {})
                    result = gain_estimator_step(images, vae, allocator, channel, estimator, snr,
                                                 transmitter_config.get("temperature", 1.0),
                                                 None if training else generator, **floor_options)
                finite = torch.isfinite(result["loss"]).to(torch.int32)
                if distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite gain estimator loss")
                if training:
                    scaler.scale(result["loss"]).backward()
                    optimizer_step_with_schedule(optimizer, scheduler, scaler, parameters, args.grad_clip)
                    global_step += 1
                size = images.shape[0]
                totals += torch.tensor([result["loss"].item() * size,
                                        result["gain_relative_mae"].item() * size, size], device=device)
                if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                    logger.info("epoch=%d phase=%s batch=%d/%d log_gain_mse=%.6f relative_mae=%.4f lr_estimator=%.8g",
                                epoch + 1, phase, batch_index + 1, len(loader), result["loss"].item(),
                                result["gain_relative_mae"].item(), batch_lr)
            if distributed:
                dist.all_reduce(totals)
            metrics[phase] = {"log_gain_mse": (totals[0] / totals[2]).item(),
                              "gain_relative_mae": (totals[1] / totals[2]).item()}
        if not isinstance(scheduler, WarmupCosineLR):
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
