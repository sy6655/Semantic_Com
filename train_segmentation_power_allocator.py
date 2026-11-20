import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.data_processing import (
    CITYSCAPES_IGNORE_INDEX,
    CITYSCAPES_INPUT_HEIGHT,
    CITYSCAPES_INPUT_WIDTH,
    CITYSCAPES_NUM_CLASSES,
    construct_cityscapes_dataloaders,
)
from inference_power import ActualPowerTransport
from models.models import construct_model
from train_power_allocator import (
    VAE_MODEL_ID,
    VAE_MODEL_REVISION,
    VAE_SCALE,
    PowerAllocator,
    QuantizedAWGNChannel,
    RayleighFadingChannel,
    allocate_power,
    build_logger,
    cleanup_distributed,
    encode_latents,
    initialize_distributed,
    install_termination_handlers,
    save_config,
    unwrap_model,
)
from power_utils import sd15_alpha_bars
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, MEAN_REAL_COMPONENT_POWER, NOISE_REAL_COMPONENT_VARIANCE,
    RAYLEIGH_CHANNEL_MODEL,
)
from segmentation_transport import (
    TRANSPORT_VERSION, SegmentationLayout, build_segmentation_allocator,
    prepare_segmentation_transmissions, validate_segmentation_config,
)


NUM_CLASSES = CITYSCAPES_NUM_CLASSES
IGNORE_INDEX = CITYSCAPES_IGNORE_INDEX
TASK_MODEL_NAME = "SegFormer-B0"
VAE_DOWNSAMPLE_FACTOR = 8
PATCH_HEIGHT = 8
PATCH_WIDTH = 8
LATENT_HEIGHT = CITYSCAPES_INPUT_HEIGHT // VAE_DOWNSAMPLE_FACTOR
LATENT_WIDTH = CITYSCAPES_INPUT_WIDTH // VAE_DOWNSAMPLE_FACTOR
GROUP_ROWS = LATENT_HEIGHT // PATCH_HEIGHT
GROUP_COLS = LATENT_WIDTH // PATCH_WIDTH
NUM_GROUPS = GROUP_ROWS * GROUP_COLS
PILOTS_PER_GROUP = 0
ALLOCATOR_SNR_SCALE_DB = 10.0


def sample_uniform_snr_db(
    batch_size: int,
    device: torch.device,
    minimum: float,
    maximum: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if minimum == maximum:
        return torch.full((batch_size,), minimum, device=device)
    return minimum + (maximum - minimum) * torch.rand(
        batch_size,
        device=device,
        generator=generator,
    )


class AnalogAWGNChannel(QuantizedAWGNChannel):
    """Analog-only view of the existing power-controlled AWGN channel."""

    def __init__(
        self,
        group_rows: int,
        group_cols: int,
        default_snr_db: float,
        channel_gain_abs: torch.Tensor | list[float] | float,
        pilots_per_group: int,
    ) -> None:
        super().__init__(
            group_rows,
            group_cols,
            default_snr_db=default_snr_db,
            channel_gain_abs=channel_gain_abs,
            pilots_per_group=pilots_per_group,
            transmission_mode="analog",
        )

    def prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def allocate_power_with_timestep_floor(
    z0: torch.Tensor,
    snr_db: torch.Tensor,
    allocator: nn.Module,
    channel: AnalogAWGNChannel,
    temperature: float,
    max_timestep: int,
    use_pilot_uncertainty: bool | None = None,
    allocator_snr_scale_db: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not 0 <= max_timestep < 1000:
        raise ValueError("max_timestep must be between 0 and 999")
    if allocator_snr_scale_db <= 0.0:
        raise ValueError("allocator_snr_scale_db must be positive")

    _, allocator_power, allocation_info = allocate_power(
        z0,
        snr_db / allocator_snr_scale_db,
        allocator,
        channel,
        temperature,
    )
    alpha_bar = sd15_alpha_bars().to(
        device=z0.device,
        dtype=z0.dtype,
    )[max_timestep]
    maximum_noise_variance = (1.0 - alpha_bar) / alpha_bar
    channel_gains = channel._channel_gains(
        allocator_power,
        None,
    ).abs().clamp_min(1e-12)
    noise_spectral_density = channel._noise_spectral_density(z0, snr_db)
    oracle_minimum_group_power = (
        channel.kappa_m_squared
        * noise_spectral_density[:, None]
        / (
            2.0
            * channel_gains.square()
            * maximum_noise_variance
        )
    )

    payload_energy = allocation_info["payload_energy"]
    num_groups = payload_energy.shape[1]
    payload_symbols_per_group = z0[0].numel() / (2 * num_groups)
    payload_power = payload_energy / payload_symbols_per_group
    if use_pilot_uncertainty is None:
        use_pilot_uncertainty = channel.pilots_per_group > 0
    if use_pilot_uncertainty:
        if channel.pilots_per_group == 0:
            raise ValueError(
                "pilot uncertainty requires at least one pilot"
            )
        pilot_uncertainty_factor = (
            1.0 + payload_power / channel.pilots_per_group
        )
    else:
        pilot_uncertainty_factor = torch.ones_like(payload_power)
    minimum_group_power = oracle_minimum_group_power
    if use_pilot_uncertainty:
        minimum_group_power = (
            minimum_group_power * pilot_uncertainty_factor
        )

    unscaled_block_energy = allocation_info["unscaled_block_energy"]
    minimum_frame_energy = (
        minimum_group_power * unscaled_block_energy
    ).sum(dim=1)
    target_frame_energy = float(allocation_info["frame_channel_uses"])
    remaining_frame_energy = target_frame_energy - minimum_frame_energy
    if (remaining_frame_energy < 0.0).any():
        required = minimum_frame_energy.max().item()
        raise ValueError(
            f"max_timestep={max_timestep} requires frame energy "
            f"{required:.6f}, exceeding the budget "
            f"{target_frame_energy:.6f}"
        )

    group_power = minimum_group_power + allocator_power * (
        remaining_frame_energy[:, None] / target_frame_energy
    )
    block_transmit_energy = group_power * unscaled_block_energy
    frame_energy = block_transmit_energy.sum(dim=1)
    allocation_ratio = block_transmit_energy / target_frame_energy
    allocation_info = dict(allocation_info)
    allocation_info.update(
        {
            "frame_energy": frame_energy,
            "mean_symbol_power": frame_energy / target_frame_energy,
            "oracle_minimum_group_power": oracle_minimum_group_power,
            "minimum_group_power": minimum_group_power,
            "payload_power": payload_power,
            "pilot_uncertainty_factor": pilot_uncertainty_factor,
            "uses_pilot_uncertainty": use_pilot_uncertainty,
            "minimum_frame_energy": minimum_frame_energy,
            "remaining_frame_energy": remaining_frame_energy,
            "maximum_noise_variance": maximum_noise_variance,
            "max_timestep": max_timestep,
        }
    )
    return allocation_ratio, group_power, allocation_info


class LatentSegFormerB0(nn.Module):
    """SegFormer-B0 with its RGB projection replaced for VAE latents."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.model = construct_model(TASK_MODEL_NAME)
        patch_embedding = self.model.backbone.layers[0][0]
        original_projection = patch_embedding.projection
        # The VAE latent is already 8x smaller than the RGB image, so avoid
        # SegFormer's additional 4x downsampling at the first projection.
        patch_embedding.projection = nn.Conv2d(
            input_channels,
            original_projection.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=original_projection.bias is not None,
        )

    def forward(
        self,
        received_latent: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        features = self.model.extract_feat(received_latent)
        logits = self.model.decode_head.forward(features)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=self.model.align_corners,
        )


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    predictions = logits.argmax(dim=1)
    valid = labels.ne(IGNORE_INDEX)
    encoded = labels[valid] * NUM_CLASSES + predictions[valid]
    return torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def segmentation_metrics(matrix: torch.Tensor) -> tuple[float, float]:
    matrix = matrix.to(dtype=torch.float64)
    true_positive = matrix.diag()
    total = matrix.sum()
    pixel_accuracy = true_positive.sum() / total.clamp_min(1.0)
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - true_positive
    valid_classes = union > 0
    mean_iou = (
        (true_positive[valid_classes] / union[valid_classes]).mean()
        if valid_classes.any()
        else matrix.new_tensor(0.0)
    )
    return pixel_accuracy.item(), mean_iou.item()


def segmentation_step(
    images: torch.Tensor,
    labels: torch.Tensor,
    vae: AutoencoderKL,
    allocator: nn.Module,
    task_model: nn.Module,
    channel: AnalogAWGNChannel | RayleighFadingChannel,
    transport: ActualPowerTransport | None,
    snr_db: torch.Tensor,
    temperature: float,
    max_timestep: int,
    generator: torch.Generator | None = None,
) -> dict:
    z0 = encode_latents(images, vae, generator)
    if z0.shape[-2:] != (LATENT_HEIGHT, LATENT_WIDTH):
        raise ValueError(
            "Expected VAE latent spatial size "
            f"{LATENT_HEIGHT}x{LATENT_WIDTH}, got "
            f"{z0.shape[-2]}x{z0.shape[-1]}"
        )
    module = unwrap_model(allocator)
    if module.orthogonal is not None:
        if not isinstance(channel, RayleighFadingChannel):
            raise ValueError("Q segmentation training requires the Rayleigh channel")
        transmission = prepare_segmentation_transmissions(
            z0, snr_db, allocator, channel, generator, temperature,
            max_timestep=max_timestep, strategies=("allocated",),
        )
        received = transmission["strategies"]["allocated"]
        logits = task_model(received["direct_latent"], labels.shape[-2:])
        loss = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX)
        allocation = transmission["allocation_info"]
        return {
            "loss": loss, "task_loss": loss.detach(),
            "confusion_matrix": confusion_matrix(logits, labels).detach(),
            "mean_symbol_power": (received["frame_energy"] / allocation["frame_channel_uses"]).detach(),
            "mean_real_component_power": (received["frame_energy"] / z0[0].numel()).detach(),
            "frame_energy": received["frame_energy"].detach(),
            "allocation_ratio": received["energy_allocation_ratio"].detach(),
            "group_power": received["group_power"].detach(),
            "snr": received["channel_info"]["snr"].detach(),
            "channel_coefficient": transmission["h"].detach(),
            "gain_floor_fallback": allocation["gain_floor_fallback"].detach(),
        }
    allocation_ratio, group_power, allocation_info = (
        allocate_power_with_timestep_floor(
            z0,
            snr_db,
            allocator,
            channel,
            temperature,
            max_timestep,
            False,
            ALLOCATOR_SNR_SCALE_DB,
        )
    )
    received_latent, channel_info, _ = transport(
        z0,
        group_power,
        snr_db,
        generator,
    )
    logits = task_model(received_latent, labels.shape[-2:])
    task_loss = F.cross_entropy(
        logits.float(),
        labels,
        ignore_index=IGNORE_INDEX,
    )
    return {
        "loss": task_loss,
        "task_loss": task_loss.detach(),
        "confusion_matrix": confusion_matrix(logits, labels).detach(),
        "mean_symbol_power": allocation_info[
            "mean_symbol_power"
        ].detach(),
        "frame_energy": allocation_info["frame_energy"].detach(),
        "allocation_ratio": allocation_ratio.detach(),
        "group_power": group_power.detach(),
        "snr": channel_info["snr"].detach(),
    }


def run_training() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Cityscapes Q mixing and power allocation over Rayleigh fading "
            "with a latent-input SegFormer-B0; DDRM is used at inference/evaluation"
        )
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--data_path", default="/data/bsy/")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--allocator_lr", type=float, default=None)
    parser.add_argument("--save_path", default="latest.pth")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patch_height", type=int, default=8, help="Latent patch height (default: 8)")
    parser.add_argument("--patch_width", type=int, default=8, help="Latent patch width (default: 8)")
    parser.add_argument("--legacy_awgn", action="store_true",
                        help="Use the original CNN/AWGN segmentation training path")
    parser.add_argument("--segmenter_checkpoint", default=None,
                        help="Warm-start only SegFormer from an existing segmentation checkpoint")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_timestep", type=int, default=999)
    parser.add_argument("--snr_db_min", type=float, default=-10.0)
    parser.add_argument("--snr_db_max", type=float, default=10.0)
    parser.add_argument(
        "--channel_gain_abs",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.batch <= 0 or args.num_workers < 0 or args.epochs <= 0 or args.log_interval <= 0:
        raise ValueError("batch, epochs and log_interval must be positive; num_workers must be non-negative")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("--snr_db_min must not exceed --snr_db_max")
    if not 0 <= args.max_timestep <= 999:
        raise ValueError("--max_timestep must be between 0 and 999")
    if args.resume is not None and args.segmenter_checkpoint is not None:
        raise ValueError("Use --resume or --segmenter_checkpoint, not both")
    layout = SegmentationLayout(args.patch_height, args.patch_width)
    num_groups = NUM_GROUPS if args.legacy_awgn else layout.num_groups
    if args.legacy_awgn and (args.patch_height, args.patch_width) != (PATCH_HEIGHT, PATCH_WIDTH):
        raise ValueError("Legacy AWGN checkpoints require the original 8x8 latent patches")
    if not args.legacy_awgn and args.channel_gain_abs != [1.0]:
        raise ValueError("Rayleigh samples one complex h per image; remove --channel_gain_abs")
    if len(args.channel_gain_abs) not in (1, num_groups):
        raise ValueError(
            f"--channel_gain_abs requires one value or {num_groups} values"
        )

    device, distributed, rank, world_size = initialize_distributed(
        args.device,
        args.local_rank,
    )
    is_main_process = rank == 0
    target_name = "snr_uniform"
    result_dir = (
        Path("checkpoints_power")
        / "Cityscapes"
        / TASK_MODEL_NAME
        / target_name
        / args.exp_name
    )
    if is_main_process:
        result_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = build_logger(result_dir / "train.log", is_main_process)
    checkpoint_path = result_dir / Path(args.save_path).name

    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(process_seed)

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        subfolder="vae",
        revision=VAE_MODEL_REVISION,
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    train_loader, val_loader = construct_cityscapes_dataloaders(
        args.data_path,
        args.batch,
        distributed,
        args.seed,
        args.num_workers,
    )

    latent_channels = int(vae.config.latent_channels)
    if latent_channels != layout.latent_channels:
        raise ValueError("The selected VAE must have four latent channels")
    if args.legacy_awgn:
        allocator = PowerAllocator(latent_channels, NUM_GROUPS, args.hidden_dim, sampling=False).to(device)
        channel = AnalogAWGNChannel(allocator.group_rows, allocator.group_cols,
                                   .5 * (args.snr_db_min + args.snr_db_max),
                                   args.channel_gain_abs, PILOTS_PER_GROUP).to(device)
        transport = ActualPowerTransport(channel, "oracle")
    else:
        allocator = build_segmentation_allocator(layout, args.hidden_dim).to(device)
        channel = RayleighFadingChannel(layout.rows, layout.cols).to(device)
        transport = None
    task_model = LatentSegFormerB0(latent_channels).to(device)
    if args.segmenter_checkpoint is not None:
        warm_start = torch.load(args.segmenter_checkpoint, map_location="cpu", weights_only=False)
        if warm_start.get("config", {}).get("model") != TASK_MODEL_NAME:
            raise ValueError("--segmenter_checkpoint must contain a latent-input SegFormer-B0")
        task_model.load_state_dict(warm_start["segmenter"], strict=True)

    start_epoch = 0
    global_step = 0
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        checkpoint_config = checkpoint["config"]
        if checkpoint_config["model"] != TASK_MODEL_NAME:
            raise ValueError(
                "Resume checkpoint was not trained with SegFormer-B0"
            )
        if checkpoint_config.get("transmission_mode") != "analog":
            raise ValueError(
                "Only an analog-channel checkpoint can be resumed"
            )
        if not args.legacy_awgn:
            if validate_segmentation_config(checkpoint_config) != layout:
                raise ValueError("Resume patch geometry differs; match --patch_height/--patch_width to the checkpoint")
        elif checkpoint_config.get("orthogonal_transform", False):
            raise ValueError("A Q/Rayleigh checkpoint cannot resume with --legacy_awgn")
        if checkpoint_config.get("num_groups") != num_groups:
            raise ValueError(
                f"Resume checkpoint must use {num_groups} power groups"
            )
        if int(checkpoint_config.get("pilots_per_group", 0)) != 0:
            raise ValueError(
                "Resume checkpoint includes pilot symbols; "
                "start a new pilot-free experiment"
            )
        if checkpoint_config.get(
            "training_uses_pilot_estimation",
            False,
        ):
            raise ValueError(
                "Resume checkpoint was trained with pilot estimation; "
                "start a new experiment for oracle training"
            )
        if args.legacy_awgn and float(
            checkpoint_config.get("allocator_snr_scale_db", 1.0)
        ) != ALLOCATOR_SNR_SCALE_DB:
            raise ValueError(
                "Resume checkpoint used incompatible allocator SNR "
                "conditioning; start a new experiment"
            )
        allocator.load_state_dict(checkpoint["allocator"])
        task_model.load_state_dict(checkpoint["segmenter"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    if distributed:
        ddp_options = {}
        if device.type == "cuda":
            ddp_options = {
                "device_ids": [args.local_rank],
                "output_device": args.local_rank,
            }
        allocator = DistributedDataParallel(allocator, **ddp_options)
        task_model = DistributedDataParallel(task_model, **ddp_options)

    allocator_parameters = list(allocator.parameters())
    task_parameters = list(task_model.parameters())
    parameters = allocator_parameters + task_parameters
    allocator_lr = args.lr if args.allocator_lr is None else args.allocator_lr
    optimizer = optim.AdamW(
        [
            {"params": allocator_parameters, "lr": allocator_lr},
            {"params": task_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp_enabled and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

    config = vars(args).copy()
    config.update(
        {
            "stage": "segmentation_power_allocator",
            "dataset": "Cityscapes",
            "model": TASK_MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "task_model_frozen": False,
            "task_model_input": "received_vae_latent",
            "task_model_input_channels": latent_channels,
            "task_loss": "pixel_cross_entropy",
            "transmission_mode": "analog",
            "receiver_mode": "oracle",
            "channel_estimation": "oracle_gain",
            "training_uses_pilot_estimation": False,
            "pilots_per_group": PILOTS_PER_GROUP,
            "pilot_symbols_in_frame": 0,
            "allocator_snr_conditioning": "snr_db_divided_by_scale",
            "allocator_snr_scale_db": ALLOCATOR_SNR_SCALE_DB,
            "allocator_snr_input_range": [-1.0, 1.0],
            "snr_sampling": "uniform_continuous",
            "snr_db": [args.snr_db_min, args.snr_db_max],
            "latent_shape": [LATENT_HEIGHT, LATENT_WIDTH],
            "patch_shape": [PATCH_HEIGHT, PATCH_WIDTH],
            "group_grid": [GROUP_ROWS, GROUP_COLS],
            "num_groups": NUM_GROUPS,
            "vae_model_id": VAE_MODEL_ID,
            "vae_model_revision": VAE_MODEL_REVISION,
            "vae_scale": VAE_SCALE,
            "checkpoint_path": str(checkpoint_path),
            "distributed": distributed,
            "world_size": world_size,
            "global_batch_size": args.batch * world_size,
            "input_shape": [
                CITYSCAPES_INPUT_HEIGHT,
                CITYSCAPES_INPUT_WIDTH,
            ],
            "power_definition": (
                "sum_g a_g^2 ||u_g||^2 = N_payload"
            ),
        }
    )
    if not args.legacy_awgn:
        config.update(layout.config())
        config.update(
            segmentation_transport_version=TRANSPORT_VERSION, orthogonal_transform=True,
            channel_model=RAYLEIGH_CHANNEL_MODEL, allocator_architecture=ALLOCATOR_ARCHITECTURE,
            noise_real_component_variance=NOISE_REAL_COMPONENT_VARIANCE,
            mean_real_component_power=MEAN_REAL_COMPONENT_POWER,
            channel_gain_abs=None, perfect_csi=True,
            fading_granularity="one_complex_h_per_image_shared_by_all_blocks",
            allocator_snr_conditioning="log_gamma=ln(abs(h)^2*P/N0)",
            task_model_input="Q_applied_to_oracle_gain_corrected_received_latent",
            training_uses_diffusion=False, restoration_method="ddrm",
            power_definition="sum_b a_b^2 ||U_b||^2 = N_real * P, P=1",
        )
        config.pop("allocator_snr_scale_db", None)
        config.pop("allocator_snr_input_range", None)
    if is_main_process:
        save_config(result_dir / "config.json", config)
    logger.info(
        "experiment=%s model=%s result_dir=%s device=%s world_size=%d",
        args.exp_name,
        TASK_MODEL_NAME,
        result_dir,
        device,
        world_size,
    )
    logger.info(
        "task=Cityscapes-19 loss=pixel_ce task_model_frozen=false "
        "task_input=received_vae_latent "
        "transmission_mode=analog receiver_mode=oracle "
        "channel_estimation=oracle_gain pilots_per_group=0 "
        "allocator_snr_conditioning=%s "
        "snr_db_uniform=[%.2f, %.2f] "
        "groups=%d",
        f"snr_db/{ALLOCATOR_SNR_SCALE_DB:g}" if args.legacy_awgn else "log_gamma=ln(abs(h)^2*P/N0)",
        args.snr_db_min,
        args.snr_db_max,
        num_groups,
    )
    if not args.legacy_awgn:
        logger.info("Q shape=%dx%d grid=%dx%d latent_patch=%dx%d channel=Rayleigh "
                    "one_complex_h_per_image N0_per_real_component P=1", num_groups, num_groups,
                    layout.rows, layout.cols, layout.patch_height, layout.patch_width)

    for epoch in range(start_epoch, args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        allocator.train()
        task_model.train()
        train_totals = torch.zeros(3, device=device, dtype=torch.float64)
        train_confusion = torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            device=device,
            dtype=torch.int64,
        )
        power_sum = torch.zeros(num_groups, device=device)
        allocation_sum = torch.zeros(num_groups, device=device)

        logger.info(
            "epoch=%d/%d train_start batches=%d",
            epoch + 1,
            args.epochs,
            len(train_loader),
        )
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            snr_db = sample_uniform_snr_db(
                images.shape[0],
                device,
                args.snr_db_min,
                args.snr_db_max,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                result = segmentation_step(
                    images,
                    labels,
                    vae,
                    allocator,
                    task_model,
                    channel,
                    transport,
                    snr_db,
                    args.temperature,
                    args.max_timestep,
                )

            finite_loss = torch.isfinite(result["loss"]).to(torch.int32)
            if distributed:
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if not finite_loss.item():
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            global_step += 1
            train_totals += torch.tensor(
                [
                    result["loss"].item() * batch_size,
                    result["mean_symbol_power"].mean().item() * batch_size,
                    batch_size,
                ],
                device=device,
                dtype=torch.float64,
            )
            train_confusion += result["confusion_matrix"]
            power_sum += result["group_power"].sum(dim=0)
            allocation_sum += result["allocation_ratio"].sum(dim=0)

            if batch_index == 0 or (batch_index + 1) % args.log_interval == 0:
                batch_pixel_acc, _ = segmentation_metrics(
                    result["confusion_matrix"]
                )
                logger.info(
                    "epoch=%d/%d batch=%d/%d loss=%.4f pixel_acc=%.4f "
                    "frame_snr_db=%.2f N0=%.6f mean_symbol_power=%.6f "
                    "frame_energy=%.2f gain_power_min=%.6f "
                    "gain_power_max=%.6f latent_snr_db=%.2f",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    result["loss"].item(),
                    batch_pixel_acc,
                    snr_db.mean().item(),
                    torch.pow(10.0, -snr_db / 10.0).mean().item(),
                    result["mean_symbol_power"].mean().item(),
                    result["frame_energy"].mean().item(),
                    result["group_power"].min().item(),
                    result["group_power"].max().item(),
                    (10.0 * torch.log10(result["snr"])).mean().item(),
                )
                if not args.legacy_awgn:
                    logger.info("mean_abs_h=%.6f mean_real_component_power=%.6f gain_floor_infeasible_images=%d",
                                result["channel_coefficient"].abs().mean().item(),
                                result["mean_real_component_power"].mean().item(),
                                result["gain_floor_fallback"].sum().item())

        if distributed:
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_confusion, op=dist.ReduceOp.SUM)
            dist.all_reduce(power_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(allocation_sum, op=dist.ReduceOp.SUM)
        train_samples = int(train_totals[-1].item())
        train_pixel_acc, train_miou = segmentation_metrics(train_confusion)

        allocator.eval()
        task_model.eval()
        validation_totals = torch.zeros(
            4,
            device=device,
            dtype=torch.float64,
        )
        validation_confusion = torch.zeros_like(train_confusion)
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(process_seed)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                snr_db = sample_uniform_snr_db(
                    images.shape[0],
                    device,
                    args.snr_db_min,
                    args.snr_db_max,
                    validation_generator,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    result = segmentation_step(
                        images,
                        labels,
                        vae,
                        allocator,
                        task_model,
                        channel,
                        transport,
                        snr_db,
                        args.temperature,
                        args.max_timestep,
                        validation_generator,
                    )
                batch_size = images.shape[0]
                validation_totals += torch.tensor(
                    [
                        result["loss"].item() * batch_size,
                        result["mean_symbol_power"].mean().item()
                        * batch_size,
                        (10.0 * torch.log10(result["snr"])).mean().item()
                        * batch_size,
                        batch_size,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                validation_confusion += result["confusion_matrix"]

        if distributed:
            dist.all_reduce(validation_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(validation_confusion, op=dist.ReduceOp.SUM)
        validation_samples = int(validation_totals[-1].item())
        scheduler.step()

        train_mean = train_totals[:2] / train_samples
        validation_mean = validation_totals[:3] / validation_samples
        val_pixel_acc, val_miou = segmentation_metrics(
            validation_confusion
        )
        logger.info(
            "epoch=%d/%d train_loss=%.4f train_pixel_acc=%.4f "
            "train_miou=%.4f train_symbol_power=%.6f val_loss=%.4f "
            "val_pixel_acc=%.4f val_miou=%.4f val_symbol_power=%.6f "
            "val_latent_snr_db=%.2f",
            epoch + 1,
            args.epochs,
            train_mean[0].item(),
            train_pixel_acc,
            train_miou,
            train_mean[1].item(),
            validation_mean[0].item(),
            val_pixel_acc,
            val_miou,
            validation_mean[1].item(),
            validation_mean[2].item(),
        )
        logger.info(
            "epoch=%d mean_gain_power_min=%.6f "
            "mean_gain_power_max=%.6f allocation_min=%.6f "
            "allocation_max=%.6f",
            epoch + 1,
            (power_sum / train_samples).min().item(),
            (power_sum / train_samples).max().item(),
            (allocation_sum / train_samples).min().item(),
            (allocation_sum / train_samples).max().item(),
        )

        if is_main_process:
            torch.save(
                {
                    "allocator": unwrap_model(allocator).state_dict(),
                    "segmenter": unwrap_model(task_model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": config,
                    "metrics": {
                        "train_pixel_accuracy": train_pixel_acc,
                        "train_miou": train_miou,
                        "val_pixel_accuracy": val_pixel_acc,
                        "val_miou": val_miou,
                    },
                },
                checkpoint_path,
            )
            logger.info("checkpoint_saved path=%s", checkpoint_path)
        if distributed:
            dist.barrier()


def main() -> None:
    install_termination_handlers()
    try:
        run_training()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
