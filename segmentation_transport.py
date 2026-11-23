"""Q-mixed Cityscapes transport with paired channel and receiver draws."""

from dataclasses import dataclass
from pathlib import Path

import torch

from datasets.data_processing import CITYSCAPES_INPUT_HEIGHT, CITYSCAPES_INPUT_WIDTH
from inference_oracle_gain import oracle_receive
from inference_power import allocate_power_for_inference
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, MEAN_REAL_COMPONENT_POWER, NOISE_REAL_COMPONENT_VARIANCE,
    RAYLEIGH_CHANNEL_MODEL, OrthogonalPatchTransform, gain_estimator_receive,
    patchify, prepare_gain_estimator_input, sample_raw_channel,
)
from train_power_allocator import PowerAllocator, RayleighFadingChannel, restore_spatial_latent


TRANSPORT_VERSION = 1
DEFAULT_PATCH_HEIGHT = 8
DEFAULT_PATCH_WIDTH = 8


@dataclass(frozen=True)
class SegmentationLayout:
    patch_height: int = DEFAULT_PATCH_HEIGHT
    patch_width: int = DEFAULT_PATCH_WIDTH
    latent_height: int = CITYSCAPES_INPUT_HEIGHT // 8
    latent_width: int = CITYSCAPES_INPUT_WIDTH // 8
    latent_channels: int = 4

    def __post_init__(self):
        values = (self.patch_height, self.patch_width, self.latent_height,
                  self.latent_width, self.latent_channels)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Patch and latent dimensions must be positive integers")
        if self.latent_height % self.patch_height or self.latent_width % self.patch_width:
            raise ValueError("Patch dimensions must divide the VAE latent dimensions")
        if self.block_dim % 2:
            raise ValueError("Each transport block must contain an even number of real I/Q values")

    @property
    def rows(self):
        return self.latent_height // self.patch_height

    @property
    def cols(self):
        return self.latent_width // self.patch_width

    @property
    def num_groups(self):
        return self.rows * self.cols

    @property
    def block_dim(self):
        return self.latent_channels * self.patch_height * self.patch_width

    def check_latent(self, latent):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (
            self.latent_channels, self.latent_height, self.latent_width
        ):
            raise ValueError("VAE latent does not match the checkpoint's channels, spatial size, and patches")

    def config(self):
        return {
            "latent_channels": self.latent_channels,
            "latent_shape": [self.latent_height, self.latent_width],
            "patch_shape": [self.patch_height, self.patch_width],
            "group_grid": [self.rows, self.cols], "num_groups": self.num_groups,
            "allocator_block_dim": self.block_dim,
            "input_shape": [self.latent_height * 8, self.latent_width * 8],
        }

    @classmethod
    def from_config(cls, config):
        patch = config["patch_shape"]
        spatial = config["latent_shape"]
        layout = cls(int(patch[0]), int(patch[1]), int(spatial[0]), int(spatial[1]),
                     int(config["latent_channels"]))
        for key, expected in layout.config().items():
            if config.get(key) != expected:
                raise ValueError(f"Inconsistent segmentation checkpoint layout: {key}")
        return layout


def validate_segmentation_config(config):
    if (config.get("dataset") != "Cityscapes"
            or config.get("stage") != "segmentation_power_allocator"
            or config.get("segmentation_transport_version") != TRANSPORT_VERSION
            or not config.get("orthogonal_transform")
            or config.get("allocator_architecture") != ALLOCATOR_ARCHITECTURE
            or config.get("channel_model") != RAYLEIGH_CHANNEL_MODEL
            or config.get("noise_real_component_variance") != NOISE_REAL_COMPONENT_VARIANCE
            or config.get("mean_real_component_power") != MEAN_REAL_COMPONENT_POWER
            or config.get("transmission_mode") != "analog"
            or config.get("receiver_mode") != "oracle"
            or config.get("pilots_per_group") != 0):
        raise ValueError("Expected a Cityscapes Q/Rayleigh checkpoint with oracle gain, N0 noise, and P=1. "
                         "Use --legacy_awgn for an old AWGN checkpoint; Q training requires a new experiment.")
    layout = SegmentationLayout.from_config(config)
    if config["input_shape"] != [CITYSCAPES_INPUT_HEIGHT, CITYSCAPES_INPUT_WIDTH]:
        raise ValueError("Cityscapes input must remain 1024x2048")
    return layout


def build_segmentation_allocator(layout, hidden_dim=64):
    return PowerAllocator(layout.latent_channels, layout.num_groups, hidden_dim,
                          orthogonal_transform=True, block_dim=layout.block_dim,
                          group_grid=(layout.rows, layout.cols))


def load_segmentation_transmitter(path, device, *, checkpoint=None):
    path = Path(path).expanduser()
    if path.is_dir():
        path /= "latest.pth"
    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    layout = validate_segmentation_config(config)
    allocator = build_segmentation_allocator(layout, int(config["hidden_dim"])).to(device)
    allocator.load_state_dict(checkpoint["allocator"], strict=True)
    allocator.eval().requires_grad_(False)
    channel = RayleighFadingChannel(layout.rows, layout.cols).to(device)
    return allocator, channel, layout, checkpoint, path


def prepare_segmentation_transmissions(latent, snr_db, allocator, channel, generator=None,
                                      temperature=1.0, max_timestep=999, min_gain=0.0,
                                      strategies=("uniform", "allocated"), *,
                                      channel_model="rayleigh", receiver_mode="oracle",
                                      uniform_receiver_mode="oracle", gain_estimator=None):
    """Pair both transmitters using one h per image and common real AWGN.

    The learned path uses Q; the uniform baseline transmits spatial latents.
    Gradients through allocation, Q^T transmission and Q reception are retained.
    """
    module = getattr(allocator, "module", allocator)
    rows, cols = module.group_rows, module.group_cols
    if (channel.group_rows, channel.group_cols) != (rows, cols):
        raise ValueError("Channel and allocator block grids must match")
    if not strategies or any(name not in ("uniform", "allocated") for name in strategies):
        raise ValueError("Transport strategies must be uniform and/or allocated")
    if channel_model not in ("rayleigh", "awgn"):
        raise ValueError("channel_model must be rayleigh or awgn")
    receiver_modes = {"allocated": receiver_mode, "uniform": uniform_receiver_mode}
    if any(mode not in ("oracle", "gain_estimator") for mode in receiver_modes.values()):
        raise ValueError("Receiver modes must be oracle or gain_estimator")
    if any(receiver_modes[name] == "gain_estimator" for name in strategies) and gain_estimator is None:
        raise ValueError("Gain estimator reception requires a loaded estimator")
    h = channel.sample_channel_coefficient(latent, generator)
    # Consume the same h draws in either mode, preserving the paired AWGN stream.
    if channel_model == "awgn":
        h = torch.ones_like(h)
    ratio, allocated_power, info = allocate_power_for_inference(
        latent, snr_db, allocator, channel, temperature,
        channel_coefficient=h, max_timestep=max_timestep, min_gain=min_gain,
    )
    q = info["orthogonal_matrix"]
    if q is None:
        raise ValueError("Segmentation Rayleigh transmission requires an orthogonal allocator")
    energy = patchify(latent, rows, cols).square().sum(-1)
    budget = info["target_frame_energy"]
    uniform_power = (budget / energy.sum(1, keepdim=True).clamp_min(1e-12)).expand_as(allocated_power)
    n0 = channel._noise_spectral_density(latent, snr_db)
    epsilon = torch.randn(latent.shape, device=latent.device, dtype=torch.float32, generator=generator)
    identity = torch.eye(rows * cols, device=latent.device, dtype=torch.float32)
    result = {}
    for name in strategies:
        uses_q = name == "allocated"
        signal = info["channel_input"] if uses_q else latent
        matrix = q if uses_q else identity
        power = allocated_power if uses_q else uniform_power
        block_energy = info["unscaled_block_energy"] if uses_q else energy
        raw = sample_raw_channel(signal, power, n0, h, rows=rows, cols=cols, epsilon=epsilon)
        if receiver_modes[name] == "gain_estimator":
            estimator_input = prepare_gain_estimator_input(raw, n0, h, rows=rows, cols=cols)
            log_gain = gain_estimator(estimator_input["channel_equalized_latent"],
                                      estimator_input["log_noise_variance"], estimator_input["log_power"])
            received = gain_estimator_receive(raw, log_gain, n0, h, rows=rows, cols=cols)
            received["observation"] = received["equalized_latent"]
        else:
            received = oracle_receive(raw, power, n0, h, rows, cols)
        spatial, channel_info = restore_spatial_latent(
            received["observation"],
            {"orthogonal_matrix": matrix, "group_grid": (rows, cols), "source_latent": latent},
            {"noise_variance": received["noise_variance"]},
        )
        result[name] = {
            **received, "Q": matrix, "uses_Q": uses_q, "raw_received": raw,
            "receiver_mode": receiver_modes[name], "true_amplitude_gain": power.sqrt(),
            "transmitted_latent": signal, "direct_latent": spatial,
            "group_power": power, "frame_energy": (power * block_energy).sum(1),
            "energy_allocation_ratio": ratio if uses_q else power * block_energy / budget,
            "channel_info": channel_info,
        }
    return {"strategies": result, "h": h, "N0": n0, "epsilon": epsilon,
            "layout": (rows, cols), "allocation_info": info}


def restore_segmentation_transmissions(transmission, sampler, generator, verbose=False):
    """Accept one shared sampler or a strategy-to-sampler mapping."""
    rows, cols = transmission["layout"]
    initial_state = generator.get_state()
    restored, diagnostics = {}, {}
    for name, received in transmission["strategies"].items():
        strategy_sampler = sampler[name] if isinstance(sampler, dict) else sampler
        generator.set_state(initial_state)
        try:
            restored[name] = strategy_sampler(received["observation"], received["precision"], received["Q"],
                                             generator, rows=rows, cols=cols, verbose=verbose)
        except ValueError as error:
            policy = getattr(strategy_sampler, "initialization_policy", None)
            raise ValueError(f"Restoration failed: strategy={name}, receiver_mode={received['receiver_mode']}, "
                             f"initialization_policy={policy}: {error}") from error
        diagnostics[name] = strategy_sampler.last_diagnostics
    return restored, diagnostics
