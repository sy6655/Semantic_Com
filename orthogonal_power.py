"""Block-space orthogonal transmission and pilot-free gain estimation."""

import hashlib

import torch
import torch.nn as nn


RAYLEIGH_CHANNEL_MODEL = "rayleigh_block_fading"
ALLOCATOR_ARCHITECTURE = "block_mlp_v4"
GAIN_ESTIMATOR_ARCHITECTURE = "block_conv1d_global_mlp_v4"
GAIN_ESTIMATOR_VERSION = 5
NOISE_REAL_COMPONENT_VARIANCE = "N0"
MEAN_REAL_COMPONENT_POWER = 1.0
GAIN_STATISTIC_NAMES = (
    "mean", "log_second_moment", "mean_absolute_value",
    "log_normalized_fourth_moment", "normalized_adjacent_correlation", "log_variance",
)


def sample_rayleigh_fading(batch_size, device, generator=None):
    """One h~CN(0,1) per image, shared by every transmitted block."""
    real = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    imaginary = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    return (real + 1j * imaginary) / (2 ** 0.5)


def common_channel_magnitude(channel_gain, reference):
    if channel_gain.shape != (reference.shape[0],) or not channel_gain.is_complex():
        raise ValueError("Perfect CSI h must be a complex tensor of shape [B], shared by all blocks")
    magnitude = channel_gain.abs().float()
    if not torch.isfinite(magnitude).all() or (magnitude == 0).any():
        raise ValueError("Channel coefficients must be finite and nonzero")
    return magnitude


def patchify(latent: torch.Tensor, rows: int = 8, cols: int = 8) -> torch.Tensor:
    """P_p: [B,C,H,W] -> [B,rows*cols,C*patch_h*patch_w], row major."""
    if latent.ndim != 4:
        raise ValueError("latent must have shape [B,C,H,W]")
    batch, channels, height, width = latent.shape
    if rows <= 0 or cols <= 0 or height % rows or width % cols:
        raise ValueError("Latent dimensions must be divisible by the block grid")
    return latent.reshape(
        batch, channels, rows, height // rows, cols, width // cols
    ).permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, -1)


def unpatchify(
    blocks: torch.Tensor, shape: tuple[int, ...], rows: int = 8, cols: int = 8
) -> torch.Tensor:
    batch, channels, height, width = shape
    if height % rows or width % cols or blocks.shape != (
        batch, rows * cols, channels * (height // rows) * (width // cols)
    ):
        raise ValueError("Block shape does not match latent shape and grid")
    return blocks.reshape(
        batch, rows, cols, channels, height // rows, width // cols
    ).permute(0, 3, 1, 4, 2, 5).reshape(shape)


def block_map(values: torch.Tensor, reference: torch.Tensor, rows: int, cols: int):
    if values.shape != (reference.shape[0], rows * cols):
        raise ValueError("Block values must have shape [B,rows*cols]")
    if reference.shape[-2] % rows or reference.shape[-1] % cols:
        raise ValueError("Latent dimensions must be divisible by the block grid")
    return values.reshape(-1, 1, rows, cols).repeat_interleave(
        reference.shape[-2] // rows, dim=2
    ).repeat_interleave(reference.shape[-1] // cols, dim=3)


def apply_complex_channel(latent, channel_gain, rows=8, cols=8):
    """Multiply paired real I/Q values by a common complex coefficient h."""
    common_channel_magnitude(channel_gain, latent)
    with torch.autocast(device_type=latent.device.type, enabled=False):
        blocks = patchify(latent.float(), rows, cols)
        if blocks.shape[-1] % 2:
            raise ValueError("Each block must contain an even number of real I/Q values")
        symbols = torch.complex(blocks[..., 0::2], blocks[..., 1::2])
        received = channel_gain[:, None, None] * symbols
        packed = torch.stack((received.real, received.imag), dim=-1).flatten(-2)
        return unpatchify(packed, latent.shape, rows, cols)


class OrthogonalPatchTransform(nn.Module):
    """Q=exp(L-L^T), with only the strict upper triangle trainable."""

    def __init__(self, rows: int = 8, cols: int = 8):
        super().__init__()
        self.rows, self.cols = rows, cols
        count = rows * cols
        indices = torch.triu_indices(count, count, offset=1)
        self.register_buffer("upper_indices", indices, persistent=False)
        self.upper = nn.Parameter(torch.zeros(indices.shape[1]))

    def matrix(self) -> torch.Tensor:
        # matrix_exp and the block rotation stay in float32 under AMP.
        with torch.autocast(device_type=self.upper.device.type, enabled=False):
            count = self.rows * self.cols
            upper = self.upper.new_zeros(count, count, dtype=torch.float32)
            upper[self.upper_indices[0], self.upper_indices[1]] = self.upper.float()
            return torch.linalg.matrix_exp(upper - upper.T)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.matrix()
        return self.apply_matrix(latent, q.T, self.rows, self.cols), q

    @staticmethod
    def apply_matrix(latent, matrix, rows=8, cols=8):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            blocks = patchify(latent.float(), rows, cols)
            rotated = torch.matmul(matrix.float(), blocks)
            return unpatchify(rotated, latent.shape, rows, cols)


class BlockPowerAllocator(nn.Module):
    """Shared block MLP with mean global context and learned block IDs.

    Default: 256->128->64 features, then [64+64+8+1+1]->64->1.
    The allocator has 50,625 parameters, excluding the separate Q transform.
    """

    def __init__(self, block_dim=256, num_blocks=64, feature_dim=64, embedding_dim=8):
        super().__init__()
        self.block_dim, self.num_blocks = block_dim, num_blocks
        self.block_mlp = nn.Sequential(
            nn.Linear(block_dim, 2 * feature_dim), nn.SiLU(),
            nn.Linear(2 * feature_dim, feature_dim), nn.SiLU(),
        )
        self.block_embedding = nn.Embedding(num_blocks, embedding_dim)
        self.register_buffer("block_ids", torch.arange(num_blocks), persistent=False)
        self.logit_head = nn.Sequential(
            nn.Linear(2 * feature_dim + embedding_dim + 2, feature_dim), nn.SiLU(),
            nn.Linear(feature_dim, 1),
        )
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.logit_head[-1].bias)

    def forward(self, blocks, log_gamma):
        if blocks.ndim != 3 or blocks.shape[1:] != (self.num_blocks, self.block_dim):
            raise ValueError(f"Allocator blocks must have shape [B,{self.num_blocks},{self.block_dim}]")
        if log_gamma.shape != (blocks.shape[0],):
            raise ValueError("log_gamma must have shape [B]")
        features = self.block_mlp(blocks)
        context = features.mean(dim=1, keepdim=True).expand(-1, self.num_blocks, -1)
        block_ids = self.block_embedding(self.block_ids)[None].expand(blocks.shape[0], -1, -1)
        with torch.autocast(device_type=blocks.device.type, enabled=False):
            log_energy = blocks.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-8).log()
        log_snr = log_gamma[:, None, None].expand(-1, self.num_blocks, 1)
        head_input = torch.cat((features, context, block_ids, log_energy, log_snr), dim=-1)
        return self.logit_head(head_input).squeeze(-1)


def allocator_fingerprint(allocator: nn.Module) -> str:
    """Bind a gain estimator to the exact frozen transmitter it learned."""
    digest = hashlib.sha256()
    for name, value in sorted(allocator.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gain_block_statistics(blocks, epsilon=1e-8):
    """Section VI-B statistics of channel-corrected real block vectors."""
    if blocks.ndim != 3 or blocks.shape[-1] < 2:
        raise ValueError("Gain statistics require [B,num_blocks,m] with m>=2")
    with torch.autocast(device_type=blocks.device.type, enabled=False):
        values = blocks.float()
        mean = values.mean(dim=-1)
        second = values.square().mean(dim=-1)
        fourth = values.square().square().mean(dim=-1)
        adjacent = (values[..., :-1] * values[..., 1:]).mean(dim=-1)
        safe_second = second.clamp_min(epsilon)
        return torch.stack((
            mean,
            safe_second.log(),
            values.abs().mean(dim=-1),
            (fourth / safe_second.square()).clamp_min(epsilon).log(),
            adjacent / safe_second,
            (second - mean.square()).clamp_min(epsilon).log(),
        ), dim=-1)


def prepare_gain_estimator_input(raw_received, noise_spectral_density, channel_gain,
                                 mean_power=MEAN_REAL_COMPONENT_POWER, rows=8, cols=8):
    """Perfect-CSI receiver input R_h=Y/h, nu=N0/|h|^2, and log P."""
    magnitude = common_channel_magnitude(channel_gain, raw_received)
    if noise_spectral_density.shape != (raw_received.shape[0],) or (noise_spectral_density <= 0).any():
        raise ValueError("N0 must have shape [B] and be positive")
    r_h = apply_complex_channel(raw_received, channel_gain.reciprocal(), rows, cols)
    nu = noise_spectral_density.float() / magnitude.square()
    power = torch.as_tensor(mean_power, device=raw_received.device, dtype=torch.float32)
    if power.ndim == 0:
        power = power.expand(raw_received.shape[0])
    if power.shape != (raw_received.shape[0],) or (power <= 0).any():
        raise ValueError("P must be positive and scalar or shape [B]")
    return {"channel_equalized_latent": r_h, "noise_variance": nu,
            "log_noise_variance": nu.log(), "log_power": power.log()}


class PatchGainEstimator(nn.Module):
    """Block-wise Conv1D features followed by a global log-gain MLP.

    Inputs: R_h=Y/h, log(nu), log(P), without removing the unknown a_b.
    Each block contributes 32 pooled Conv1D features, 6 statistics, and
    an 8-D ID embedding. Default global head: 2946->256->64 (774,080 params).
    A different group_grid builds a separate model with that block count;
    existing 64-block checkpoints keep their original state-dict shapes.
    """

    def __init__(self, latent_channels=4, patch_height=8, patch_width=8, hidden_dim=256,
                 group_grid=(8, 8)):
        super().__init__()
        if len(group_grid) != 2 or any(not isinstance(value, int) or value < 1 for value in group_grid):
            raise ValueError("Estimator group_grid must be a positive (rows,cols) pair")
        self.latent_channels = latent_channels
        self.patch_height, self.patch_width = patch_height, patch_width
        self.block_dim = latent_channels * patch_height * patch_width
        self.group_rows, self.group_cols = group_grid
        self.num_blocks = self.group_rows * self.group_cols
        self.statistic_names = GAIN_STATISTIC_NAMES
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.SiLU(),
        )
        self.block_embedding = nn.Embedding(self.num_blocks, 8)
        self.register_buffer("block_ids", torch.arange(self.num_blocks), persistent=False)
        self.global_mlp = nn.Sequential(nn.Linear(self.num_blocks * 46 + 2, hidden_dim), nn.SiLU())
        self.log_gain_head = nn.Linear(hidden_dim, self.num_blocks)
        nn.init.zeros_(self.log_gain_head.weight)
        nn.init.zeros_(self.log_gain_head.bias)

    def forward(self, channel_equalized, log_noise_variance, log_power):
        if channel_equalized.ndim == 4:
            if (channel_equalized.shape[1] != self.latent_channels
                    or channel_equalized.shape[2] != self.group_rows * self.patch_height
                    or channel_equalized.shape[3] != self.group_cols * self.patch_width):
                raise ValueError("R_h shape must match the estimator's block grid and patch shape")
            blocks = patchify(channel_equalized, self.group_rows, self.group_cols)
        else:
            blocks = channel_equalized
        if blocks.ndim != 3 or blocks.shape[1:] != (self.num_blocks, self.block_dim):
            raise ValueError(f"R_h must contain [B,{self.num_blocks},{self.block_dim}] real values")
        if log_noise_variance.shape != (blocks.shape[0],) or log_power.shape != (blocks.shape[0],):
            raise ValueError("log(nu) and log(P) must each have shape [B]")
        local_features = self.stem(blocks.reshape(-1, 1, self.block_dim)).mean(dim=-1)
        local_features = local_features.reshape(blocks.shape[0], self.num_blocks, 32)
        statistics = gain_block_statistics(blocks)
        embeddings = self.block_embedding(self.block_ids)[None].expand(blocks.shape[0], -1, -1)
        block_features = torch.cat((local_features, statistics, embeddings), dim=-1)
        global_input = torch.cat((block_features.flatten(1),
                                  log_noise_variance[:, None], log_power[:, None]), dim=-1)
        return self.log_gain_head(self.global_mlp(global_input)).float()


def gain_estimator_receive(raw_received, log_gain, noise_spectral_density,
                           channel_gain, rows=8, cols=8):
    """Equalize both h and estimated a, then compute the requested precision.

    Each packed real I/Q component has variance N0, so the equalized
    real-component precision is lambda=|h|^2*a_hat^2/N0.
    """
    magnitude = common_channel_magnitude(channel_gain, raw_received)
    gain = log_gain.clamp(-20, 20).exp()
    effective_gain = channel_gain[:, None] * gain
    # Divide by complex h itself, removing both phase and amplitude.
    channel_equalized = apply_complex_channel(raw_received, channel_gain.reciprocal(), rows, cols)
    equalized = channel_equalized / block_map(gain, raw_received, rows, cols)
    precision = magnitude[:, None].square() * gain.square() / noise_spectral_density[:, None]
    return {"equalized_latent": equalized, "amplitude_gain": gain,
            "effective_gain": effective_gain, "precision": precision,
            "noise_variance": 1.0 / precision}


def sample_raw_channel(transformed, group_power, noise_spectral_density,
                       channel_gain, generator=None, rows=8, cols=8, epsilon=None):
    """Y_b=h*a_b*U_b+n_b, with Var(Re(n))=Var(Im(n))=N0."""
    transmitted = block_map(group_power.sqrt(), transformed, rows, cols) * transformed.float()
    faded = apply_complex_channel(transmitted, channel_gain, rows, cols)
    if epsilon is None:
        epsilon = torch.randn(transformed.shape, device=transformed.device,
                              dtype=torch.float32, generator=generator)
    if epsilon.shape != transformed.shape:
        raise ValueError("epsilon must have the same shape as the transmitted latent")
    return faded + noise_spectral_density.sqrt()[:, None, None, None] * epsilon


def load_gain_estimator(checkpoint_path, allocator, device, *, check_allocator_fingerprint=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config.get("stage") != "block_gain_estimator"
            or config.get("estimator_version") != GAIN_ESTIMATOR_VERSION
            or config.get("channel_model") != RAYLEIGH_CHANNEL_MODEL
            or config.get("noise_real_component_variance") != NOISE_REAL_COMPONENT_VARIANCE
            or config.get("mean_real_component_power") != MEAN_REAL_COMPONENT_POWER
            or config.get("estimator_architecture") != GAIN_ESTIMATOR_ARCHITECTURE):
        raise ValueError("Expected a Rayleigh/perfect-CSI gain estimator with real noise variance N0 and P=1; retrain the previous estimator")
    if check_allocator_fingerprint and config.get("allocator_fingerprint") != allocator_fingerprint(allocator):
        raise ValueError("Gain estimator was trained with a different Q/allocator checkpoint")
    estimator = PatchGainEstimator(
        config["latent_channels"], *config["patch_shape"], config["hidden_dim"],
        group_grid=tuple(config.get("group_grid", (8, 8))),
    ).to(device)
    expected_grid = (allocator.group_rows, allocator.group_cols)
    if (estimator.group_rows, estimator.group_cols) != expected_grid or estimator.block_dim != allocator.block_dim:
        raise ValueError(f"Gain estimator grid/patch differs from transmitter: estimator "
                         f"{estimator.group_rows}x{estimator.group_cols} blocks, transmitter "
                         f"{expected_grid[0]}x{expected_grid[1]} blocks. Use a checkpoint trained for this layout.")
    estimator.load_state_dict(checkpoint["estimator"])
    estimator.eval().requires_grad_(False)
    return estimator, config


def observation_loss(predicted_z0, equalized_observation, precision, q, rows=8, cols=8):
    """Per-image 1/2 sum_b lambda_b ||[Q^T P_p(z0)]_b-U_tilde_b||^2."""
    with torch.autocast(device_type=predicted_z0.device.type, enabled=False):
        predicted = torch.matmul(q.float().T, patchify(predicted_z0.float(), rows, cols))
        observed = patchify(equalized_observation.float(), rows, cols)
        if predicted.shape != observed.shape or precision.shape != predicted.shape[:2]:
            raise ValueError("Observation or precision does not match predicted block shape")
        return 0.5 * (precision.float() * (predicted - observed).square().sum(dim=-1)).sum(dim=1)
