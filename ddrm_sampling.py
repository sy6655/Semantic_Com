"""DDRM (arXiv:2201.11793, Eqs. 7–8) in learned transport coordinates.

U_tilde = Q^T P_p(z0) + diag(lambda^-1/2) epsilon.  The existing
orthogonal Q supplies the right singular basis; no SVD is needed.
"""

import math

import torch
import torch.nn.functional as F

from orthogonal_power import observation_loss, patchify, unpatchify


class OrthogonalDDRM:
    """Use the frozen SD prior in spatial VP coordinates and sample in Q space.

    This full-rank receiver requires positive finite precision. By default,
    sigma_T^2 >= 1/lambda_b is required for every image/block. Oracle and gain
    estimator callers use initialization_policy="clip": excess-noise blocks
    start from the observation with zero added noise, retaining the supplied
    precision in Eq. 8. This initialization is an approximation above sigma_T.
    """

    def __init__(self, reverse_model, eta=0.85, eta_b=1.0, initialization_policy="error"):
        for name, value in (("eta", eta), ("eta_b", eta_b)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"DDRM {name} must be finite and between 0 and 1")
        if initialization_policy not in ("error", "clip"):
            raise ValueError("DDRM initialization_policy must be 'error' or 'clip'")
        self.model = reverse_model
        self.eta, self.eta_b = eta, eta_b
        self.initialization_policy = initialization_policy
        self.last_diagnostics = {}

    @torch.no_grad()
    def __call__(self, equalized_observation, precision, q, generator,
                 rows=8, cols=8, verbose=True):
        self.last_diagnostics = {}
        with torch.autocast(device_type=equalized_observation.device.type, enabled=False):
            observed = equalized_observation.detach().float()
            blocks = patchify(observed, rows, cols)
            weights = precision.detach().to(device=observed.device, dtype=torch.float32)
            matrix = q.detach().to(device=observed.device, dtype=torch.float32)
            count = rows * cols
            if weights.shape != (observed.shape[0], count):
                raise ValueError("DDRM precision must have shape [B,rows*cols]")
            if not torch.isfinite(observed).all():
                raise ValueError("DDRM observation must be finite")
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("DDRM precision must be finite and strictly positive")
            identity = torch.eye(count, device=observed.device)
            if (matrix.shape != (count, count) or not torch.isfinite(matrix).all()
                    or not torch.allclose(matrix.T @ matrix, identity, atol=1e-4, rtol=1e-4)):
                raise ValueError("DDRM Q must be a finite orthogonal [rows*cols,rows*cols] matrix")
            schedule = self.model.inference_timesteps(observed.device)
            maximum = int(self.model.scheduler.config.num_train_timesteps) - 1
            if (schedule.ndim != 1 or len(schedule) == 0 or int(schedule[0]) != maximum
                    or (schedule < 0).any() or (schedule[1:] >= schedule[:-1]).any()):
                raise ValueError("DDRM requires a descending trailing schedule starting at maximum timestep")
            alphas = self.model.scheduler.alphas_cumprod.to(
                device=observed.device, dtype=torch.float32
            )[schedule]
            if not torch.isfinite(alphas).all() or (alphas <= 0).any() or (alphas > 1).any():
                raise ValueError("DDRM alpha_bars must be finite and in (0,1]")
            sigmas_squared = (1 - alphas) / alphas
            variance = weights.reciprocal()
            unsupported = variance > sigmas_squared[0]
            clipped_blocks = unsupported.sum(1).cpu().tolist()
            clipped_block_count = sum(clipped_blocks)
            clipped_images = sum(count > 0 for count in clipped_blocks)
            if clipped_block_count and self.initialization_policy == "error":
                raise ValueError(
                    f"DDRM initialization requires sigma_T^2 >= 1/lambda_b; "
                    f"{clipped_block_count} image/block entries violate this condition "
                    f"(sigma_T={sigmas_squared[0].sqrt().item():.6g}, "
                    f"max observation std={variance.max().sqrt().item():.6g}). "
                    "No noise clipping or transmitter changes were applied."
                )
            prediction_type = self.model.scheduler.config.prediction_type
            if prediction_type not in ("epsilon", "v_prediction", "sample"):
                raise ValueError(f"Unsupported DDRM prediction_type: {prediction_type}")

            def noise():
                return torch.randn(blocks.shape, device=observed.device,
                                   dtype=torch.float32, generator=generator)

            # Eq. 7 for supported blocks. Initialization approximation for excess
            # noise: W_T,b=U_tilde_b (zero added noise). Do not clamp weights
            # or variance: reverse updates still use the supplied precision.
            spectral = blocks + (sigmas_squared[0] - variance).clamp_min(0).sqrt()[..., None] * noise()
            current = alphas[0].sqrt() * unpatchify(matrix @ spectral, observed.shape, rows, cols)
            if verbose and clipped_block_count:
                print(f"DDRM initialization: zero added noise for {clipped_block_count} blocks "
                      f"in {clipped_images} images; supplied precision retained", flush=True)
            prompt = self.model._empty_prompt_embedding(observed.shape[0])
            history = []
            for index, timestep in enumerate(schedule):
                alpha = alphas[index]
                model_input = self.model.scheduler.scale_model_input(current, timestep)
                height, width = current.shape[-2:]
                padded = F.pad(model_input, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
                prediction = self.model.unet(
                    padded.to(dtype=self.model.dtype), timestep, encoder_hidden_states=prompt
                ).sample[..., :height, :width].float()
                if prediction_type == "epsilon":
                    clean = (current - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                elif prediction_type == "v_prediction":
                    clean = alpha.sqrt() * current - (1 - alpha).sqrt() * prediction
                else:
                    clean = prediction
                clean_blocks = matrix.T @ patchify(clean, rows, cols)

                # Use the actual next scheduled alpha. The terminal state is
                # clean (alpha=1, sigma=0), regardless of DDIM's final alpha.
                next_timestep = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
                alpha_next = alphas[index + 1] if next_timestep >= 0 else alpha.new_tensor(1)
                sigma_squared = (1 - alpha_next) / alpha_next
                sigma = sigma_squared.sqrt()
                case_b = sigma_squared >= variance
                # Eq. 8 A: diffusion noise is smaller than observation noise.
                mean_a = clean_blocks + (
                    math.sqrt(1 - self.eta**2) * sigma * weights.sqrt()
                )[..., None] * (blocks - clean_blocks)
                # Eq. 8 B: directly mix the sufficiently precise observation.
                mean_b = (1 - self.eta_b) * clean_blocks + self.eta_b * blocks
                std = torch.where(
                    case_b, (sigma_squared - self.eta_b**2 * variance).clamp_min(0).sqrt(),
                    self.eta * sigma,
                )
                spectral = torch.where(case_b[..., None], mean_b, mean_a) + std[..., None] * noise()
                current = alpha_next.sqrt() * unpatchify(matrix @ spectral, observed.shape, rows, cols)
                losses = observation_loss(clean, observed, weights, matrix, rows, cols)
                if not torch.isfinite(current).all() or not torch.isfinite(losses).all():
                    raise FloatingPointError(f"Non-finite DDRM state or loss at timestep {int(timestep)}")
                history.append({
                    "timestep": int(timestep), "next_timestep": next_timestep,
                    "alpha_bar_next": alpha_next.item(), "sigma_next": sigma.item(),
                    "case_a_blocks": (~case_b).sum(1).cpu().tolist(),
                    "case_b_blocks": case_b.sum(1).cpu().tolist(),
                    "observation_loss": losses.cpu().tolist(),
                })
                if verbose and (index == 0 or (index + 1) % 5 == 0 or index + 1 == len(schedule)):
                    print(f"DDRM step={index + 1}/{len(schedule)} t={int(timestep)} "
                          f"sigma_next={sigma.item():.4f} observation_loss={losses.mean().item():.4f}", flush=True)
            self.last_diagnostics = {
                "method": "ddrm", "initialization": ("spectral_observation_clipped_variance"
                    if clipped_block_count else "spectral_observation_eq7"),
                "initial_timestep": maximum, "timesteps": schedule.cpu().tolist(),
                "initial_sigma": sigmas_squared[0].sqrt().item(),
                "max_observation_std": variance.max().sqrt().item(),
                "initialization_policy": self.initialization_policy,
                "initialization_clipped_blocks": clipped_blocks,
                "initialization_clipped_block_count": clipped_block_count,
                "initialization_clipped_images": clipped_images,
                "eta": self.eta, "eta_b": self.eta_b, "prediction_type": prediction_type,
                "terminal_alpha_bar": 1.0, "terminal_sigma": 0.0, "steps": history,
            }
            return current
