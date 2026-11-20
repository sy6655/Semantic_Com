"""Diffusion posterior sampling using observations in learned block space."""

import math

import torch
import torch.nn.functional as F

from orthogonal_power import observation_loss


def resolve_max_timestep(args):
    """Default DDRM's allocation noise ceiling to the SD1.5 maximum.

    An explicit --max_timestep always takes precedence. DPS keeps its
    existing opt-in allocation floor.
    """
    value = getattr(args, "max_timestep", None)
    if value is None and getattr(args, "restoration_method", "dps") == "ddrm":
        return 999
    return value


def add_restoration_arguments(parser):
    """Shared receiver selection for Q-based inference and evaluation."""
    parser.add_argument("--restoration_method", choices=("dps", "ddrm"), default="dps")
    parser.add_argument("--ddrm_eta", type=float, default=0.85,
                        help="DDRM Case A sampling noise, in [0,1]")
    parser.add_argument("--ddrm_eta_b", type=float, default=1.0,
                        help="DDRM Case B observation mixing, in [0,1]")


def build_restoration_sampler(reverse_model, args, receiver_mode=None):
    method = getattr(args, "restoration_method", "dps")
    if method == "ddrm":
        from ddrm_sampling import OrthogonalDDRM
        receiver_mode = receiver_mode or getattr(args, "receiver_mode", None)
        return OrthogonalDDRM(reverse_model, getattr(args, "ddrm_eta", 0.85),
                              getattr(args, "ddrm_eta_b", 1.0),
                              initialization_policy=("clip" if receiver_mode in ("oracle", "gain_estimator")
                                                     else "error"))
    if method != "dps":
        raise ValueError(f"Unknown restoration_method: {method}")
    return OrthogonalPosteriorSampling(reverse_model, args.guidance_scale,
                                       args.guidance_max_rms, args.posterior_eta)


class OrthogonalPosteriorSampling:
    """Run every scheduled step from Gaussian z_T, backpropagating through z0.

    The supplied reverse_model provides the frozen UNet, empty text prompt,
    and DDIM scheduler already used by the classification inference code.
    guidance_scale is eta_0; eta_i=eta_0*sqrt(alpha_bar_next).
    max_grad_rms bounds each image's gradient RMS; 0 disables clipping.
    """

    def __init__(self, reverse_model, guidance_scale=0.1, max_grad_rms=1.0, eta=0.0):
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(max_grad_rms) or max_grad_rms < 0:
            raise ValueError("max_grad_rms must be finite and non-negative")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be between 0 and 1")
        self.model = reverse_model
        self.guidance_scale, self.max_grad_rms, self.eta = guidance_scale, max_grad_rms, eta
        self.last_diagnostics = {}

    def __call__(self, equalized_observation, precision, q, generator,
                 rows=8, cols=8, verbose=True):
        # Inference callers may use inference_mode. DPS needs an ordinary
        # autograd graph through the frozen denoiser, including its z0 output.
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type=equalized_observation.device.type, enabled=False
        ):
            observed = equalized_observation.detach().float().clone()
            weights = precision.detach().float().clone()
            matrix = q.detach().float().clone()
            if not torch.isfinite(observed).all() or not torch.isfinite(weights).all():
                raise ValueError("Observation and precision must be finite")
            if (weights < 0).any():
                raise ValueError("Precision must be non-negative")
            schedule = self.model.inference_timesteps(observed.device)
            maximum = int(self.model.scheduler.config.num_train_timesteps) - 1
            if int(schedule[0]) != maximum:
                raise ValueError("Posterior sampling must use a trailing schedule starting at maximum timestep")
            alpha_bars = self.model.scheduler.alphas_cumprod.to(
                device=observed.device, dtype=torch.float32
            )
            terminal_alpha = self.model.scheduler.final_alpha_cumprod.to(
                device=observed.device, dtype=torch.float32
            )
            current = torch.randn(observed.shape, device=observed.device,
                                  dtype=torch.float32, generator=generator)
            current = current * float(self.model.scheduler.init_noise_sigma)
            with torch.no_grad():
                prompt = self.model._empty_prompt_embedding(observed.shape[0]).detach().clone()
            history = []
            for index, timestep in enumerate(schedule):
                state = current.detach().requires_grad_(True)
                model_input = self.model.scheduler.scale_model_input(state, timestep)
                height, width = state.shape[-2:]
                padded = F.pad(model_input, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
                prediction = self.model.unet(
                    padded.to(dtype=self.model.dtype), timestep,
                    encoder_hidden_states=prompt,
                ).sample[..., :height, :width].float()
                step = self.model.scheduler.step(
                    prediction, timestep, state, eta=self.eta, generator=generator
                )
                losses = observation_loss(step.pred_original_sample, observed, weights,
                                          matrix, rows, cols)
                # Sum across independent images preserves the per-image step
                # size when inference batch size changes.
                gradient, = torch.autograd.grad(losses.sum(), state)
                rms = gradient.flatten(1).square().mean(dim=1).sqrt()
                if not torch.isfinite(losses).all() or not torch.isfinite(rms).all():
                    raise FloatingPointError(f"Non-finite DPS loss or gradient at timestep {int(timestep)}")
                scale = torch.ones_like(rms)
                if self.max_grad_rms > 0:
                    scale = (self.max_grad_rms / rms.clamp_min(1e-12)).clamp(max=1)
                    gradient = gradient * scale[:, None, None, None]
                next_timestep = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
                alpha_next = alpha_bars[next_timestep] if next_timestep >= 0 else terminal_alpha
                guidance_step_size = self.guidance_scale * alpha_next.sqrt()
                current = (step.prev_sample.detach() - guidance_step_size * gradient).detach()
                history.append({"timestep": int(timestep),
                                "next_timestep": next_timestep,
                                "alpha_bar_next": alpha_next.item(),
                                "guidance_step_size": guidance_step_size.item(),
                                "observation_loss": losses.detach().cpu().tolist(),
                                "gradient_rms": rms.detach().cpu().tolist(),
                                "clipped_gradient_rms": (rms * scale).detach().cpu().tolist()})
                if verbose and (index == 0 or (index + 1) % 5 == 0 or index + 1 == len(schedule)):
                    print(f"DPS step={index + 1}/{len(schedule)} t={int(timestep)} "
                          f"observation_loss={losses.mean().item():.4f}", flush=True)
                # Release the UNet graph before building the next step.
                del state, model_input, padded, prediction, step, losses, gradient, rms, scale
            self.last_diagnostics = {"initialization": "Gaussian_noise", "initial_timestep": maximum,
                                     "timesteps": schedule.cpu().tolist(),
                                     "guidance_scale": self.guidance_scale,
                                     "guidance_schedule": "eta_i=eta_0*sqrt(alpha_bar_next)",
                                     "max_grad_rms": self.max_grad_rms, "eta": self.eta,
                                     "observation_loss": "0.5*sum_b(lambda_b*||Q^T*P(z0)_b-U_tilde_b||^2)",
                                     "steps": history}
            return current
