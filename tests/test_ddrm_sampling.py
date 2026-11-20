import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import DDIMScheduler

from ddrm_sampling import OrthogonalDDRM
from orthogonal_power import OrthogonalPatchTransform, patchify
from posterior_sampling import (
    OrthogonalPosteriorSampling, add_restoration_arguments, build_restoration_sampler,
)


class DDRMSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def backend(self, steps=2, prediction_type="epsilon", denoiser=None):
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            timestep_spacing="trailing", clip_sample=False, set_alpha_to_one=False,
            prediction_type=prediction_type,
        )

        def inference_timesteps(device):
            scheduler.set_timesteps(steps, device=device)
            return scheduler.timesteps

        return SimpleNamespace(
            scheduler=scheduler, dtype=torch.float32,
            inference_timesteps=inference_timesteps,
            _empty_prompt_embedding=lambda batch: torch.zeros(batch, 1, 1),
            unet=denoiser or (lambda latent, timestep, encoder_hidden_states:
                             SimpleNamespace(sample=0.1 * latent)),
        )

    def test_eq7_initialization_in_spatial_vp_coordinates(self):
        seen = []

        def unet(latent, timestep, encoder_hidden_states):
            seen.append(latent.clone())
            self.assertFalse(torch.is_grad_enabled())
            return SimpleNamespace(sample=0.1 * latent)

        backend = self.backend(steps=1, denoiser=unet)
        transform = OrthogonalPatchTransform(rows=2, cols=2)
        with torch.no_grad():
            transform.upper.normal_(std=0.2)
        q = transform.matrix().detach()
        observation = torch.randn(2, 2, 4, 6)
        weights = torch.tensor([[1., 2., 4., 8.], [3., 5., 7., 9.]])
        alpha = backend.scheduler.alphas_cumprod[-1]
        sigma_squared = (1 - alpha) / alpha
        epsilon = torch.randn(patchify(observation, 2, 2).shape,
                              generator=torch.Generator().manual_seed(17))
        spectral_observation = patchify(observation, 2, 2)
        expected_spectral = spectral_observation + (sigma_squared - weights.reciprocal()).sqrt()[..., None] * epsilon
        sampler = OrthogonalDDRM(backend)
        with torch.inference_mode():
            result = sampler(observation, weights, q, torch.Generator().manual_seed(17),
                             rows=2, cols=2, verbose=False)
        initial = seen[0][..., :4, :6]
        torch.testing.assert_close(patchify(initial / alpha.sqrt(), 2, 2), q @ expected_spectral)
        # With one step the terminal transition returns the denoiser's z0,
        # even though this scheduler's final_alpha_cumprod is not one.
        expected_clean = (initial - (1 - alpha).sqrt() * 0.1 * initial) / alpha.sqrt()
        torch.testing.assert_close(result, expected_clean, atol=2e-5, rtol=2e-5)
        self.assertEqual(sampler.last_diagnostics["steps"][-1]["alpha_bar_next"], 1.)
        self.assertEqual(sampler.last_diagnostics["steps"][-1]["sigma_next"], 0.)

    def test_eq8_empirical_mean_and_variance_for_both_noise_regimes(self):
        batch = 24000
        seen = []

        def unet(latent, timestep, encoder_hidden_states):
            seen.append(latent[..., :1, :2].clone())
            return SimpleNamespace(sample=torch.full_like(latent, 0.4))

        backend = self.backend(prediction_type="sample", denoiser=unet)
        backend.scheduler.alphas_cumprod[-1] = 0.1  # sigma_T = 3
        backend.scheduler.alphas_cumprod[499] = 0.5  # sigma_next = 1
        observation = torch.tensor([2., -1.]).reshape(1, 1, 1, 2).expand(batch, -1, -1, -1)
        weights = torch.tensor([[0.25, 4.]]).expand(batch, -1)
        sampler = OrthogonalDDRM(backend, eta=0.6, eta_b=0.7)
        result = sampler(observation, weights, torch.eye(2), torch.Generator().manual_seed(8),
                         rows=1, cols=2, verbose=False)
        samples = (seen[1] / math.sqrt(0.5)).reshape(batch, 2)
        # A: mu=.4 + sqrt(1-.6^2)*(2-.4)/2; variance=.6^2.
        # B: mu=(1-.7)*.4 + .7*(-1); variance=1-.7^2*.25.
        torch.testing.assert_close(samples.mean(0), torch.tensor([1.04, -0.58]), atol=0.02, rtol=0)
        torch.testing.assert_close(samples.var(0), torch.tensor([0.36, 0.8775]), atol=0.03, rtol=0)
        torch.testing.assert_close(result, torch.full_like(result, 0.4))
        self.assertEqual(sampler.last_diagnostics["steps"][0]["case_a_blocks"], [1] * batch)
        self.assertEqual(sampler.last_diagnostics["steps"][0]["case_b_blocks"], [1] * batch)

    def test_equality_uses_case_b_and_zero_variance_is_valid(self):
        seen = []

        def unet(latent, timestep, encoder_hidden_states):
            seen.append(latent.clone())
            return SimpleNamespace(sample=torch.zeros_like(latent))

        backend = self.backend(prediction_type="sample", denoiser=unet)
        backend.scheduler.alphas_cumprod[-1] = 0.1
        backend.scheduler.alphas_cumprod[499] = 0.5
        observation = torch.tensor([[[[2., -3.]]]])
        sampler = OrthogonalDDRM(backend, eta_b=1)
        sampler(observation, torch.ones(1, 2), torch.eye(2), torch.Generator().manual_seed(9),
                rows=1, cols=2, verbose=False)
        torch.testing.assert_close(seen[1][..., :1, :2], math.sqrt(0.5) * observation)
        self.assertEqual(sampler.last_diagnostics["steps"][0]["case_b_blocks"], [2])

    def test_initial_noise_condition_is_checked_before_denoising(self):
        backend = self.backend(denoiser=lambda *args, **kwargs: self.fail("UNet must not run"))
        sampler = OrthogonalDDRM(backend)
        with self.assertRaisesRegex(ValueError, "sigma_T.*1/lambda_b"):
            sampler(torch.zeros(1, 4, 8, 8), torch.full((1, 64), 1e-10), torch.eye(64),
                    torch.Generator(), verbose=False)
        self.assertEqual(sampler.last_diagnostics, {})

    def test_oracle_clips_initial_added_noise_at_logged_observation_levels(self):
        for observation_std in (14.8243, 46.8784):
            with self.subTest(observation_std=observation_std):
                seen = []

                def unet(latent, timestep, encoder_hidden_states):
                    seen.append(latent.clone())
                    return SimpleNamespace(sample=.1 * latent)

                backend = self.backend(steps=1, denoiser=unet)
                observed = torch.randn(1, 4, 8, 8)
                weights = torch.full((1, 64), observation_std**-2)
                original_weights = weights.clone()
                sampler = OrthogonalDDRM(backend, initialization_policy="clip")
                with torch.inference_mode():
                    restored = sampler(observed, weights, torch.eye(64), torch.Generator(), verbose=False)
                alpha = backend.scheduler.alphas_cumprod[-1]
                torch.testing.assert_close(seen[0], alpha.sqrt() * observed)
                torch.testing.assert_close(weights, original_weights, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(restored).all())
                self.assertEqual(sampler.last_diagnostics["initialization_policy"], "clip")
                self.assertEqual(sampler.last_diagnostics["initialization_clipped_images"], 1)
                self.assertEqual(sampler.last_diagnostics["initialization_clipped_block_count"], 64)
                self.assertEqual(sampler.last_diagnostics["initialization_clipped_blocks"], [64])

    def test_clipped_initialization_keeps_true_precision_in_reverse_sampling(self):
        batch, seen = 12000, []

        def unet(latent, timestep, encoder_hidden_states):
            seen.append(latent[..., :1, :2].clone())
            return SimpleNamespace(sample=torch.full_like(latent, .4))

        backend = self.backend(prediction_type="sample", denoiser=unet)
        backend.scheduler.alphas_cumprod[-1] = .1  # sigma_T=3
        backend.scheduler.alphas_cumprod[499] = .5  # sigma_next=1
        observed = torch.tensor([2., 3.]).reshape(1, 1, 1, 2).expand(batch, -1, -1, -1)
        weights = torch.tensor([[.25, .0625]]).expand(batch, -1)  # observation std=2,4
        sampler = OrthogonalDDRM(backend, eta=.6, initialization_policy="clip")
        sampler(observed, weights, torch.eye(2), torch.Generator().manual_seed(8),
                rows=1, cols=2, verbose=False)
        # Only the second direction exceeds sigma_T and gets no added noise.
        initial = seen[0].reshape(batch, 2) / math.sqrt(.1)
        torch.testing.assert_close(initial[:, 1], torch.full((batch,), 3.))
        self.assertGreater(initial[:, 0].var().item(), 4.5)
        # Eq. 8 must use the original std=4, not a capped std=3.
        # mu=.4 + sqrt(1-.6^2)*1*(3-.4)/4 = .92 for the clipped direction.
        next_state = seen[1].reshape(batch, 2) / math.sqrt(.5)
        torch.testing.assert_close(next_state.mean(0), torch.tensor([1.04, .92]), atol=.02, rtol=0)
        torch.testing.assert_close(next_state.var(0), torch.full((2,), .36), atol=.025, rtol=0)
        self.assertEqual(sampler.last_diagnostics["initialization_clipped_blocks"], [1] * batch)

    def test_oracle_policy_leaves_supported_initialization_unchanged(self):
        observed, weights, q = torch.randn(2, 4, 8, 8), torch.ones(2, 64), torch.eye(64)
        strict = OrthogonalDDRM(self.backend())
        oracle = OrthogonalDDRM(self.backend(), initialization_policy="clip")
        expected = strict(observed, weights, q, torch.Generator().manual_seed(4), verbose=False)
        actual = oracle(observed, weights, q, torch.Generator().manual_seed(4), verbose=False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(oracle.last_diagnostics["initialization_clipped_block_count"], 0)

    def test_prediction_parameterizations_and_terminal_clean_output(self):
        target = 0.3
        for prediction_type in ("epsilon", "v_prediction", "sample"):
            with self.subTest(prediction_type=prediction_type):
                backend = self.backend(steps=1, prediction_type=prediction_type)
                alpha = backend.scheduler.alphas_cumprod[-1]

                def unet(latent, timestep, encoder_hidden_states):
                    if prediction_type == "epsilon":
                        prediction = (latent - alpha.sqrt() * target) / (1 - alpha).sqrt()
                    elif prediction_type == "v_prediction":
                        prediction = (alpha.sqrt() * latent - target) / (1 - alpha).sqrt()
                    else:
                        prediction = torch.full_like(latent, target)
                    return SimpleNamespace(sample=prediction)

                backend.unet = unet
                result = OrthogonalDDRM(backend)(torch.ones(1, 4, 8, 8), torch.ones(1, 64),
                                                torch.eye(64), torch.Generator(), verbose=False)
                torch.testing.assert_close(result, torch.full_like(result, target), atol=1e-5, rtol=1e-5)

    def test_seed_replay_with_heterogeneous_batch_precision(self):
        sampler = OrthogonalDDRM(self.backend(steps=4))
        observation = torch.randn(2, 4, 8, 8)
        weights = torch.stack((torch.linspace(.1, 2, 64), torch.linspace(2, 5, 64)))
        first = sampler(observation, weights, torch.eye(64), torch.Generator().manual_seed(10), verbose=False)
        replay = sampler(observation, weights, torch.eye(64), torch.Generator().manual_seed(10), verbose=False)
        other = sampler(observation, weights, torch.eye(64), torch.Generator().manual_seed(11), verbose=False)
        torch.testing.assert_close(first, replay, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, other))

    def test_invalid_inputs_and_sampling_parameters(self):
        for kwargs in ({"eta": -0.1}, {"eta": float("nan")}, {"eta_b": 1.1},
                       {"eta_b": float("inf")}, {"initialization_policy": "invalid"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OrthogonalDDRM(None, **kwargs)
        sampler = OrthogonalDDRM(self.backend())
        observed, weights, q = torch.ones(1, 4, 8, 8), torch.ones(1, 64), torch.eye(64)
        for bad_weights in (weights[:, :2], weights * 0, weights * -1, weights * float("nan")):
            with self.assertRaisesRegex(ValueError, "precision"):
                sampler(observed, bad_weights, q, torch.Generator(), verbose=False)
        for bad_q in (q[:2], 2 * q, q * float("nan")):
            with self.assertRaisesRegex(ValueError, "orthogonal"):
                sampler(observed, weights, bad_q, torch.Generator(), verbose=False)

    def test_receiver_selection_defaults_to_dps(self):
        parser = argparse.ArgumentParser()
        add_restoration_arguments(parser)
        args = parser.parse_args([])
        args.guidance_scale, args.guidance_max_rms, args.posterior_eta = .1, 1., 0.
        self.assertIsInstance(build_restoration_sampler(None, args), OrthogonalPosteriorSampling)
        args = parser.parse_args(["--restoration_method", "ddrm"])
        sampler = build_restoration_sampler(None, args)
        self.assertIsInstance(sampler, OrthogonalDDRM)
        self.assertEqual((sampler.eta, sampler.eta_b), (.85, 1.))
        self.assertEqual(sampler.initialization_policy, "error")
        args.receiver_mode = "gain_estimator"
        self.assertEqual(build_restoration_sampler(None, args).initialization_policy, "error")
        args.receiver_mode = "oracle"
        self.assertEqual(build_restoration_sampler(None, args).initialization_policy, "clip")
        self.assertEqual(build_restoration_sampler(None, args, receiver_mode="gain_estimator").initialization_policy, "error")
        args.receiver_mode = None
        self.assertEqual(build_restoration_sampler(None, args, receiver_mode="oracle").initialization_policy, "clip")

    def test_oracle_inference_writes_ddrm_outputs_for_both_strategies(self):
        self.check_oracle_inference_outputs(deep_fading=False)

    def test_oracle_inference_continues_through_deep_fading_for_both_strategies(self):
        self.check_oracle_inference_outputs(deep_fading=True)

    def check_oracle_inference_outputs(self, deep_fading):
        from inference_oracle_gain import run
        from train_power_allocator import PowerAllocator

        class TinyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))

            def decode(self, latent):
                return SimpleNamespace(sample=torch.nn.functional.interpolate(latent[:, :3], scale_factor=8))

        latent = torch.randn(1, 4, 8, 8) * .2
        transform = OrthogonalPatchTransform()
        with torch.no_grad():
            transform.upper.normal_(std=.03)
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=4)
        with torch.no_grad():
            allocator.orthogonal.upper.copy_(transform.upper)
        transformed, q = transform(latent)
        powers = torch.full((1, 64), 25.)
        energy = patchify(transformed).square().sum(-1)
        budget = float((powers * energy).sum())
        info = {"channel_input": transformed, "orthogonal_matrix": q,
                "payload_energy": energy, "unscaled_block_energy": energy, "target_frame_energy": budget,
                "allocator_snr_db": torch.tensor([10.]), "frame_channel_uses": 128}
        config = {"vae_model_id": "local-test", "vae_model_revision": "local-test",
                  "channel_model": "rayleigh_block_fading", "mean_real_component_power": 1.,
                  "noise_real_component_variance": "N0"}
        channel = SimpleNamespace(
            sample_channel_coefficient=lambda latent, generator: torch.tensor([.001+0j if deep_fading else 1.+0j]),
            _noise_spectral_density=lambda latent, snr: torch.tensor([.1]),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pth"
            torch.save({"allocator": allocator.state_dict(), "config": config, "epoch": 0}, checkpoint_path)
            args = SimpleNamespace(
                device="cpu", checkpoint=str(checkpoint_path), vae_path=None, local_files_only=True,
                seed=3, input_image=None, sample_index=0,
                channel_model="rayleigh" if deep_fading else "awgn", snr_db=10.,
                output_dir=directory, diffusion_steps=2, diffusion_model=None, strategy="both",
                restoration_method="ddrm", ddrm_eta=.85, ddrm_eta_b=1.,
                quiet=True, lpips=False, no_save_latents=False,
            )
            with patch("inference_orthogonal_power.load_transmitter", return_value=(allocator, channel, config, checkpoint_path)), \
                    patch("inference_orthogonal_power.load_input", return_value=(torch.rand(1, 3, 64, 64), 0, "test.png")), \
                    patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                    patch("inference_orthogonal_power.encode_latents", return_value=latent), \
                    patch("inference_orthogonal_power.allocate_power_for_inference", return_value=(powers * energy / budget, powers, info)), \
                    patch("inference_orthogonal_power.StaggeredReverseDiffusion", return_value=self.backend()):
                result = run(args)
            self.assertEqual(result["restoration_method"], "ddrm")
            self.assertEqual(result["max_timestep"], 999)
            self.assertEqual(json.loads((Path(directory) / "result.json").read_text())["restoration_method"], "ddrm")
            for name, tag in (("allocated", "allocated"), ("uniform", "uniform_no_q")):
                strategy = result["strategies"][name]
                self.assertIn("ddrm_vs_input", strategy)
                self.assertIn("ddrm_latent_mse", strategy)
                self.assertNotIn("dps_vs_input", strategy)
                self.assertEqual(strategy["diffusion"]["method"], "ddrm")
                self.assertEqual(strategy["diffusion"]["initialization_policy"], "clip")
                self.assertEqual(strategy["diffusion"]["initialization_clipped_block_count"], 64 if deep_fading else 0)
                self.assertTrue((Path(directory) / f"{tag}_oracle_ddrm.png").is_file())
                state = torch.load(Path(directory) / f"{tag}_oracle_state.pt", weights_only=True)
                self.assertTrue(torch.isfinite(state["restored_latent"]).all())
                if name == "uniform":
                    torch.testing.assert_close(state["Q"], torch.eye(64))
                if deep_fading:
                    self.assertTrue((state["precision"].reciprocal() > strategy["diffusion"]["initial_sigma"]**2).all())

    def test_maxvit_ddrm_transmission_path_and_result_filename(self):
        from evaluate_power_maxvit import evaluation_output_path, restore_transmissions

        latent = torch.randn(2, 4, 8, 8) * .2
        transform = OrthogonalPatchTransform()
        with torch.no_grad():
            transform.upper.normal_(std=.03)
        transformed, q = transform(latent)
        powers = torch.full((2, 64), 25.)
        budget = 25. * patchify(latent).square().sum(-1).sum(1, keepdim=True)
        info = {"channel_input": transformed, "orthogonal_matrix": q,
                "target_frame_energy": budget}
        h = torch.tensor([1.+.2j, .8-.3j])
        channel = SimpleNamespace(
            sample_channel_coefficient=lambda latent, generator: h,
            _noise_spectral_density=lambda latent, snr: torch.full((2,), .1),
        )
        sampler = build_restoration_sampler(self.backend(), SimpleNamespace(restoration_method="ddrm"), receiver_mode="oracle")
        diagnostics = {}
        with patch("evaluate_power_maxvit.allocate_power_for_inference", return_value=(None, powers, info)):
            restored, actual_h, fallback = restore_transmissions(
                latent, torch.full((2,), 10.), None, channel, sampler,
                torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                diagnostics=diagnostics,
            )
        self.assertEqual(set(restored), {"uniform", "allocated"})
        for value in restored.values():
            self.assertEqual(value.shape, latent.shape)
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(actual_h, h)
        self.assertFalse(fallback.any())
        self.assertEqual(set(diagnostics), {"uniform", "allocated"})
        for row in diagnostics.values():
            self.assertEqual(row["initialization_policy"], "clip")
        args = SimpleNamespace(output_path=None, snr_db=[10.], restoration_method="ddrm",
                               diffusion_steps=2, ddrm_eta=.85, ddrm_eta_b=1., seed=0)
        path = evaluation_output_path(args, Path("/tmp/checkpoint/latest.pth"), 10.)
        self.assertEqual(path.name, "maxvit_ddrm_rayleigh_snr_10_steps_2_eta_0.85_etab_1_seed_0_balanced_n5000_maxt_999.json")


if __name__ == "__main__":
    unittest.main()
