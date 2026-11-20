import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import DDIMScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, GAIN_ESTIMATOR_ARCHITECTURE,
    GAIN_ESTIMATOR_VERSION, MEAN_REAL_COMPONENT_POWER,
    NOISE_REAL_COMPONENT_VARIANCE, RAYLEIGH_CHANNEL_MODEL,
    OrthogonalPatchTransform, PatchGainEstimator, allocator_fingerprint,
    apply_complex_channel, block_map,
    gain_block_statistics, gain_estimator_receive, load_gain_estimator, observation_loss, patchify,
    prepare_gain_estimator_input, sample_raw_channel, sample_rayleigh_fading, unpatchify,
)
from posterior_sampling import OrthogonalPosteriorSampling
from train_gain_estimator import gain_estimator_step, load_transmitter
from train_power_allocator import (
    VAE_SCALE, PowerAllocator, RayleighFadingChannel, allocate_power,
    encode_latents, restore_spatial_latent,
    training_step,
)


class OrthogonalTransmissionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def transmitter(self):
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16)
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=0.03)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=0.3)
        channel = RayleighFadingChannel(8, 8)
        return allocator, channel

    def test_vae_posterior_sampling_replays_seed_and_keeps_vae_frozen(self):
        posterior = DiagonalGaussianDistribution(torch.zeros(2, 8, 2, 2, requires_grad=True))
        vae = SimpleNamespace(encode=lambda image: SimpleNamespace(latent_dist=posterior))
        images = torch.zeros(2, 3, 16, 16)
        sampled = encode_latents(images, vae, torch.Generator().manual_seed(31))
        replayed = encode_latents(images, vae, torch.Generator().manual_seed(31))
        different = encode_latents(images, vae, torch.Generator().manual_seed(32))
        expected = posterior.sample(generator=torch.Generator().manual_seed(31)) * VAE_SCALE
        torch.testing.assert_close(sampled, expected)
        torch.testing.assert_close(sampled, replayed, rtol=0, atol=0)
        self.assertFalse(torch.equal(sampled, different))
        self.assertFalse(torch.equal(sampled, posterior.mode() * VAE_SCALE))
        self.assertFalse(sampled.requires_grad)
        self.assertEqual(sampled.dtype, torch.float32)

    def test_patch_order_rotation_orthogonality_energy_and_inverse(self):
        latent = torch.randn(2, 4, 16, 24)
        transform = OrthogonalPatchTransform()
        self.assertEqual(transform.upper.numel(), 2016)
        with torch.no_grad():
            transform.upper.normal_(std=0.05)
        transformed, q = transform(latent)
        torch.testing.assert_close(q.T @ q, torch.eye(64), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(patchify(latent)[:, 0], latent[:, :, :2, :3].reshape(2, -1))
        torch.testing.assert_close(unpatchify(patchify(latent), latent.shape), latent)
        torch.testing.assert_close(patchify(transformed), q.T @ patchify(latent))
        torch.testing.assert_close(transformed.square().sum(), latent.square().sum())
        restored = transform.apply_matrix(transformed, q)
        torch.testing.assert_close(restored, latent, atol=3e-6, rtol=3e-6)

    def test_exact_power_budget_uses_transformed_block_energy(self):
        allocator, channel = self.transmitter()
        latent = torch.randn(2, 4, 16, 16)
        h = torch.tensor([0.3 + 0.4j, -1.2 + 1.6j])
        snr_db = torch.tensor([-5., 10.])
        with patch.object(allocator, "forward", wraps=allocator.forward) as forward:
            ratio, power, info = allocate_power(latent, snr_db, allocator, channel,
                                               channel_coefficient=h)
        expected_snr = snr_db + 20 * h.abs().log10()
        expected_log_gamma = (expected_snr / 10) * torch.log(torch.tensor(10.))
        torch.testing.assert_close(forward.call_args.args[1], expected_log_gamma)
        torch.testing.assert_close(info["allocator_snr_db"], expected_snr)
        torch.testing.assert_close(info["log_gamma"], expected_log_gamma)
        transformed_energy = patchify(info["channel_input"]).square().sum(-1)
        torch.testing.assert_close(info["payload_energy"], transformed_energy)
        torch.testing.assert_close((power * transformed_energy).sum(1), torch.full((2,), 1024.))
        self.assertEqual(info["target_frame_energy"], 1024.)
        self.assertEqual(info["frame_channel_uses"], 512)
        torch.testing.assert_close(info["mean_real_component_power"], torch.ones(2))
        torch.testing.assert_close(info["mean_symbol_power"], torch.full((2,), 2.))
        torch.testing.assert_close(ratio.sum(1), torch.ones(2))
        self.assertFalse(torch.allclose(transformed_energy, patchify(latent).square().sum(-1)))

    def test_full_latent_frame_power_and_configured_real_snr_match(self):
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True)
        channel = RayleighFadingChannel()
        latent, snr_db = torch.randn(2, 4, 64, 64), torch.tensor([0., 10.])
        _, power, info = allocate_power(latent, snr_db, allocator, channel,
                                       channel_coefficient=torch.ones(2, dtype=torch.complex64))
        transmitted = block_map(power.sqrt(), latent, 8, 8) * info["channel_input"]
        measured_energy = transmitted.square().flatten(1).sum(1)
        torch.testing.assert_close(measured_energy, torch.full((2,), 16384.))
        self.assertEqual(info["frame_channel_uses"], 8192)
        mean_real_power = transmitted.square().flatten(1).mean(1)
        n0 = channel._noise_spectral_density(latent, snr_db)
        torch.testing.assert_close(10 * (mean_real_power / n0).log10(), snr_db, atol=2e-5, rtol=1e-5)

    def test_task_loss_updates_q_allocator_and_classifier(self):
        allocator, channel = self.transmitter()
        classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1024, 7))
        latent = torch.randn(2, 4, 16, 16)
        before = allocator.orthogonal.upper.detach().clone()
        optimizer = torch.optim.SGD(list(allocator.parameters()) + list(classifier.parameters()), lr=0.01)
        with patch("train_power_allocator.encode_latents", return_value=latent):
            result = training_step(latent, torch.tensor([1, 2]), None, allocator, classifier,
                                   channel, torch.tensor([0., 5.]), condition_classifier_on_snr=False)
        result["loss"].backward()
        for parameter in (allocator.orthogonal.upper, allocator.block_allocator.logit_head[-1].weight,
                          classifier[1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, allocator.orthogonal.upper))

    def test_gain_h_correction_precision_and_analytic_observation_gradient(self):
        allocator, channel = self.transmitter()
        latent = torch.randn(2, 4, 16, 16)
        h = torch.tensor([0.3 + 0.4j, -1.2 + 1.6j])
        _, power, info = allocate_power(latent, torch.zeros(2), allocator, channel,
                                       channel_coefficient=h)
        q, transformed = info["orthogonal_matrix"].detach(), info["channel_input"].detach()
        n0 = torch.tensor([0.2, 1.0])
        raw = sample_raw_channel(transformed, power, n0, h, epsilon=torch.zeros_like(transformed))
        received = gain_estimator_receive(raw, 0.5 * power.log(), n0, h)
        torch.testing.assert_close(received["equalized_latent"], transformed)
        torch.testing.assert_close(received["precision"], h.abs()[:, None].square() * power / n0[:, None])
        torch.testing.assert_close(received["noise_variance"], n0[:, None] / (h.abs()[:, None].square() * power))
        restored, info_out = restore_spatial_latent(
            received["equalized_latent"], info, {"noise_variance": received["noise_variance"]}
        )
        torch.testing.assert_close(restored, latent, atol=3e-6, rtol=3e-6)
        expected_variance = (q.square() @ received["noise_variance"].T).T
        torch.testing.assert_close(info_out["noise_variance"], expected_variance)
        weights = received["precision"].detach()
        self.assertLess(observation_loss(latent, transformed, weights, q).max().item(), 1e-8)
        predicted = torch.randn_like(latent, requires_grad=True)
        loss = observation_loss(predicted, transformed, weights, q)
        gradient, = torch.autograd.grad(loss.sum(), predicted)
        residual = q.T @ patchify(predicted.detach()) - patchify(transformed)
        expected = unpatchify(q @ (weights[..., None] * residual), predicted.shape)
        torch.testing.assert_close(gradient, expected)

    def test_raw_channel_noise_variance_and_fading(self):
        latent = torch.ones(4, 4, 64, 64)
        h = torch.tensor([1.5 + 0j, 0 + 1.5j, -0.9 + 1.2j, 0.9 - 1.2j])
        power, n0 = torch.full((4, 64), 4.), torch.full((4,), 0.6)
        raw = sample_raw_channel(latent, power, n0, h, torch.Generator().manual_seed(5))
        residual = raw - apply_complex_channel(2 * latent, h)
        self.assertAlmostEqual(residual.mean().item(), 0, delta=0.01)
        self.assertAlmostEqual(residual.var().item(), 0.6, delta=0.01)
        deterministic_raw = sample_raw_channel(latent, power, n0, h, epsilon=torch.ones_like(latent))
        torch.testing.assert_close(deterministic_raw - apply_complex_channel(2 * latent, h),
                                   n0.sqrt()[:, None, None, None].expand_as(latent), rtol=1e-5, atol=1e-6)
        received = gain_estimator_receive(raw, 0.5 * power.log(), n0, h)
        self.assertAlmostEqual((received["equalized_latent"] - latent).var().item(),
                               0.6 / (1.5 ** 2 * 4), delta=0.001)

    def test_rayleigh_draw_matches_requested_formula_and_unit_mean_power(self):
        generator = torch.Generator().manual_seed(18)
        h = sample_rayleigh_fading(100000, "cpu", generator)
        reference = torch.Generator().manual_seed(18)
        expected = (torch.randn(100000, generator=reference)
                    + 1j * torch.randn(100000, generator=reference)) / (2 ** 0.5)
        torch.testing.assert_close(h, expected, rtol=0, atol=0)
        self.assertEqual(h.shape, (100000,))
        self.assertAlmostEqual(h.real.var().item(), 0.5, delta=0.01)
        self.assertAlmostEqual(h.imag.var().item(), 0.5, delta=0.01)
        self.assertAlmostEqual(h.abs().square().mean().item(), 1, delta=0.01)
        self.assertFalse(torch.equal(sample_rayleigh_fading(2, "cpu", generator), h[:2]))

    def test_common_complex_h_rotates_every_block_and_is_fully_equalized(self):
        blocks = torch.zeros(2, 64, 16)
        blocks[..., 0::2] = 1
        latent = unpatchify(blocks, (2, 4, 16, 16))
        h = torch.tensor([1j, -0.6 + 0.8j])
        power, n0 = torch.full((2, 64), 4.), torch.ones(2)
        raw = sample_raw_channel(latent, power, n0, h, epsilon=torch.zeros_like(latent))
        raw_blocks = patchify(raw)
        torch.testing.assert_close(raw_blocks[..., 0::2], 2 * h.real[:, None, None].expand(2, 64, 8))
        torch.testing.assert_close(raw_blocks[..., 1::2], 2 * h.imag[:, None, None].expand(2, 64, 8))
        corrected = gain_estimator_receive(raw, 0.5 * power.log(), n0, h)
        torch.testing.assert_close(corrected["equalized_latent"], latent)
        with self.assertRaisesRegex(ValueError, "shared by all blocks"):
            sample_raw_channel(latent, power, n0, h[:, None].expand(2, 64))

    def test_gain_estimator_uses_perfect_csi_to_remove_phase(self):
        estimator = PatchGainEstimator(4, 2, 2, 8)
        with torch.no_grad():
            estimator.log_gain_head.weight.normal_(std=0.3)
        raw, n0 = torch.randn(2, 4, 16, 16), torch.tensor([0.2, 0.6])
        h = torch.tensor([0.3 + 0.4j, -0.6 + 0.8j])
        rotated_raw = apply_complex_channel(raw, torch.full((2,), 1j))
        inputs = prepare_gain_estimator_input(raw, n0, h)
        rotated_inputs = prepare_gain_estimator_input(rotated_raw, n0, 1j * h)
        with patch.object(estimator.stem, "forward", wraps=estimator.stem.forward) as stem:
            predicted = estimator(inputs["channel_equalized_latent"], inputs["log_noise_variance"], inputs["log_power"])
        expected_r_h = apply_complex_channel(raw, h.reciprocal())
        expected_input = patchify(expected_r_h).reshape(-1, 1, 16)
        torch.testing.assert_close(stem.call_args.args[0], expected_input)
        torch.testing.assert_close(inputs["log_noise_variance"], (n0 / h.abs().square()).log())
        torch.testing.assert_close(inputs["log_power"], torch.zeros(2))
        torch.testing.assert_close(predicted, estimator(rotated_inputs["channel_equalized_latent"],
                                                        rotated_inputs["log_noise_variance"], rotated_inputs["log_power"]))

    def test_v4_default_architectures_match_specification_parameter_counts(self):
        allocator = PowerAllocator(4, 64, 64, orthogonal_transform=True)
        estimator = PatchGainEstimator()
        self.assertEqual(sum(p.numel() for p in allocator.block_allocator.parameters()), 50625)
        self.assertEqual(sum(p.numel() for p in allocator.parameters()), 50625 + 2016)
        self.assertEqual(sum(p.numel() for p in estimator.parameters()), 774080)
        self.assertFalse(any(isinstance(m, torch.nn.Conv2d) for m in allocator.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.Conv2d) for m in estimator.modules()))

    def test_gain_statistics_match_section_vi_b_formulas(self):
        blocks = torch.tensor([[[1., 2., 3., 4.], [-2., -1., 1., 2.]]])
        actual = gain_block_statistics(blocks)
        expected = torch.tensor([[[2.5, 7.5, 2.5, 88.5 / 7.5**2, (20/3) / 7.5, 1.25],
                                  [0., 2.5, 1.5, 8.5 / 2.5**2, 1. / 2.5, 2.5]]])
        expected[..., [1, 3, 5]] = expected[..., [1, 3, 5]].log()
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(gain_block_statistics(torch.zeros_like(blocks))).all())

    def test_rayleigh_channel_requires_transmitter_csi_before_allocation(self):
        allocator, channel = self.transmitter()
        with self.assertRaisesRegex(ValueError, "before allocation"):
            allocate_power(torch.randn(2, 4, 16, 16), torch.zeros(2), allocator, channel)

    def test_gain_estimator_trains_with_frozen_transmitter(self):
        allocator, channel = self.transmitter()
        allocator.requires_grad_(False)
        estimator = PatchGainEstimator(4, 2, 2, 8)
        latent = torch.randn(2, 4, 16, 16)
        with patch("train_gain_estimator.encode_latents", return_value=latent):
            result = gain_estimator_step(latent, None, allocator, channel, estimator,
                                         torch.tensor([-2., 3.]))
        result["loss"].backward()
        self.assertGreater(estimator.log_gain_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in allocator.parameters()))

    def test_checkpoint_roundtrip_and_estimator_transmitter_binding(self):
        allocator, _ = self.transmitter()
        estimator = PatchGainEstimator(4, 2, 2, 8)
        config = {"dataset": "ImageNet", "orthogonal_transform": True, "num_groups": 64,
                  "transmission_mode": "analog", "pilots_per_group": 0, "hidden_dim": 8,
                  "channel_model": RAYLEIGH_CHANNEL_MODEL,
                  "allocator_architecture": ALLOCATOR_ARCHITECTURE, "allocator_block_dim": 16,
                  "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
                  "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE}
        with tempfile.TemporaryDirectory() as directory:
            transmitter_path = Path(directory) / "power.pth"
            torch.save({"allocator": allocator.state_dict(), "config": config}, transmitter_path)
            loaded, _, _, _ = load_transmitter(transmitter_path, "cpu")
            self.assertEqual(allocator_fingerprint(loaded), allocator_fingerprint(allocator))
            with patch("train_gain_estimator.torch.load", side_effect=AssertionError("Checkpoint must not be reopened")):
                supplied, _, _, _ = load_transmitter(
                    transmitter_path, "cpu", checkpoint={"allocator": allocator.state_dict(), "config": config})
            self.assertEqual(allocator_fingerprint(supplied), allocator_fingerprint(allocator))
            estimator_path = Path(directory) / "gain.pth"
            torch.save({"estimator": estimator.state_dict(), "config": {
                "stage": "block_gain_estimator", "estimator_version": GAIN_ESTIMATOR_VERSION,
                "estimator_architecture": GAIN_ESTIMATOR_ARCHITECTURE,
                "channel_model": RAYLEIGH_CHANNEL_MODEL,
                "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
                "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE,
                "allocator_fingerprint": allocator_fingerprint(loaded),
                "latent_channels": 4, "patch_shape": [2, 2], "hidden_dim": 8,
            }}, estimator_path)
            load_gain_estimator(estimator_path, loaded, "cpu")
            with torch.no_grad():
                loaded.orthogonal.upper[0] += 0.1
            with self.assertRaisesRegex(ValueError, "different Q/allocator"):
                load_gain_estimator(estimator_path, loaded, "cpu")

    def test_legacy_allocator_state_dict_remains_compatible(self):
        from power_utils import LegacyPowerAllocator
        # Frozen outputs from the original CNN before the bit-code cleanup.
        expected = torch.tensor([
            [0.1480638236, 0.1662092656, 0.1548195034, 0.1704529673],
            [0.1225445867, 0.1177293882, 0.1342644840, 0.1255979389],
        ])
        for sampling in (False, True):
            with self.subTest(sampling=sampling):
                legacy = LegacyPowerAllocator(4, 4, 3, sampling=sampling)
                generator = torch.Generator().manual_seed(31)
                with torch.no_grad():
                    for parameter in legacy.parameters():
                        parameter.copy_(.1 * torch.randn(parameter.shape, generator=generator))
                head = 'log_sigma_head' if sampling else 'mu_head'
                self.assertEqual(set(legacy.state_dict()), {
                    'backbone.0.weight', 'backbone.0.bias', 'backbone.2.weight', 'backbone.2.bias',
                    head + '.weight', head + '.bias',
                })
                model = PowerAllocator(4, 4, 3, sampling=sampling)
                model.load_state_dict(legacy.state_dict(), strict=True)
                latent = torch.randn(2, 4, 5, 7, generator=generator)
                torch.testing.assert_close(model(latent, torch.tensor([-5., 10.])), expected)


class PosteriorSamplingTests(unittest.TestCase):
    def backend(self, steps=1, set_alpha_to_one=True):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule="scaled_linear", timestep_spacing="trailing",
                                  clip_sample=False, set_alpha_to_one=set_alpha_to_one)
        def inference_timesteps(device):
            scheduler.set_timesteps(steps, device=device)
            return scheduler.timesteps
        return SimpleNamespace(
            scheduler=scheduler, dtype=torch.float32,
            inference_timesteps=inference_timesteps,
            _empty_prompt_embedding=lambda batch: torch.zeros(batch, 1, 1),
            unet=lambda latent, timestep, encoder_hidden_states: SimpleNamespace(sample=0.1 * latent),
        )

    def test_dps_exact_step_includes_denoiser_jacobian_under_inference_mode(self):
        backend = self.backend(set_alpha_to_one=False)
        q = OrthogonalPatchTransform().matrix().detach()
        observed, weights = torch.ones(1, 4, 8, 8), torch.linspace(0.5, 2, 64)[None]
        schedule = backend.inference_timesteps(torch.device("cpu"))
        initial = torch.randn(observed.shape, generator=torch.Generator().manual_seed(17)).requires_grad_()
        step = backend.scheduler.step(0.1 * initial, schedule[0], initial)
        loss = observation_loss(step.pred_original_sample, observed, weights, q).sum()
        gradient, = torch.autograd.grad(loss, initial)
        final_alpha = backend.scheduler.final_alpha_cumprod
        expected = step.prev_sample.detach() - 0.001 * final_alpha.sqrt() * gradient
        with torch.inference_mode():
            sampler = OrthogonalPosteriorSampling(backend, guidance_scale=0.001, max_grad_rms=0)
            actual = sampler(observed, weights, q, torch.Generator().manual_seed(17), verbose=False)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(sampler.last_diagnostics["initial_timestep"], 999)
        self.assertEqual(len(sampler.last_diagnostics["steps"]), 1)
        self.assertEqual(sampler.last_diagnostics["steps"][0]["alpha_bar_next"], final_alpha.item())

    def test_dps_caps_per_image_rms_and_scales_with_next_timestep_alpha(self):
        backend = self.backend(steps=4)
        observed = torch.ones(2, 4, 8, 8)
        weights = torch.tensor([1e-12, 2e3])[:, None].expand(2, 64)
        schedule = backend.inference_timesteps(torch.device("cpu"))
        expected = torch.randn(observed.shape, generator=torch.Generator().manual_seed(24))
        base_step_size = 0.1
        expected_steps = []
        for index, timestep in enumerate(schedule):
            alpha = backend.scheduler.alphas_cumprod[timestep]
            coefficient = (1 - 0.1 * (1 - alpha).sqrt()) / alpha.sqrt()
            # Analytic derivative through epsilon(z)=0.1*z and z0(z).
            gradient = coefficient * (coefficient * expected - observed) * block_map(weights, observed, 8, 8)
            rms = gradient.flatten(1).norm(dim=1) / (gradient[0].numel() ** 0.5)
            clipped_gradient = gradient / rms.clamp_min(1)[:, None, None, None]
            next_alpha = (backend.scheduler.alphas_cumprod[schedule[index + 1]]
                          if index + 1 < len(schedule) else backend.scheduler.final_alpha_cumprod)
            step_size = base_step_size * next_alpha.sqrt()
            proposal = backend.scheduler.step(0.1 * expected, timestep, expected).prev_sample
            expected = proposal - step_size * clipped_gradient
            expected_steps.append(step_size.item())
        sampler = OrthogonalPosteriorSampling(backend, guidance_scale=base_step_size, max_grad_rms=1)
        actual = sampler(observed, weights, torch.eye(64), torch.Generator().manual_seed(24), verbose=False)
        torch.testing.assert_close(actual, expected)
        history = sampler.last_diagnostics["steps"]
        torch.testing.assert_close(torch.tensor([step["guidance_step_size"] for step in history]),
                                   torch.tensor(expected_steps))
        self.assertLess(history[0]["gradient_rms"][0], 1)
        self.assertGreater(history[0]["gradient_rms"][1], 1)
        for step in history:
            self.assertLessEqual(max(step["clipped_gradient_rms"]), 1.000001)
        self.assertLess(history[0]["guidance_step_size"], history[-1]["guidance_step_size"])

    def test_zero_precision_matches_unconditional_full_schedule(self):
        backend = self.backend(steps=4)
        observed = torch.ones(1, 4, 8, 8)
        q = torch.eye(64)
        schedule = backend.inference_timesteps(torch.device("cpu"))
        expected = torch.randn(observed.shape, generator=torch.Generator().manual_seed(9))
        for timestep in schedule:
            expected = backend.scheduler.step(0.1 * expected, timestep, expected).prev_sample
        sampler = OrthogonalPosteriorSampling(backend)
        actual = sampler(observed, torch.zeros(1, 64), q,
                         torch.Generator().manual_seed(9), verbose=False)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(sampler.last_diagnostics["timesteps"], [999, 749, 499, 249])

    def test_imagenet_inference_writes_estimated_observations_and_dps_outputs(self):
        self.check_imagenet_inference_outputs("dps")

    def test_imagenet_inference_writes_estimated_observations_and_ddrm_outputs(self):
        self.check_imagenet_inference_outputs("ddrm")

    def test_imagenet_inference_connects_estimator_to_changed_allocator(self):
        self.check_imagenet_inference_outputs("ddrm", changed_allocator=True, uniform_receiver_mode="gain_estimator")

    def test_imagenet_oracle_and_estimator_share_awgn_transmission(self):
        self.check_imagenet_inference_outputs("ddrm", compare_oracle=True)

    def test_imagenet_oracle_and_estimator_share_rayleigh_transmission(self):
        self.check_imagenet_inference_outputs("ddrm", compare_oracle=True, channel_model="rayleigh")

    def test_uniform_oracle_can_be_compared_with_allocated_estimator(self):
        self.check_imagenet_inference_outputs("ddrm", uniform_receiver_mode="oracle")

    def test_uniform_estimator_can_be_compared_with_allocated_oracle(self):
        self.check_imagenet_inference_outputs("ddrm", receiver_mode="oracle", uniform_receiver_mode="gain_estimator")

    def check_imagenet_inference_outputs(self, method, changed_allocator=False, compare_oracle=False,
                                        channel_model="awgn", receiver_mode=None, uniform_receiver_mode=None):
        from inference_orthogonal_power import run_inference

        class TinyVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(latent_channels=4)

            def encode(self, image):
                rgb = torch.nn.functional.avg_pool2d(image, 8)
                latent = torch.cat((rgb, rgb.mean(1, keepdim=True)), 1)
                posterior = DiagonalGaussianDistribution(torch.cat((latent, -2 * torch.ones_like(latent)), 1))
                return SimpleNamespace(latent_dist=posterior)

            def decode(self, latent):
                return SimpleNamespace(sample=torch.nn.functional.interpolate(latent[:, :3], scale_factor=8))

        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=4)
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=0.03)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=0.3)
        classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(256, 7))
        estimator = PatchGainEstimator(4, 1, 1, 8)
        config = {"dataset": "ImageNet", "orthogonal_transform": True, "num_groups": 64,
                  "transmission_mode": "analog", "pilots_per_group": 0, "hidden_dim": 8,
                  "channel_model": RAYLEIGH_CHANNEL_MODEL,
                  "allocator_architecture": ALLOCATOR_ARCHITECTURE, "allocator_block_dim": 4,
                  "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
                  "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE,
                  "image_size": 64, "model": "tiny", "num_classes": 7,
                  "vae_model_id": "local-test", "vae_model_revision": "local-test", "snr_db": [5.]}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            transmitter_path, estimator_path = directory / "power.pth", directory / "gain.pth"
            checkpoint = {"allocator": allocator.state_dict(), "classifier": classifier.state_dict(),
                          "config": config, "epoch": 0}
            torch.save(checkpoint, transmitter_path)
            torch.save({"estimator": estimator.state_dict(), "config": {
                "stage": "block_gain_estimator", "estimator_version": GAIN_ESTIMATOR_VERSION,
                "estimator_architecture": GAIN_ESTIMATOR_ARCHITECTURE,
                "channel_model": RAYLEIGH_CHANNEL_MODEL,
                "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
                "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE,
                "allocator_fingerprint": allocator_fingerprint(allocator), "latent_channels": 4,
                "patch_shape": [1, 1], "hidden_dim": 8,
            }}, estimator_path)
            if changed_allocator:
                with torch.no_grad():
                    allocator.orthogonal.upper[0] += .1
                    allocator.block_allocator.logit_head[-1].weight.add_(.1)
                checkpoint["allocator"] = allocator.state_dict()
                torch.save(checkpoint, transmitter_path)
            args = SimpleNamespace(device="cpu", receiver_mode=receiver_mode,
                                   gain_estimator_checkpoint=str(estimator_path), diffusion_steps=2,
                                   snr_db=5., channel_gain_abs=None, local_files_only=True,
                                   vae_path=None, diffusion_model=None, guidance_scale=0.1,
                                   guidance_max_rms=1., posterior_eta=0., output_dir=str(directory / "output"),
                                   sample_index=0, seed=5, input_image=None, lpips=False,
                                   restoration_method=method, ddrm_eta=.85, ddrm_eta_b=1.)
            if channel_model == "rayleigh":
                args.channel_model = channel_model
            if uniform_receiver_mode is not None:
                args.uniform_receiver_mode = uniform_receiver_mode
            image = torch.rand(1, 3, 64, 64)
            with patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                    patch("inference_orthogonal_power.build_latent_classifier", return_value=classifier), \
                    patch("inference_orthogonal_power.StaggeredReverseDiffusion", return_value=self.backend(2)), \
                    patch("inference_orthogonal_power.load_input", return_value=(image, 1, "test.png")):
                run_inference(args, transmitter_path, checkpoint)
                if compare_oracle:
                    oracle_args = SimpleNamespace(**vars(args))
                    oracle_args.receiver_mode = "oracle"
                    oracle_args.output_dir = str(directory / "oracle")
                    run_inference(oracle_args, transmitter_path, checkpoint)
            output = Path(args.output_dir)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["restoration_method"], method)
            self.assertEqual(result["max_timestep"], 999 if method == "ddrm" else None)
            expected_modes = {"allocated": receiver_mode or "gain_estimator",
                              "uniform": uniform_receiver_mode or "oracle"}
            uses_estimator = "gain_estimator" in expected_modes.values()
            self.assertEqual(result["receiver_mode"], expected_modes["allocated"])
            self.assertEqual(result["uniform_receiver_mode"], expected_modes["uniform"])
            self.assertEqual(result["allocator_matches_training"], not changed_allocator if uses_estimator else None)
            self.assertEqual(result["uses_gain_estimator"], uses_estimator)
            self.assertEqual(result["channel_model"], RAYLEIGH_CHANNEL_MODEL if channel_model == "rayleigh" else "awgn")
            shared_h = None
            shared_noise = None
            for strategy in ("allocated", "uniform"):
                self.assertIn(f"{method}_vs_input", result["strategies"][strategy])
                mode = expected_modes[strategy]
                self.assertEqual(result["strategies"][strategy]["receiver_mode"], mode)
                if method == "ddrm":
                    self.assertEqual(result["strategies"][strategy]["diffusion"]["initialization_policy"],
                                     "clip")
                self.assertAlmostEqual(result["strategies"][strategy]["frame_energy"], 256., places=4)
                self.assertAlmostEqual(result["strategies"][strategy]["mean_real_component_power"], 1., places=5)
                self.assertAlmostEqual(result["strategies"][strategy]["mean_symbol_power"], 2., places=5)
                self.assertEqual(result["strategies"][strategy]["diffusion"]["initial_timestep"], 999)
                self.assertEqual(len(result["strategies"][strategy]["precision"]), 64)
                payload = torch.load(output / f"{strategy}_baseband.pt", weights_only=True)
                self.assertEqual(payload["receiver_mode"], mode)
                if mode == "oracle":
                    torch.testing.assert_close(payload["estimated_gain"], payload["group_power"].sqrt())
                    self.assertIn("oracle_amplitude_gain", payload)
                else:
                    torch.testing.assert_close(payload["estimated_gain"], torch.ones_like(payload["estimated_gain"]))
                    self.assertNotIn("oracle_amplitude_gain", payload)
                self.assertEqual(payload["channel_coefficient"].shape, (1,))
                if shared_h is None:
                    shared_h = payload["channel_coefficient"]
                torch.testing.assert_close(payload["channel_coefficient"], shared_h)
                if channel_model == "awgn":
                    torch.testing.assert_close(shared_h, torch.ones_like(shared_h))
                if strategy == "uniform":
                    torch.testing.assert_close(payload["Q"], torch.eye(64), atol=0, rtol=0)
                    torch.testing.assert_close(payload["transformed_latent"], payload["clean_latent"], atol=0, rtol=0)
                    torch.testing.assert_close(payload["group_power"], payload["group_power"][:, :1].expand(-1, 64))
                torch.testing.assert_close(
                    payload["precision"], shared_h.abs().square()[:, None]
                    * payload["estimated_gain"].square() / payload["N0"][:, None],
                )
                from visualization_utils import noise_variance_to_diffusion_timestep
                true_variance = (shared_h.abs().square()[:, None] * payload["group_power"]
                                 / payload["N0"][:, None]).reciprocal()
                torch.testing.assert_close(payload["true_timestep"], noise_variance_to_diffusion_timestep(true_variance))
                torch.testing.assert_close(payload["estimated_timestep"],
                                           noise_variance_to_diffusion_timestep(payload["precision"].reciprocal()))
                self.assertEqual(result["strategies"][strategy]["true_timestep"], payload["true_timestep"][0].tolist())
                self.assertEqual(result["strategies"][strategy]["estimated_timestep"], payload["estimated_timestep"][0].tolist())
                if mode == "oracle":
                    torch.testing.assert_close(payload["true_timestep"], payload["estimated_timestep"], rtol=0, atol=0)
                transmitted = block_map(payload["group_power"].sqrt(),
                                        payload["transformed_latent"], 8, 8) * payload["transformed_latent"]
                noise = payload["raw_received_latent"] - apply_complex_channel(transmitted, shared_h)
                if shared_noise is None:
                    shared_noise = noise
                torch.testing.assert_close(noise, shared_noise, rtol=1e-5, atol=1e-6)
                channel_corrected = apply_complex_channel(payload["raw_received_latent"],
                                                           payload["channel_coefficient"].reciprocal())
                torch.testing.assert_close(payload["equalized_observation"],
                                           channel_corrected / block_map(
                                               payload["estimated_gain"],
                                               payload["raw_received_latent"], 8, 8))
                if compare_oracle:
                    oracle_payload = torch.load(directory / "oracle" / f"{strategy}_baseband.pt", weights_only=True)
                    for key in ("clean_latent", "Q", "transformed_latent", "channel_coefficient", "epsilon",
                                "raw_received_latent", "group_power", "N0"):
                        torch.testing.assert_close(payload[key], oracle_payload[key], atol=0, rtol=0)
                    torch.testing.assert_close(oracle_payload["estimated_gain"], oracle_payload["group_power"].sqrt())
                    expected_oracle = apply_complex_channel(oracle_payload["raw_received_latent"], shared_h.reciprocal())
                    expected_oracle /= block_map(oracle_payload["group_power"].sqrt(), expected_oracle, 8, 8)
                    torch.testing.assert_close(oracle_payload["equalized_observation"], expected_oracle)
            self.assertTrue((output / "comparison.png").is_file())
            self.assertTrue((output / "reconstructed.png").is_file())
            self.assertTrue((output / "power_allocation.png").is_file())

    def test_uniform_estimator_requires_checkpoint_before_loading_models(self):
        from inference_orthogonal_power import run_inference
        args = SimpleNamespace(device="cpu", receiver_mode="oracle", uniform_receiver_mode="gain_estimator",
                               gain_estimator_checkpoint=None)
        with patch("inference_orthogonal_power.load_transmitter", return_value=(None, None, {}, Path("power.pth"))), \
                patch("inference_orthogonal_power.AutoencoderKL.from_pretrained") as vae_loader:
            with self.assertRaisesRegex(ValueError, "gain_estimator_checkpoint.*required"):
                run_inference(args, Path("power.pth"), {})
        vae_loader.assert_not_called()

    def test_unified_cli_defaults_and_legacy_q_dispatch(self):
        from inference_orthogonal_power import build_parser
        import inference_power

        args = build_parser().parse_args(["--checkpoint", "power.pth"])
        self.assertEqual(args.channel_model, "awgn")
        self.assertEqual(args.restoration_method, "ddrm")
        self.assertEqual(args.uniform_receiver_mode, "oracle")
        self.assertIsNone(build_parser(include_legacy=True).parse_args(["--checkpoint", "power.pth"]).uniform_receiver_mode)
        self.assertEqual(build_parser().parse_args(["--checkpoint", "power.pth", "--uniform_receiver_mode", "oracle"]).uniform_receiver_mode, "oracle")
        checkpoint = {"config": {"orthogonal_transform": True}}
        with patch("sys.argv", ["inference_power.py", "--checkpoint", "power.pth", "--device", "cpu"]), \
                patch("inference_power.torch.load", return_value=checkpoint) as loader, \
                patch("inference_orthogonal_power.run_inference") as runner:
            inference_power.main()
        self.assertEqual(loader.call_count, 1)
        self.assertIs(runner.call_args.args[2], checkpoint)
        self.assertEqual(runner.call_args.args[0].restoration_method, "ddrm")
        self.assertEqual(runner.call_args.args[0].channel_model, "awgn")


if __name__ == "__main__":
    unittest.main()
