import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import evaluate_power_maxvit as evaluation
from evaluate_power_maxvit import evaluate, select_balanced_indices
from inference_power import allocate_power_for_inference
from orthogonal_power import gain_estimator_receive, prepare_gain_estimator_input
from posterior_sampling import resolve_max_timestep
from power_utils import sd15_alpha_bars
from train_power_allocator import PowerAllocator, RayleighFadingChannel


class LabeledImages(Dataset):
    """ImageFolder-compatible metadata without reading images or weights."""

    def __init__(self, counts, root="/tmp/imagenet"):
        self.classes = [f"class_{index:04d}" for index in range(len(counts))]
        self.targets = [label for label, count in enumerate(counts) for _ in range(count)]
        self.samples = [(str(Path(root) / self.classes[label] / f"image_{index:06d}.jpg"), label)
                        for index, label in enumerate(self.targets)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 2, 2), float(index)), self.targets[index]


class BalancedEvaluationTests(unittest.TestCase):
    def test_default_selects_five_unique_images_per_imagenet_class(self):
        dataset = LabeledImages([50] * 1000)
        indices = select_balanced_indices(dataset, seed=17)
        self.assertEqual(len(indices), 5000)
        self.assertEqual(len(set(indices)), 5000)
        self.assertEqual(indices, sorted(indices))
        counts = torch.bincount(torch.tensor([dataset.targets[index] for index in indices]), minlength=1000)
        torch.testing.assert_close(counts, torch.full((1000,), 5))

    def test_selection_replays_seed_without_consuming_global_rng(self):
        dataset = LabeledImages([50] * 1000)
        torch.manual_seed(21)
        state = torch.get_rng_state().clone()
        first = select_balanced_indices(dataset, seed=3)
        replay = select_balanced_indices(dataset, seed=3)
        other = select_balanced_indices(dataset, seed=4)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, other)
        torch.testing.assert_close(torch.get_rng_state(), state)

    def test_nondivisible_and_small_limits_stay_balanced(self):
        dataset = LabeledImages([50] * 1000)
        for limit in (100, 5001):
            with self.subTest(limit=limit):
                indices = select_balanced_indices(dataset, max_samples=limit, seed=0)
                counts = torch.bincount(torch.tensor([dataset.targets[index] for index in indices]), minlength=1000)
                self.assertEqual(len(indices), limit)
                self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(select_balanced_indices(dataset, None), select_balanced_indices(dataset))

    def test_explicit_full_validation_count_selects_every_image(self):
        dataset = LabeledImages([50] * 1000)
        self.assertEqual(select_balanced_indices(dataset, 50000), list(range(50000)))

    def test_insufficient_class_images_raise_instead_of_biasing_selection(self):
        dataset = LabeledImages([4, 10])
        with self.assertRaisesRegex(ValueError, "class_0000.*requires 5"):
            select_balanced_indices(dataset, max_samples=10)
        with self.assertRaisesRegex(ValueError, "positive"):
            select_balanced_indices(dataset, max_samples=0)

    def test_all_strategies_and_snrs_share_the_selected_images(self):
        self.check_shared_evaluation_images("dps", None)

    def test_ddrm_defaults_to_maximum_timestep_floor_in_evaluation(self):
        self.check_shared_evaluation_images("ddrm", None)

    def test_explicit_ddrm_timestep_floor_is_forwarded_and_saved(self):
        self.check_shared_evaluation_images("ddrm", 650)

    def test_gain_estimator_evaluation_keeps_uniform_oracle_and_shared_samples(self):
        self.check_shared_evaluation_images("ddrm", None, receiver_mode="gain_estimator")

    def test_gain_estimator_also_supports_dps(self):
        self.check_shared_evaluation_images("dps", None, receiver_mode="gain_estimator")

    def check_shared_evaluation_images(self, method, max_timestep, receiver_mode="oracle"):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LabeledImages([10] * 10, root=directory)
            config = {"vae_model_id": "local-test", "vae_model_revision": None, "image_size": 2}
            args = SimpleNamespace(
                checkpoint=str(Path(directory) / "latest.pth"), data_path=directory,
                local_files_only=True, vae_path=None, diffusion_model=None,
                diffusion_steps=2, max_samples=50, seed=7, batch_size=7,
                num_workers=0, snr_db=[0., 10.], restoration_method=method,
                max_timestep=max_timestep,
                guidance_scale=.1, guidance_max_rms=1., posterior_eta=0.,
                log_interval=100, output_path=str(Path(directory) / "result.json"),
                receiver_mode=receiver_mode,
                gain_estimator_checkpoint=(str(Path(directory) / "estimator.pth")
                                           if receiver_mode == "gain_estimator" else None),
            )
            estimator = torch.nn.Identity() if receiver_mode == "gain_estimator" else None
            estimator_config = {"allocator_fingerprint": "different-training-Q"}
            expected = select_balanced_indices(dataset, max_samples=50, seed=7)
            seen = [[], [], []]
            calls = []
            received_indices = []
            expected_floor = 999 if method == "ddrm" and max_timestep is None else max_timestep

            def accumulate(stats, reconstructed, reference, labels, task_model, lpips_model):
                ids = reference[:, 0, 0, 0].to(dtype=torch.long).tolist()
                torch.testing.assert_close(reconstructed, reference)
                self.assertEqual(labels.tolist(), [dataset.targets[index] for index in ids])
                seen[len(calls) % 3].extend(ids)
                calls.append(ids)
                stats[0] += len(ids)

            def restore(latent, *unused_args, **unused_kwargs):
                self.assertEqual(unused_kwargs["max_timestep"], expected_floor)
                self.assertEqual(unused_kwargs["receiver_mode"], receiver_mode)
                self.assertIs(unused_kwargs["gain_estimator"], estimator)
                if method == "ddrm":
                    self.assertEqual(unused_kwargs["uniform_sampler"].initialization_policy, "clip")
                    self.assertEqual(unused_args[3].initialization_policy,
                                     "error" if receiver_mode == "gain_estimator" else "clip")
                received_indices.extend(latent[:, 0, 0, 0].to(dtype=torch.long).tolist())
                batch = latent.shape[0]
                if method == "ddrm":
                    clip_images = int((latent[:, 0, 0, 0].to(dtype=torch.long) % 2 == 0).sum())
                    for strategy, blocks_per_image in (("uniform", 64), ("allocated", 16)):
                        unused_kwargs["diagnostics"][strategy] = {
                            "initialization_clipped_images": clip_images,
                            "initialization_clipped_block_count": clip_images * blocks_per_image,
                        }
                return ({"uniform": latent.clone(), "allocated": latent.clone()},
                        torch.ones(batch, dtype=torch.complex64), torch.zeros(batch, dtype=torch.bool))

            with patch("evaluate_power_maxvit.load_transmitter", return_value=(None, None, config, None)), \
                    patch("evaluate_power_maxvit.load_gain_estimator", return_value=(estimator, estimator_config)) as load_estimator, \
                    patch("evaluate_power_maxvit.load_lpips", return_value=None), \
                    patch("evaluate_power_maxvit.AutoencoderKL.from_pretrained", return_value=torch.nn.Identity()), \
                    patch("evaluate_power_maxvit.StaggeredReverseDiffusion", return_value=None), \
                    patch("evaluate_power_maxvit.build_restoration_sampler", wraps=evaluation.build_restoration_sampler), \
                    patch("evaluate_power_maxvit.construct_model", return_value=torch.nn.Identity()), \
                    patch("evaluate_power_maxvit.datasets.ImageFolder", return_value=dataset), \
                    patch("evaluate_power_maxvit.encode_latents", side_effect=lambda images, *args: images), \
                    patch("evaluate_power_maxvit._decode_latent", side_effect=lambda vae, latent, scale: latent), \
                    patch("evaluate_power_maxvit.accumulate", side_effect=accumulate), \
                    patch("evaluate_power_maxvit.restore_transmissions", side_effect=restore):
                result = evaluate(args, torch.device("cpu"))

            if receiver_mode == "gain_estimator":
                load_estimator.assert_called_once_with(Path(args.gain_estimator_checkpoint), None,
                                                       torch.device("cpu"), check_allocator_fingerprint=False)
                self.assertEqual(result["gain_estimator_config"], estimator_config)
                self.assertEqual(result["gain_estimator_checkpoint"], args.gain_estimator_checkpoint)
            else:
                load_estimator.assert_not_called()
                self.assertIsNone(result["gain_estimator_checkpoint"])
            self.assertEqual(result["receiver_mode"], receiver_mode)
            self.assertEqual(result["uniform_receiver_mode"], "oracle")
            self.assertEqual(result["uses_gain_estimator"], receiver_mode == "gain_estimator")
            self.assertEqual(result["strategy_receiver_modes"], {"uniform": "oracle", "allocated": receiver_mode})
            if method == "ddrm":
                self.assertEqual(result["ddrm_initialization_policies"],
                                 {"uniform": "clip", "allocated": "error" if receiver_mode == "gain_estimator" else "clip"})
            for image_ids in seen:
                self.assertEqual(image_ids, expected * 2)
            self.assertEqual(received_indices, expected * 2)
            self.assertEqual(result["sample_selection"]["indices"], expected)
            self.assertEqual(result["max_timestep"], expected_floor)
            self.assertEqual(set(result["sample_selection"]["class_counts"].values()), {5})
            for snr in (0, 10):
                saved = json.loads((Path(directory) / f"result_snr_{snr}.json").read_text())
                self.assertEqual(saved["max_timestep"], expected_floor)
                selection = saved["sample_selection"]
                self.assertEqual(selection["indices"], expected)
                self.assertEqual(selection["relative_paths"],
                                 [str(Path(dataset.samples[index][0]).relative_to(directory))
                                  for index in expected])
                self.assertEqual(saved["evaluations"][str(snr)]["strategies"]["uniform"]["num_samples"], 50)
                self.assertEqual(saved["evaluations"][str(snr)]["strategies"]["allocated"]["num_samples"], 50)
                clipped = saved["evaluations"][str(snr)]["ddrm_initialization_clipped"]
                expected_images = sum(index % 2 == 0 for index in expected) if method == "ddrm" else 0
                self.assertEqual(clipped["uniform"], {"images": expected_images, "blocks": expected_images * 64})
                self.assertEqual(clipped["allocated"], {"images": expected_images, "blocks": expected_images * 16})


class GainEstimatorReceptionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_estimated_gain_drives_allocated_observation_and_precision_only(self):
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16).eval()
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=.03)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=.3)
        channel = RayleighFadingChannel(8, 8)
        latent, snr = torch.randn(3, 4, 16, 16), torch.tensor([-5., 0., 10.])
        estimator_inputs = []

        def estimator(equalized, log_noise, log_power):
            self.assertFalse(torch.is_grad_enabled())
            estimator_inputs.append((equalized.clone(), log_noise.clone(), log_power.clone()))
            return torch.full((latent.shape[0], 64), .4)

        class RecordingSampler:
            def __init__(self):
                self.calls = []
                self.last_diagnostics = {"initialization_clipped_images": 0}

            def __call__(self, observation, precision, q, generator, **kwargs):
                noise = torch.randn(observation.shape, generator=generator)
                self.calls.append((observation.clone(), precision.clone(), q.clone(), noise))
                return observation.clone()

        uniform_sampler, allocated_sampler = RecordingSampler(), RecordingSampler()
        with patch.object(evaluation, "sample_raw_channel", wraps=evaluation.sample_raw_channel) as raw:
            restored, h, _ = evaluation.restore_transmissions(
                latent, snr, allocator, channel, allocated_sampler,
                torch.Generator().manual_seed(31), torch.Generator().manual_seed(32),
                max_timestep=999, receiver_mode="gain_estimator", gain_estimator=estimator,
                uniform_sampler=uniform_sampler,
            )
        self.assertEqual(len(estimator_inputs), 1)
        self.assertEqual(len(uniform_sampler.calls), 1)
        self.assertEqual(len(allocated_sampler.calls), 1)
        uniform_call, allocated_call = uniform_sampler.calls[0], allocated_sampler.calls[0]
        torch.testing.assert_close(uniform_call[2], torch.eye(64))
        torch.testing.assert_close(allocated_call[2], allocator.orthogonal.matrix())
        torch.testing.assert_close(uniform_call[3], allocated_call[3], rtol=0, atol=0)
        for index in (2, 3):
            torch.testing.assert_close(raw.call_args_list[0].args[index], raw.call_args_list[1].args[index])
        torch.testing.assert_close(raw.call_args_list[0].kwargs["epsilon"], raw.call_args_list[1].kwargs["epsilon"])
        transmitted, power, n0, common_h = raw.call_args_list[1].args
        allocated_raw = evaluation.sample_raw_channel(transmitted, power, n0, common_h,
                                                      epsilon=raw.call_args_list[1].kwargs["epsilon"])
        expected_inputs = prepare_gain_estimator_input(allocated_raw, n0, h)
        for actual, key in zip(estimator_inputs[0], ("channel_equalized_latent", "log_noise_variance", "log_power")):
            torch.testing.assert_close(actual, expected_inputs[key])
        estimated = gain_estimator_receive(allocated_raw, torch.full((3, 64), .4), n0, h)
        truth = evaluation.oracle_receive(allocated_raw, power, n0, h)
        torch.testing.assert_close(allocated_call[0], estimated["equalized_latent"])
        torch.testing.assert_close(allocated_call[1], estimated["precision"])
        self.assertFalse(torch.allclose(allocated_call[1], truth["precision"]))
        torch.testing.assert_close(restored["allocated"], estimated["equalized_latent"])
        baseline_sampler = RecordingSampler()
        baseline, baseline_h, _ = evaluation.restore_transmissions(
            latent, snr, allocator, channel, baseline_sampler,
            torch.Generator().manual_seed(31), torch.Generator().manual_seed(32), max_timestep=999,
        )
        torch.testing.assert_close(h, baseline_h, rtol=0, atol=0)
        torch.testing.assert_close(restored["uniform"], baseline["uniform"], rtol=0, atol=0)
        torch.testing.assert_close(uniform_call[1], baseline_sampler.calls[0][1], rtol=0, atol=0)

    def test_command_line_options_and_auto_selection(self):
        args = evaluation.parse_arguments(["--checkpoint", "/tmp/allocator.pth"])
        self.assertEqual(evaluation.resolve_receiver_mode(args), "oracle")
        args = evaluation.parse_arguments([
            "--checkpoint", "/tmp/allocator.pth", "--gain_estimator_checkpoint", "/tmp/estimator.pth",
            "--restoration_method", "ddrm",
        ])
        self.assertEqual(evaluation.resolve_receiver_mode(args), "gain_estimator")
        args.receiver_mode = "oracle"
        self.assertEqual(evaluation.resolve_receiver_mode(args), "oracle")

    def test_missing_estimator_is_rejected_before_model_loading(self):
        args = SimpleNamespace(restoration_method="ddrm", receiver_mode="gain_estimator",
                               max_timestep=None, gain_estimator_checkpoint=None)
        with patch.object(evaluation, "load_transmitter") as load:
            with self.assertRaisesRegex(ValueError, "gain_estimator_checkpoint"):
                evaluate(args, torch.device("cpu"))
            load.assert_not_called()

    def test_estimator_default_output_does_not_overwrite_oracle(self):
        args = evaluation.parse_arguments([
            "--checkpoint", "/tmp/allocator.pth", "--restoration_method", "ddrm", "--seed", "42",
        ])
        checkpoint = Path(args.checkpoint)
        oracle_path = evaluation.evaluation_output_path(args, checkpoint, 0)
        args.gain_estimator_checkpoint = "/tmp/estimator.pth"
        estimator_path = evaluation.evaluation_output_path(args, checkpoint, 0)
        self.assertEqual(oracle_path.name, "maxvit_ddrm_rayleigh_snr_0_steps_50_eta_0.85_etab_1_seed_42_balanced_n5000_maxt_999.json")
        self.assertNotEqual(oracle_path, estimator_path)
        self.assertEqual(estimator_path.stem, oracle_path.stem + "_receiver_gain_estimator_uniform_oracle")


class MinimumGainFloorTests(unittest.TestCase):
    def test_default_and_explicit_floor_resolution(self):
        for method, supplied, expected in (("ddrm", None, 999), ("ddrm", 650, 650),
                                            ("ddrm", 0, 0), ("dps", None, None), ("dps", 999, 999)):
            with self.subTest(method=method, supplied=supplied):
                args = SimpleNamespace(restoration_method=method, max_timestep=supplied)
                self.assertEqual(resolve_max_timestep(args), expected)

    def test_maximum_noise_floor_preserves_budget_and_meets_initialization_bound(self):
        latent = torch.full((2, 4, 8, 8), .5)
        energy = torch.ones(2, 64)
        base_power = torch.linspace(.01, 5., 64)[None].expand(2, -1)
        base_power = 256 * base_power / base_power.sum(1, keepdim=True)
        n0, h = torch.ones(2), torch.tensor([1.+0j, .05+0j])
        channel = SimpleNamespace(_noise_spectral_density=lambda latent, snr: n0)
        info = {"unscaled_block_energy": energy, "target_frame_energy": 256., "frame_channel_uses": 128}
        with patch("inference_power.allocate_power", return_value=(base_power / 256, base_power, info)):
            ratio, power, updated = allocate_power_for_inference(
                latent, torch.zeros(2), None, channel, channel_coefficient=h, max_timestep=999,
            )
        alpha = sd15_alpha_bars()[999]
        maximum_variance = (1 - alpha) / alpha
        minimum_power = n0[:, None] / (h.abs().square()[:, None] * maximum_variance)
        self.assertTrue((power >= minimum_power).all())
        precision = h.abs().square()[:, None] * power / n0[:, None]
        self.assertTrue((precision.reciprocal() <= maximum_variance).all())
        torch.testing.assert_close((power * energy).sum(1), torch.full((2,), 256.))
        torch.testing.assert_close(ratio.sum(1), torch.ones(2))
        self.assertFalse(updated["gain_floor_fallback"].any())
        self.assertEqual(updated["max_timestep"], 999)

    def test_infeasible_timestep_floor_still_reports_uniform_fallback(self):
        latent, energy = torch.full((1, 4, 8, 8), .5), torch.ones(1, 64)
        base_power = torch.linspace(.01, 5., 64)[None]
        base_power = 256 * base_power / base_power.sum()
        channel = SimpleNamespace(_noise_spectral_density=lambda latent, snr: torch.ones(1))
        info = {"unscaled_block_energy": energy, "target_frame_energy": 256., "frame_channel_uses": 128}
        with patch("inference_power.allocate_power", return_value=(base_power / 256, base_power, info)):
            _, power, updated = allocate_power_for_inference(
                latent, torch.zeros(1), None, channel,
                channel_coefficient=torch.tensor([.001+0j]), max_timestep=999,
            )
        torch.testing.assert_close(power, torch.full((1, 64), 4.))
        self.assertTrue(updated["gain_floor_fallback"].all())


if __name__ == "__main__":
    unittest.main()
