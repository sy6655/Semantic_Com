"""Gain comparison, paired SNR sweep and plotting without pretrained downloads."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image

from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, GAIN_ESTIMATOR_ARCHITECTURE, GAIN_ESTIMATOR_VERSION,
    MEAN_REAL_COMPONENT_POWER, NOISE_REAL_COMPONENT_VARIANCE, RAYLEIGH_CHANNEL_MODEL,
    OrthogonalPatchTransform, PatchGainEstimator, allocator_fingerprint,
    apply_complex_channel, block_map, load_gain_estimator, sample_raw_channel,
)
from power_utils import sd15_alpha_bars
from train_power_allocator import PowerAllocator
from visualize_gain_estimator import GainEvaluationAccumulator, compare_gains, compare_timesteps, parse_arguments, run


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(latent_channels=4)

    def encode(self, images):
        mean = self.encoder(F.avg_pool2d(images, 8))
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(
            torch.cat((mean, -2 * torch.ones_like(mean)), 1)))


class GainVisualizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)

    def test_log_loss_and_gain_precision_relative_errors(self):
        truth = torch.tensor([1., .1, .01], dtype=torch.float64)
        metrics, blocks = compare_gains(truth, truth.log() + math.log(1.1))
        self.assertAlmostEqual(metrics["log_gain_mse"], math.log(1.1) ** 2)
        self.assertAlmostEqual(metrics["gain_relative_mae"], .1)
        self.assertAlmostEqual(metrics["precision_relative_mae"], .21)
        self.assertAlmostEqual(metrics["training_gain_relative_mae"], .1)
        torch.testing.assert_close(torch.tensor(blocks["signed_relative_error_percent"]), torch.full((3,), 10.))
        self.assertEqual(metrics["clipped_prediction_blocks"], 0)

    def test_extreme_predictions_report_receiver_clamp_and_raw_log_error(self):
        metrics, blocks = compare_gains(torch.ones(2), torch.tensor([-30., 30.]))
        self.assertEqual(metrics["clipped_prediction_blocks"], 2)
        self.assertEqual(blocks["predicted_log_gain"], [-30., 30.])
        self.assertEqual(blocks["used_log_gain"], [-20., 20.])
        self.assertEqual(metrics["log_gain_mse"], 900.)
        self.assertAlmostEqual(blocks["estimated_gain"][1], math.exp(20))
        for truth, prediction in ((torch.ones(2), torch.ones(3)), (torch.tensor([0.]), torch.zeros(1)),
                                  (torch.ones(1), torch.tensor([float("nan")]))):
            with self.assertRaises(ValueError):
                compare_gains(truth, prediction)

    def test_precision_is_converted_to_sd15_noise_timestep_and_signed_step_error(self):
        alpha = sd15_alpha_bars().double()
        variances = (1-alpha)/alpha
        def precision_at_step(steps):
            return 1 / torch.stack([(variances[t-1] + variances[t]) / 2 for t in steps])
        metrics, blocks = compare_timesteps(precision_at_step([200, 600, 800]),
                                            precision_at_step([180, 650, 850]))
        self.assertEqual(blocks["true_timestep"], [200, 600, 800])
        self.assertEqual(blocks["estimated_timestep"], [180, 650, 850])
        self.assertEqual(blocks["signed_timestep_error"], [-20, 50, 50])
        self.assertEqual(metrics["timestep_mae"], 40.)
        self.assertAlmostEqual(metrics["timestep_rmse"], math.sqrt(1800))
        self.assertEqual(metrics["timestep_max_error"], 50)

    def test_timestep_saturation_is_flagged_instead_of_hiding_excess_noise(self):
        alpha = sd15_alpha_bars().double()
        variances = (1-alpha)/alpha
        true_variance = torch.tensor([float(variances[0]) / 2, 1., float(variances[-1]) * 2])
        estimated_variance = torch.tensor([float(variances[0]) / 3, 1., float(variances[-1]) * 4])
        metrics, blocks = compare_timesteps(1/true_variance, 1/estimated_variance)
        self.assertEqual(blocks["true_timestep"][0], 0)
        self.assertEqual(blocks["true_timestep"][-1], 999)
        self.assertEqual(blocks["estimated_timestep"][-1], 999)
        self.assertEqual(blocks["true_timestep_above_schedule"], [False, False, True])
        self.assertEqual(blocks["estimated_timestep_below_schedule"], [True, False, False])
        self.assertEqual(metrics["true_timestep_above_schedule_blocks"], 1)
        self.assertEqual(metrics["estimated_timestep_below_schedule_blocks"], 1)

    def test_larger_gain_estimate_means_lower_noise_timestep(self):
        true_gain = torch.tensor([.5, 1., 2.])
        n0, h_squared = .8, .2
        true_precision = h_squared * true_gain.square() / n0
        metrics, blocks = compare_timesteps(true_precision, 4 * true_precision)
        self.assertTrue(all(value < 0 for value in blocks["signed_timestep_error"]))
        self.assertGreater(metrics["timestep_mae"], 0)

    def test_aggregate_errors_do_not_cancel_and_rmse_is_pooled_over_all_blocks(self):
        alpha = sd15_alpha_bars().double()
        variance = (1-alpha)/alpha
        middle = lambda t: (variance[t-1] + variance[t]) / 2
        truth = torch.full((4, 64), float(1/middle(400)))
        estimated = torch.tensor([float(1/middle(t)) for t in (392, 400, 400, 408)])[:, None].expand(-1, 64)
        true_gain = torch.ones(4, 64)
        predicted_log_gain = .5 * (estimated / truth).log()
        accumulator = GainEvaluationAccumulator()
        for selection in (slice(0, 3), slice(3, 4)):
            accumulator.add(true_gain[selection], predicted_log_gain[selection], truth[selection], estimated[selection],
                            torch.zeros(4)[selection], torch.full((4,), 256.)[selection], 256)
        metrics, blocks, per_image, values = accumulator.finalize()
        self.assertEqual(metrics["num_images"], 4)
        self.assertEqual(metrics["num_block_entries"], 256)
        self.assertEqual(metrics["timestep_bias"], 0.)
        self.assertEqual(metrics["timestep_mae"], 4.)
        self.assertAlmostEqual(metrics["timestep_rmse"], math.sqrt(32))
        self.assertEqual(per_image["timestep_rmse"], [8., 0., 0., 8.])
        self.assertEqual(blocks["true_timestep"], blocks["estimated_timestep"])
        self.assertEqual(blocks["signed_timestep_error"], [0.] * 64)
        self.assertEqual(blocks["mean_absolute_timestep_error"], [4.] * 64)
        self.assertEqual(values["true_gain"].shape, (4, 64))

    def make_checkpoints(self, directory):
        directory = Path(directory)
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=4)
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=.02)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=.2)
        transmitter_path = directory / "power.pth"
        config = {
            "dataset": "ImageNet", "orthogonal_transform": True, "num_groups": 64,
            "transmission_mode": "analog", "pilots_per_group": 0, "hidden_dim": 8,
            "channel_model": RAYLEIGH_CHANNEL_MODEL, "allocator_architecture": ALLOCATOR_ARCHITECTURE,
            "allocator_block_dim": 4, "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
            "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE,
            "image_size": 64, "latent_channels": 4, "vae_model_id": "local-test", "vae_model_revision": None,
        }
        torch.save({"allocator": allocator.state_dict(), "config": config, "epoch": 1}, transmitter_path)
        estimator = PatchGainEstimator(4, 1, 1, 8)
        estimator_path = directory / "estimator" / "latest.pth"
        estimator_path.parent.mkdir()
        estimator_config = {
            "stage": "block_gain_estimator", "estimator_version": GAIN_ESTIMATOR_VERSION,
            "estimator_architecture": GAIN_ESTIMATOR_ARCHITECTURE, "channel_model": RAYLEIGH_CHANNEL_MODEL,
            "mean_real_component_power": MEAN_REAL_COMPONENT_POWER,
            "noise_real_component_variance": NOISE_REAL_COMPONENT_VARIANCE,
            "allocator_fingerprint": allocator_fingerprint(allocator), "latent_channels": 4,
            "patch_shape": [1, 1], "hidden_dim": 8, "allocator_checkpoint": str(transmitter_path),
        }
        torch.save({"estimator": estimator.state_dict(), "config": estimator_config, "epoch": 3}, estimator_path)
        image_path = directory / "input.png"
        Image.new("RGB", (64, 64), (60, 100, 160)).save(image_path)
        return estimator_path, transmitter_path, image_path, allocator

    def test_snr_sweep_loads_bound_checkpoint_shares_inputs_and_saves_block_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator, transmitter, image, allocator = self.make_checkpoints(directory)
            output = Path(directory) / "plots"
            args = parse_arguments(["--gain_estimator_checkpoint", str(estimator.parent), "--input_image", str(image),
                                    "--snr_db", "0", "10", "--device", "cpu", "--output_dir", str(output)])
            self.assertIsNone(args.max_timestep)
            self.assertEqual(args.display, "timestep")
            self.assertEqual(args.channel_model, "awgn")
            vae = TinyVAE()
            with patch("visualize_gain_estimator.AutoencoderKL.from_pretrained", return_value=vae):
                result = run(args)
            self.assertTrue(result["allocation_matches_training"])
            self.assertTrue(result["same_image_latent_h_and_standard_noise_across_snrs"])
            self.assertEqual(result["channel_model"], "awgn")
            self.assertEqual(result["h"], {"real": 1., "imag": 0., "abs": 1.})
            self.assertEqual(result["estimator_epoch"], 4)
            self.assertEqual(result["allocator_checkpoint"], str(transmitter))
            self.assertTrue(result["allocator_matches_training"])
            self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
            state = torch.load(output / "observations.pt", weights_only=True)
            transformed = OrthogonalPatchTransform.apply_matrix(state["latent"], state["Q"].T)
            for snr in ("0", "10"):
                case, tensors = result["cases"][snr], state["cases"][snr]
                self.assertAlmostEqual(case["effective_snr_db"], float(snr), places=5)
                blocks = case["blocks"]
                self.assertEqual(blocks["block_id"], list(range(64)))
                torch.testing.assert_close(torch.tensor(blocks["estimated_gain"]), torch.ones(64))
                torch.testing.assert_close(torch.tensor(blocks["true_gain"]), tensors["group_power"][0].sqrt())
                applied = transformed * block_map(tensors["group_power"].sqrt(), transformed, 8, 8)
                noiseless = apply_complex_channel(applied, state["h"])
                recovered_noise = (tensors["raw_received_latent"] - noiseless) / tensors["N0"].sqrt()[:, None, None, None]
                torch.testing.assert_close(recovered_noise, state["standard_noise"], atol=5e-6, rtol=5e-6)
                self.assertAlmostEqual(case["frame_energy"], state["latent"][0].numel(), places=3)
                expected_precision = state["h"].abs().square()[:, None] * tensors["group_power"] / tensors["N0"][:, None]
                torch.testing.assert_close(tensors["true_precision"], expected_precision)
                folder = output / case["directory"]
                for filename in ("timestep_heatmaps.png", "timestep_by_block.png", "blocks.csv"):
                    self.assertTrue((folder / filename).is_file(), filename)
                with (folder / "blocks.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 64)
                self.assertEqual(rows[-1]["block_id"], "63")
                self.assertIn("true_timestep", rows[0])
                self.assertIn("signed_timestep_error", rows[0])
                self.assertEqual(len(blocks["true_timestep"]), 64)
                self.assertTrue(all(0 <= value <= 999 for value in blocks["estimated_timestep"]))
                self.assertGreaterEqual(case["metrics"]["timestep_mae"], 0)
                with Image.open(folder / "timestep_heatmaps.png") as rendered:
                    self.assertGreater(rendered.width, 1000)
            self.assertAlmostEqual(result["cases"]["10"]["effective_snr_db"] - result["cases"]["0"]["effective_snr_db"], 10., places=5)
            self.assertTrue((output / "snr_comparison.png").is_file())
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(set(saved["cases"]), {"0", "10"})
            self.assertFalse(saved["uses_reverse_diffusion"])
            self.assertEqual(saved["display"], "timestep")
            self.assertEqual(saved["h"]["abs"], state["h"].abs().item())
            # Dedicated VAE/channel generators make the comparison reproducible
            # even when the process-global RNG state has changed.
            torch.manual_seed(9876)
            args.output_dir = str(Path(directory) / "replay")
            with patch("visualize_gain_estimator.AutoencoderKL.from_pretrained", return_value=vae), \
                    patch("visualize_gain_estimator.save_plots"):
                replay = run(args)
            self.assertEqual(replay["h"], result["h"])
            for key in result["cases"]:
                self.assertEqual(replay["cases"][key]["blocks"], result["cases"][key]["blocks"])

    def test_optional_timestep_floor_uses_actual_transmitted_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator, transmitter, image, allocator = self.make_checkpoints(directory)
            args = parse_arguments(["--gain_estimator_checkpoint", str(estimator), "--allocator_checkpoint", str(transmitter),
                                    "--input_image", str(image), "--snr_db", "0", "--max_timestep", "650",
                                    "--channel_model", "rayleigh",
                                    "--device", "cpu", "--output_dir", str(Path(directory) / "floor")])
            with patch("visualize_gain_estimator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                    patch("visualize_gain_estimator.save_plots"):
                result = run(args)
            self.assertEqual(result["max_timestep"], 650)
            self.assertEqual(result["channel_model"], RAYLEIGH_CHANNEL_MODEL)
            self.assertNotEqual(result["h"]["abs"], 1.)
            self.assertFalse(result["allocation_matches_training"])
            case = result["cases"]["0"]
            self.assertEqual(case["gain_floor_infeasible_images"], 0)
            variance = torch.tensor(case["blocks"]["true_precision"]).reciprocal()
            alpha = sd15_alpha_bars()[650]
            self.assertTrue((variance <= (1-alpha)/alpha + 1e-5).all())

    def test_changed_allocator_is_visualized_with_its_actual_gain_as_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator, transmitter, image, allocator = self.make_checkpoints(directory)
            checkpoint = torch.load(transmitter, map_location="cpu", weights_only=False)
            checkpoint["allocator"]["orthogonal.upper"][0] += .2
            checkpoint["allocator"]["block_allocator.logit_head.2.weight"] += .1
            allocator.load_state_dict(checkpoint["allocator"])
            torch.save(checkpoint, transmitter)
            with self.assertRaisesRegex(ValueError, "different Q/allocator"):
                load_gain_estimator(estimator, allocator, "cpu")
            output = Path(directory) / "changed_allocator"
            args = parse_arguments(["--gain_estimator_checkpoint", str(estimator), "--input_image", str(image),
                                    "--snr_db", "0", "--device", "cpu", "--output_dir", str(output)])
            with patch("visualize_gain_estimator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                    patch("visualize_gain_estimator.save_plots"):
                result = run(args)
            self.assertFalse(result["allocator_matches_training"])
            self.assertFalse(result["allocation_matches_training"])
            self.assertTrue(result["allocation_policy_matches_training"])
            self.assertEqual(result["allocator_fingerprint"], allocator_fingerprint(allocator))
            self.assertNotEqual(result["allocator_fingerprint"], result["training_allocator_fingerprint"])
            state = torch.load(output / "observations.pt", weights_only=True)
            torch.testing.assert_close(state["Q"], allocator.orthogonal.matrix())
            torch.testing.assert_close(torch.tensor(result["cases"]["0"]["blocks"]["true_gain"]),
                                       state["cases"]["0"]["group_power"][0].sqrt())

    def test_multi_image_evaluation_encodes_once_and_reuses_selection_and_noise_for_snrs(self):
        class Images(torch.utils.data.Dataset):
            def __init__(self, root):
                self.classes = ["a", "b", "c", "d"]
                self.targets = [label for label in range(4) for _ in range(3)]
                self.samples = [(str(root / self.classes[label] / f"{index}.png"), label)
                                for index, label in enumerate(self.targets)]
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, index):
                return torch.full((3, 64, 64), (index + 1) / 20), self.targets[index]

        class RecordingVAE(TinyVAE):
            def __init__(self):
                super().__init__()
                self.seen, self.calls = [], 0
            def encode(self, images):
                self.calls += 1
                self.seen.extend(((images[:, 0, 0, 0] + 1) * 10 - 1).round().long().tolist())
                return super().encode(images)

        with tempfile.TemporaryDirectory() as directory:
            estimator, _, _, _ = self.make_checkpoints(directory)
            validation_root = Path(directory) / "val"
            validation_root.mkdir()
            dataset, vae = Images(validation_root), RecordingVAE()
            output = Path(directory) / "aggregate"
            args = parse_arguments(["--gain_estimator_checkpoint", str(estimator), "--data_path", directory,
                                    "--num_samples", "9", "--batch_size", "4", "--num_workers", "0",
                                    "--snr_db", "0", "10", "--device", "cpu", "--output_dir", str(output)])
            channel_calls = []
            def transmit(signal, power, n0, h, **kwargs):
                channel_calls.append((signal.clone(), kwargs["epsilon"].clone(), h.clone()))
                return sample_raw_channel(signal, power, n0, h, **kwargs)
            with patch("visualize_gain_estimator.AutoencoderKL.from_pretrained", return_value=vae), \
                    patch("visualize_gain_estimator.datasets.ImageFolder", return_value=dataset), \
                    patch("visualize_gain_estimator.sample_raw_channel", side_effect=transmit):
                result = run(args)
            indices = result["sample_selection"]["indices"]
            self.assertEqual(len(indices), 9)
            self.assertEqual(len(set(indices)), 9)
            self.assertEqual(vae.seen, indices)
            self.assertEqual(vae.calls, 3)
            self.assertEqual(len(channel_calls), 6)
            for position in range(0, 6, 2):
                for first, second in zip(channel_calls[position], channel_calls[position+1]):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
                torch.testing.assert_close(channel_calls[position][2], torch.ones_like(channel_calls[position][2]))
            counts = list(result["sample_selection"]["class_counts"].values())
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual(result["h"], {"real": 1., "imag": 0., "abs": 1.})
            self.assertTrue(result["complete"])
            self.assertFalse(result["uses_reverse_diffusion"])
            state = torch.load(output / "sample_values.pt", weights_only=True)
            self.assertEqual(state["indices"].tolist(), indices)
            for key, case in result["cases"].items():
                values = state["cases"][key]
                self.assertEqual(values["true_gain"].shape, (9, 64))
                expected_gain, _ = compare_gains(values["true_gain"].flatten(), values["predicted_log_gain"].flatten())
                expected_t, _ = compare_timesteps(values["true_precision"].flatten(), values["estimated_precision"].flatten())
                self.assertAlmostEqual(case["metrics"]["gain_relative_mae"], expected_gain["gain_relative_mae"])
                self.assertAlmostEqual(case["metrics"]["timestep_rmse"], expected_t["timestep_rmse"])
                self.assertEqual(case["metrics"]["num_block_entries"], 9*64)
                self.assertAlmostEqual(case["mean_real_component_power"], 1., places=5)
                for name in ("timestep_heatmaps.png", "timestep_by_block.png", "blocks.csv"):
                    self.assertTrue((output / case["directory"] / name).is_file())
            with (output / "per_image_metrics.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 18)
            for snr in (0., 10.):
                self.assertEqual([int(row["index"]) for row in rows if float(row["snr_db"]) == snr], indices)
            self.assertTrue((output / "snr_comparison.png").is_file())
            self.assertEqual(json.loads((output / "result.json").read_text())["num_samples"], 9)


if __name__ == "__main__":
    unittest.main()
