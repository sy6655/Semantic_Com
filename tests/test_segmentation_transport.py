"""Cityscapes transport checks without downloading pretrained networks."""

import copy
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image

from ddrm_sampling import OrthogonalDDRM
from orthogonal_power import (
    ALLOCATOR_ARCHITECTURE, GAIN_ESTIMATOR_ARCHITECTURE, GAIN_ESTIMATOR_VERSION,
    RAYLEIGH_CHANNEL_MODEL, OrthogonalPatchTransform, PatchGainEstimator,
    apply_complex_channel, block_map, load_gain_estimator, patchify, sample_raw_channel,
)
from power_utils import sd15_alpha_bars
from segmentation_orthogonal_power import evaluate, parse_arguments, run_single
from segmentation_transport import (
    SegmentationLayout, build_segmentation_allocator, load_segmentation_transmitter,
    prepare_segmentation_transmissions, restore_segmentation_transmissions,
    validate_segmentation_config,
)
from train_power_allocator import PowerAllocator, RayleighFadingChannel, VAE_SCALE
from train_segmentation_power_allocator import segmentation_step


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(latent_channels=4, scaling_factor=VAE_SCALE)

    def encode(self, images):
        mean = self.encoder(F.interpolate(images, size=(128, 256)))
        posterior = DiagonalGaussianDistribution(torch.cat((mean, -2 * torch.ones_like(mean)), 1))
        return SimpleNamespace(latent_dist=posterior)

    def decode(self, latent):
        return SimpleNamespace(sample=F.interpolate(latent[:, :3], size=(16, 32)))


class TinyTask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(4, 19, 1)

    def forward(self, latent, output_size):
        return F.interpolate(self.head(latent), output_size)


class TinyCityscapes(torch.utils.data.Dataset):
    samples = [(Path(f"test_{index}.png"), None, True) for index in range(3)]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        labels = torch.zeros(16, 32, dtype=torch.long)
        labels[0, 0] = 255
        return torch.full((3, 16, 32), (index + 1) / 5), labels


class TinyMask2Former(torch.nn.Module):
    def forward(self, inputs):
        result = []
        for item in inputs:
            logits = torch.zeros(19, item["height"], item["width"], device=item["image"].device)
            logits[0] = 1
            result.append({"sem_seg": logits})
        return result


def tiny_reverse_model():
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
                              timestep_spacing="trailing", clip_sample=False)

    def schedule(device):
        scheduler.set_timesteps(2, device=device)
        return scheduler.timesteps

    return SimpleNamespace(scheduler=scheduler, dtype=torch.float32, inference_timesteps=schedule,
                           _empty_prompt_embedding=lambda batch: torch.zeros(batch, 1, 1),
                           unet=lambda x, t, encoder_hidden_states: SimpleNamespace(sample=.1*x))


class PairedSampler:
    def __init__(self):
        self.calls = []
        self.last_diagnostics = {}

    def __call__(self, observation, precision, q, generator, rows, cols, verbose):
        random = torch.randn(1, generator=generator)
        self.calls.append((observation.clone(), precision.clone(), q.clone(), random))
        self.last_diagnostics = {"initialization_clipped_images": observation.shape[0],
                                 "initialization_clipped_block_count": observation.shape[0] * rows * cols}
        return OrthogonalPatchTransform.apply_matrix(observation, q, rows, cols)


def checkpoint_config(layout):
    return {
        **layout.config(), "dataset": "Cityscapes", "model": "SegFormer-B0",
        "stage": "segmentation_power_allocator", "segmentation_transport_version": 1,
        "orthogonal_transform": True, "allocator_architecture": ALLOCATOR_ARCHITECTURE,
        "channel_model": RAYLEIGH_CHANNEL_MODEL, "noise_real_component_variance": "N0",
        "mean_real_component_power": 1., "transmission_mode": "analog",
        "receiver_mode": "oracle", "pilots_per_group": 0, "hidden_dim": 8,
        "data_path": "/unused", "vae_model_id": "unused", "vae_model_revision": None,
        "snr_db": [-10., 10.], "temperature": 1.,
    }


class SegmentationTransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(32)

    def tearDown(self):
        self.close_loggers()

    @staticmethod
    def close_loggers():
        for name in ("train_power_allocator", "segmentation_evaluation"):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def allocator(self, layout):
        allocator = build_segmentation_allocator(layout, hidden_dim=8)
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=.01)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=.2)
        return allocator

    def test_default_geometry_keeps_cityscapes_resolution_and_square_q512(self):
        layout = SegmentationLayout()
        self.assertEqual((layout.rows, layout.cols, layout.num_groups, layout.block_dim), (16, 32, 512, 256))
        allocator = self.allocator(layout)
        q = allocator.orthogonal.matrix()
        self.assertEqual(q.shape, (512, 512))
        self.assertEqual(allocator.orthogonal.upper.numel(), 130816)
        torch.testing.assert_close(q.T @ q, torch.eye(512), atol=2e-5, rtol=2e-5)
        latent = torch.randn(1, 4, 128, 256)
        transformed, _ = allocator.orthogonal(latent)
        recovered = OrthogonalPatchTransform.apply_matrix(transformed, q, 16, 32)
        torch.testing.assert_close(recovered, latent, atol=2e-5, rtol=2e-5)
        self.assertEqual(patchify(transformed, 16, 32).shape, (1, 512, 256))
        self.assertEqual(layout.config()["input_shape"], [1024, 2048])
        self.assertEqual(SegmentationLayout(16, 16).num_groups, 128)
        self.assertEqual(SegmentationLayout(32, 32).num_groups, 32)
        with self.assertRaisesRegex(ValueError, "divide"):
            SegmentationLayout(7, 8)

    def test_imagenet_defaults_match_explicit_grid_and_state_keys(self):
        torch.manual_seed(123)
        original = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16)
        torch.manual_seed(123)
        explicit = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16, group_grid=(8, 8))
        self.assertEqual(original.state_dict().keys(), explicit.state_dict().keys())
        for key in original.state_dict():
            torch.testing.assert_close(original.state_dict()[key], explicit.state_dict()[key], atol=0, rtol=0)
        latent, snr = torch.randn(2, 4, 16, 16), torch.tensor([0., 1.])
        torch.testing.assert_close(original(latent, snr), explicit(latent, snr), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "64 blocks"):
            PowerAllocator(4, 512, 8, orthogonal_transform=True)
        with self.assertRaisesRegex(ValueError, "group_grid"):
            PowerAllocator(4, 512, 8, orthogonal_transform=True, group_grid=(16, 16))

    def test_paired_transport_shares_complex_h_noise_and_real_power_budget(self):
        layout = SegmentationLayout(2, 2, 8, 16)
        latent = torch.randn(2, 4, 8, 16)
        allocator = self.allocator(layout)
        h = torch.tensor([.5 + .3j, -.4 + .8j])
        channel = RayleighFadingChannel(layout.rows, layout.cols)
        with patch.object(channel, "sample_channel_coefficient", return_value=h):
            transmission = prepare_segmentation_transmissions(
                latent, torch.tensor([0., 10.]), allocator, channel,
                torch.Generator().manual_seed(7), max_timestep=999,
            )
        self.assertEqual(transmission["h"].shape, (2,))
        for name, received in transmission["strategies"].items():
            self.assertEqual(received["uses_Q"], name == "allocated")
            gain = block_map(received["group_power"].sqrt(), latent, layout.rows, layout.cols)
            noiseless = apply_complex_channel(received["transmitted_latent"] * gain,
                                              h, layout.rows, layout.cols)
            expected_noise = transmission["N0"].sqrt()[:, None, None, None] * transmission["epsilon"]
            torch.testing.assert_close(received["raw_received"] - noiseless, expected_noise, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(received["precision"], h.abs().square()[:, None] *
                                       received["group_power"] / transmission["N0"][:, None])
            torch.testing.assert_close(received["frame_energy"], torch.full((2,), float(latent[0].numel())))
            torch.testing.assert_close(received["energy_allocation_ratio"].sum(1), torch.ones(2))
        uniform = transmission["strategies"]["uniform"]
        torch.testing.assert_close(uniform["group_power"], uniform["group_power"][:, :1].expand(-1, layout.num_groups))
        torch.testing.assert_close(uniform["Q"], torch.eye(layout.num_groups))
        a = sd15_alpha_bars()[999]
        self.assertTrue((transmission["strategies"]["allocated"]["noise_variance"] <= (1-a)/a + 1e-5).all())

    def test_full_resolution_training_updates_q_allocator_and_segmenter(self):
        layout = SegmentationLayout()
        allocator = self.allocator(layout)
        vae = TinyVAE().eval().requires_grad_(False)
        task = TinyTask()
        channel = RayleighFadingChannel(layout.rows, layout.cols)
        with patch.object(channel, "sample_channel_coefficient", return_value=torch.tensor([.8 + .6j])):
            result = segmentation_step(torch.rand(1, 3, 16, 32), torch.zeros(1, 16, 32, dtype=torch.long),
                                       vae, allocator, task, channel, None, torch.tensor([0.]), 1., 999,
                                       torch.Generator().manual_seed(6))
        result["loss"].backward()
        for parameter in (allocator.orthogonal.upper, allocator.block_allocator.logit_head[-1].weight, task.head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
        torch.testing.assert_close(result["mean_real_component_power"], torch.ones(1))
        self.assertEqual(result["group_power"].shape, (1, 512))

    def test_checkpoint_layout_roundtrip_and_legacy_rejection(self):
        layout = SegmentationLayout()
        config = checkpoint_config(layout)
        self.assertEqual(validate_segmentation_config(config), layout)
        for field, wrong in (("group_grid", [8, 8]), ("num_groups", 64),
                             ("noise_real_component_variance", "N0/2"), ("receiver_mode", "pilot"),
                             ("input_shape", [512, 512]), ("orthogonal_transform", False)):
            malformed = copy.deepcopy(config)
            malformed[field] = wrong
            with self.assertRaises(ValueError, msg=field):
                validate_segmentation_config(malformed)
        with self.assertRaisesRegex(ValueError, "legacy_awgn"):
            validate_segmentation_config({"dataset": "Cityscapes"})
        with tempfile.TemporaryDirectory() as directory:
            allocator = self.allocator(layout)
            path = Path(directory) / "latest.pth"
            torch.save({"config": config, "allocator": allocator.state_dict()}, path)
            restored, channel, restored_layout, _, actual_path = load_segmentation_transmitter(directory, torch.device("cpu"))
            self.assertEqual(actual_path, path)
            self.assertEqual(restored_layout, layout)
            self.assertEqual((channel.group_rows, channel.group_cols), (16, 32))
            for key in allocator.state_dict():
                torch.testing.assert_close(allocator.state_dict()[key], restored.state_dict()[key], atol=0, rtol=0)

    def test_paired_sampler_receives_grid_and_replays_diffusion_noise(self):
        layout = SegmentationLayout(2, 2, 8, 16)
        allocator = self.allocator(layout)
        transmission = prepare_segmentation_transmissions(torch.randn(2, 4, 8, 16), torch.zeros(2),
                                                          allocator, RayleighFadingChannel(4, 8),
                                                          torch.Generator().manual_seed(9))
        sampler = PairedSampler()
        restored, diagnostics = restore_segmentation_transmissions(transmission, sampler, torch.Generator().manual_seed(5))
        self.assertEqual(set(restored), {"uniform", "allocated"})
        torch.testing.assert_close(sampler.calls[0][-1], sampler.calls[1][-1], atol=0, rtol=0)
        self.assertEqual(diagnostics["uniform"]["initialization_clipped_block_count"], 64)
        for index, name in enumerate(("uniform", "allocated")):
            torch.testing.assert_close(sampler.calls[index][1], transmission["strategies"][name]["precision"])

    def test_ddrm_accepts_full_cityscapes_shape_and_oracle_deep_fading(self):
        layout = SegmentationLayout()
        channel = RayleighFadingChannel(16, 32)
        allocator = self.allocator(layout)
        with patch.object(channel, "sample_channel_coefficient", return_value=torch.tensor([.001 + .001j])):
            transmission = prepare_segmentation_transmissions(torch.randn(1, 4, 128, 256), torch.zeros(1),
                                                              allocator, channel, torch.Generator().manual_seed(4))
        self.assertTrue(transmission["allocation_info"]["gain_floor_fallback"].item())
        scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
                                  timestep_spacing="trailing", clip_sample=False)
        def schedule(device):
            scheduler.set_timesteps(2, device=device)
            return scheduler.timesteps
        backend = SimpleNamespace(scheduler=scheduler, dtype=torch.float32, inference_timesteps=schedule,
                                  _empty_prompt_embedding=lambda batch: torch.zeros(batch, 1, 1),
                                  unet=lambda x, t, encoder_hidden_states: SimpleNamespace(sample=.1*x))
        sampler = OrthogonalDDRM(backend, initialization_policy="clip")
        original = transmission["strategies"]["allocated"]["precision"].clone()
        restored, diagnostics = restore_segmentation_transmissions(transmission, sampler,
                                                                   torch.Generator().manual_seed(3))
        self.assertEqual(restored["allocated"].shape, (1, 4, 128, 256))
        self.assertTrue(torch.isfinite(restored["allocated"]).all())
        self.assertEqual(diagnostics["allocated"]["initialization_clipped_block_count"], 512)
        torch.testing.assert_close(original, transmission["strategies"]["allocated"]["precision"], atol=0, rtol=0)

    def test_mocked_evaluation_and_inference_save_results_and_share_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            config = checkpoint_config(SegmentationLayout())
            allocator = self.allocator(SegmentationLayout())
            checkpoint_path = Path(directory) / "latest.pth"
            torch.save({"config": config, "allocator": allocator.state_dict(), "epoch": 0}, checkpoint_path)
            args = parse_arguments("evaluation", ["--checkpoint", str(checkpoint_path), "--device", "cpu",
                                   "--snr_db", "0", "--batch", "2", "--num_workers", "0", "--max_samples", "3", "--no-lpips"])
            self.assertEqual(args.restoration_method, "ddrm")
            sampler = PairedSampler()
            with patch("segmentation_orthogonal_power.CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                    patch("segmentation_orthogonal_power._load_vae", return_value=TinyVAE().eval()), \
                    patch("segmentation_orthogonal_power._build_sampler", return_value=sampler), \
                    patch("models.models.construct_model", return_value=TinyMask2Former()):
                result = evaluate(args)
                self.assertEqual(result["num_samples"], 3)
                self.assertTrue(result["complete"])
                self.assertEqual(result["miou"], 1.)
                self.assertEqual(set(result["strategies"]), {"uniform", "allocated", "vae"})
                for name in ("uniform", "allocated"):
                    metrics = result["strategies"][name]
                    self.assertEqual(metrics["ddrm_initialization_clipped_blocks"], 3 * 512)
                    self.assertAlmostEqual(metrics["mean_real_component_power"], 1., places=5)
                self.assertEqual(len(sampler.calls), 4)
                for first, second in ((0, 1), (2, 3)):
                    torch.testing.assert_close(sampler.calls[first][-1], sampler.calls[second][-1], atol=0, rtol=0)
                saved_path = next((Path(directory) / "evaluation").glob("*.json"))
                self.assertEqual(json.loads(saved_path.read_text())["num_samples"], 3)
                self.assertTrue(saved_path.with_suffix(".log").is_file())
                single = parse_arguments("inference", ["--checkpoint", str(checkpoint_path), "--device", "cpu",
                                         "--snr_db", "0", "--diffusion_steps", "2", "--no-lpips", "--quiet",
                                         "--output_dir", str(Path(directory) / "inference")])
                with patch("datasets.data_processing.CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                        patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=TinyVAE().eval()), \
                        patch("inference_orthogonal_power.StaggeredReverseDiffusion", side_effect=lambda *args, **kwargs: tiny_reverse_model()):
                    output = run_single(single)
                out = Path(output["output_dir"])
                for filename in ("input.png", "power_allocation.png", "result.json", "vae_reconstruction.png",
                                 "allocated_baseband.pt", "uniform_baseband.pt", "reconstructed.png",
                                 "uniform_reconstructed.png", "comparison.png", "allocated_oracle_ddrm.png"):
                    self.assertTrue((out / filename).is_file(), msg=filename)
                self.assertEqual(Image.open(out / "input.png").getpixel((0, 0)), (51, 51, 51))
                state = torch.load(out / "allocated_baseband.pt", weights_only=True)
                self.assertEqual(state["Q"].shape, (512, 512))
                self.assertEqual(state["precision"].shape, (1, 512))
                self.assertEqual(output["dataset"], "Cityscapes")
                self.assertEqual(output["group_grid"], [16, 32])
                self.assertFalse(output["uses_classifier"])
                self.assertEqual(output["strategies"]["allocated"]["diffusion"]["initialization_policy"], "clip")
                self.assertEqual(len(output["strategies"]["allocated"]["true_timestep"]), 512)

    def test_evaluation_subset_and_output_paths_are_reproducible(self):
        from segmentation_orthogonal_power import evaluation_output_path, select_evaluation_indices

        args = parse_arguments("evaluation", ["--checkpoint", "unused.pth", "--device", "cpu",
                               "--snr_db", "-5", "5", "--channel_model", "awgn",
                               "--gain_estimator_checkpoint", "gain.pth", "--max_samples", "2",
                               "--output_path", "/tmp/seg_eval.json"])
        self.assertEqual(args.receiver_mode, "gain_estimator")
        self.assertEqual(args.uniform_receiver_mode, "oracle")
        self.assertEqual(args.snr_db, [-5., 5.])
        dataset = range(100)
        before = torch.random.get_rng_state()
        indices = select_evaluation_indices(dataset, 10, 42)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(indices, select_evaluation_indices(dataset, 10, 42))
        self.assertNotEqual(indices, select_evaluation_indices(dataset, 10, 43))
        self.assertEqual(len(set(indices)), 10)
        self.assertEqual(sorted(indices), indices)
        self.assertEqual(select_evaluation_indices(dataset, None, 42), list(range(100)))
        self.assertEqual(evaluation_output_path(args, Path("unused.pth"), -5, 2), Path("/tmp/seg_eval_snr_-5.json"))
        args.output_path = None
        estimated = evaluation_output_path(args, Path("unused.pth"), -5, 2)
        args.receiver_mode = "oracle"
        oracle = evaluation_output_path(args, Path("unused.pth"), -5, 2)
        self.assertNotEqual(estimated, oracle)
        self.assertIn("_awgn_snr_-5_", estimated.name)

    def test_evaluation_pairs_multiple_snrs_and_handles_estimator_and_oracle_noise_overflow(self):
        import segmentation_orthogonal_power as evaluation

        class QualityModel(torch.nn.Module):
            def forward(self, left, right):
                return (left - right).abs().mean(dim=(1, 2, 3), keepdim=True)

        with tempfile.TemporaryDirectory() as directory:
            layout = SegmentationLayout()
            allocator = self.allocator(layout)
            transmitter_path = Path(directory) / "latest.pth"
            torch.save({"config": checkpoint_config(layout), "allocator": allocator.state_dict(), "epoch": 4}, transmitter_path)
            estimator = PatchGainEstimator(4, 8, 8, 8, group_grid=(16, 32))
            with torch.no_grad():
                estimator.log_gain_head.bias.fill_(-4.)
            estimator_path = Path(directory) / "gain.pth"
            torch.save({"estimator": estimator.state_dict(), "config": {
                "stage": "block_gain_estimator", "estimator_version": GAIN_ESTIMATOR_VERSION,
                "estimator_architecture": GAIN_ESTIMATOR_ARCHITECTURE, "channel_model": RAYLEIGH_CHANNEL_MODEL,
                "noise_real_component_variance": "N0", "mean_real_component_power": 1.,
                "latent_channels": 4, "patch_shape": [8, 8], "hidden_dim": 8, "group_grid": [16, 32],
                "allocator_fingerprint": "different-training-checkpoint",
            }}, estimator_path)
            vae = TinyVAE().eval()
            original_transmit = evaluation._transmit
            for channel_model in ("awgn", "rayleigh"):
                with self.subTest(channel_model=channel_model):
                    transmissions, source_latents = [], []

                    def capture(*args, **kwargs):
                        source_latents.append(args[1].clone())
                        transmission = original_transmit(*args, **kwargs)
                        transmissions.append(transmission)
                        return transmission

                    args = parse_arguments("evaluation", ["--checkpoint", str(transmitter_path), "--device", "cpu",
                                           "--gain_estimator_checkpoint", str(estimator_path), "--snr_db", "-5", "5",
                                           "--channel_model", channel_model, "--max_samples", "2", "--batch_size", "2",
                                           "--num_workers", "0", "--diffusion_steps", "2", "--seed", "42",
                                           "--output_path", str(Path(directory) / f"{channel_model}.json")])
                    with patch.object(evaluation, "CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                            patch.object(evaluation, "_load_vae", return_value=vae), \
                            patch.object(evaluation, "load_lpips", return_value=QualityModel()), \
                            patch.object(evaluation, "_transmit", side_effect=capture), \
                            patch.object(evaluation, "StaggeredReverseDiffusion", side_effect=lambda *a, **kw: tiny_reverse_model()) as backend, \
                            patch("models.models.construct_model", return_value=TinyMask2Former()) as model, \
                            patch.object(RayleighFadingChannel, "sample_channel_coefficient",
                                         side_effect=lambda z, generator: torch.full((z.shape[0],), .001 + .001j)):
                        result = evaluate(args)
                    self.assertEqual(backend.call_count, 1)
                    model.assert_called_once_with("Mask2Former")
                    self.assertFalse(result["gain_estimator_allocator_matches"])
                    self.assertEqual(result["strategy_receiver_modes"], {"uniform": "oracle", "allocated": "gain_estimator"})
                    self.assertEqual(len(result["sample_selection"]["indices"]), 2)
                    self.assertEqual(set(result["evaluations"]), {"-5", "5"})
                    torch.testing.assert_close(source_latents[0], source_latents[1], atol=0, rtol=0)
                    torch.testing.assert_close(transmissions[0]["epsilon"], transmissions[1]["epsilon"], atol=0, rtol=0)
                    torch.testing.assert_close(transmissions[0]["h"], transmissions[1]["h"], atol=0, rtol=0)
                    for snr, transmission in zip((-5, 5), transmissions):
                        saved_path = Path(directory) / f"{channel_model}_snr_{snr}.json"
                        saved = json.loads(saved_path.read_text())
                        self.assertTrue(saved_path.with_suffix(".log").is_file())
                        self.assertEqual(set(saved["evaluations"]), {str(snr)})
                        self.assertEqual(saved["sample_selection"]["indices"], result["sample_selection"]["indices"])
                        self.assertEqual(saved["num_samples"], 2)
                        for name, metrics in saved["strategies"].items():
                            self.assertEqual(metrics["num_valid_pixels"], 2 * (16 * 32 - 1))
                            self.assertEqual(metrics["miou"], 1.)
                            self.assertTrue(torch.isfinite(torch.tensor([metrics["psnr_db"], metrics["lpips"]])).all())
                            if name != "vae":
                                self.assertAlmostEqual(metrics["mean_real_component_power"], 1., places=5)
                        self.assertEqual(saved["strategies"]["allocated"]["ddrm_initialization_clipped_blocks"], 1024)
                        if channel_model == "awgn":
                            self.assertEqual(saved["mean_h_power"], 1.)
                            self.assertEqual(saved["strategies"]["uniform"]["ddrm_initialization_clipped_blocks"], 0)
                        else:
                            self.assertEqual(saved["strategies"]["uniform"]["ddrm_initialization_clipped_blocks"], 1024)
                        for name, received in transmission["strategies"].items():
                            gain = torch.full((2, 512), torch.exp(torch.tensor(-4.)).item()) if name == "allocated" else received["group_power"].sqrt()
                            expected = transmission["h"].abs()[:, None].square() * gain.square() / transmission["N0"][:, None]
                            torch.testing.assert_close(received["precision"], expected)
                            torch.testing.assert_close(received["raw_received"],
                                sample_raw_channel(
                                    received["transmitted_latent"], received["group_power"], transmission["N0"],
                                    transmission["h"], rows=16, cols=32, epsilon=transmission["epsilon"]), atol=0, rtol=0)

    def test_vae_only_evaluation_skips_diffusion_and_gain_estimator(self):
        import segmentation_orthogonal_power as evaluation

        with tempfile.TemporaryDirectory() as directory:
            allocator = self.allocator(SegmentationLayout())
            path = Path(directory) / "latest.pth"
            torch.save({"config": checkpoint_config(SegmentationLayout()), "allocator": allocator.state_dict()}, path)
            args = parse_arguments("evaluation", ["--checkpoint", str(path), "--device", "cpu", "--strategy", "uncompressed",
                                   "--gain_estimator_checkpoint", "not-needed.pth", "--num_workers", "0", "--no-lpips"])
            with patch.object(evaluation, "CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                    patch.object(evaluation, "_load_vae", return_value=TinyVAE().eval()), \
                    patch.object(evaluation, "load_gain_estimator") as estimator, \
                    patch.object(evaluation, "_build_sampler") as sampler, \
                    patch("models.models.construct_model", return_value=TinyMask2Former()):
                result = evaluate(args)
            estimator.assert_not_called()
            sampler.assert_not_called()
            self.assertEqual(set(result["strategies"]), {"vae"})
            self.assertEqual(result["num_samples"], 3)
            self.assertIsNone(result["mean_h_power"])
            self.assertIsNone(result["strategies"]["vae"]["lpips"])

    def test_cityscapes_gain_estimator_uses_its_own_grid_and_rejects_imagenet_checkpoint(self):
        layout = SegmentationLayout()
        allocator = self.allocator(layout)
        estimator = PatchGainEstimator(4, 8, 8, 8, group_grid=(16, 32))
        config = {
            "stage": "block_gain_estimator", "estimator_version": GAIN_ESTIMATOR_VERSION,
            "estimator_architecture": GAIN_ESTIMATOR_ARCHITECTURE, "channel_model": RAYLEIGH_CHANNEL_MODEL,
            "noise_real_component_variance": "N0", "mean_real_component_power": 1.,
            "latent_channels": 4, "patch_shape": [8, 8], "hidden_dim": 8, "group_grid": [16, 32],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gain.pth"
            torch.save({"estimator": estimator.state_dict(), "config": config}, path)
            loaded, _ = load_gain_estimator(path, allocator, "cpu", check_allocator_fingerprint=False)
            output = loaded(torch.randn(1, 4, 128, 256), torch.zeros(1), torch.zeros(1))
            self.assertEqual(output.shape, (1, 512))
            torch.testing.assert_close(output, torch.zeros_like(output))
            config["group_grid"] = [8, 8]
            torch.save({"estimator": PatchGainEstimator(4, 8, 8, 8).state_dict(), "config": config}, path)
            with self.assertRaisesRegex(ValueError, "8x8 blocks, transmitter 16x32"):
                load_gain_estimator(path, allocator, "cpu", check_allocator_fingerprint=False)

    def test_cityscapes_gain_training_freezes_transmitter_resumes_and_connects_to_inference(self):
        import train_gain_estimator as gain_training

        layout = SegmentationLayout()
        allocator = self.allocator(layout)
        config = checkpoint_config(layout)
        config["max_timestep"] = 999
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(TinyCityscapes(), [0, 1]), batch_size=1)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(TinyCityscapes(), [2]), batch_size=1)
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                transmitter_path = Path(directory) / "transmitter.pth"
                torch.save({"config": config, "allocator": allocator.state_dict(), "epoch": 0}, transmitter_path)
                with self.assertRaisesRegex(ValueError, "ImageNet"):
                    gain_training.load_transmitter(transmitter_path, "cpu")
                frozen, channel, _, _ = gain_training.load_transmitter(transmitter_path, "cpu", allow_segmentation=True)
                self.assertEqual((frozen.group_rows, frozen.group_cols), (16, 32))
                self.assertFalse(any(parameter.requires_grad for parameter in frozen.parameters()))
                frozen_state = copy.deepcopy(frozen.state_dict())
                vae = TinyVAE()
                vae_state = copy.deepcopy(vae.state_dict())
                argv = ["train_gain_estimator.py", "--allocator_checkpoint", str(transmitter_path),
                        "--exp_name", "city_gain", "--device", "cpu", "--epochs", "1", "--hidden_dim", "8",
                        "--num_workers", "0", "--log_interval", "1", "--snr_db_min", "-3", "--snr_db_max", "15"]
                output_path = Path("checkpoints_gain_estimator/Cityscapes/city_gain/latest.pth")

                def run_training():
                    with patch("sys.argv", argv), \
                            patch.object(gain_training, "load_transmitter", return_value=(frozen, channel, config, transmitter_path)), \
                            patch.object(gain_training.AutoencoderKL, "from_pretrained", return_value=vae), \
                            patch.object(gain_training, "construct_cityscapes_dataloaders", return_value=(train_loader, val_loader)) as loader, \
                            patch.object(gain_training, "allocate_power_for_inference", wraps=gain_training.allocate_power_for_inference) as allocate:
                        gain_training.run_training()
                    self.assertEqual(loader.call_args.args[1], 1)  # Cityscapes-specific batch default.
                    self.assertEqual(loader.call_args.args[-1], 0)
                    self.assertEqual(allocate.call_count, 3)
                    for call in allocate.call_args_list:
                        self.assertEqual(call.kwargs["max_timestep"], 999)
                        self.assertEqual(call.kwargs["min_gain"], 0.)
                    self.close_loggers()
                    return torch.load(output_path, weights_only=False)

                checkpoint = run_training()
                self.assertEqual(checkpoint["config"]["dataset"], "Cityscapes")
                self.assertEqual(checkpoint["config"]["group_grid"], [16, 32])
                self.assertEqual(checkpoint["config"]["patch_shape"], [8, 8])
                self.assertEqual(checkpoint["config"]["input_shape"], [1024, 2048])
                self.assertEqual(checkpoint["config"]["estimator_head_dims"], [23554, 8, 512])
                self.assertEqual(checkpoint["global_step"], 2)
                self.assertEqual(checkpoint["scheduler"]["last_epoch"], 2)
                self.assertTrue(torch.isfinite(torch.tensor(checkpoint["metrics"]["val"]["log_gain_mse"])))
                self.assertGreater(checkpoint["estimator"]["log_gain_head.weight"].abs().sum().item(), 0)
                trained_head = checkpoint["estimator"]["log_gain_head.weight"].clone()
                argv[argv.index("--epochs") + 1] = "2"
                argv.extend(["--resume", str(output_path)])
                resumed = run_training()
                self.assertEqual(resumed["global_step"], 4)
                self.assertEqual(resumed["scheduler"]["last_epoch"], 4)
                self.assertFalse(torch.equal(resumed["estimator"]["log_gain_head.weight"], trained_head))
                for key in frozen_state:
                    torch.testing.assert_close(frozen.state_dict()[key], frozen_state[key], rtol=0, atol=0)
                for key in vae_state:
                    torch.testing.assert_close(vae.state_dict()[key], vae_state[key], rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
                estimator, _ = load_gain_estimator(output_path, frozen, "cpu")
                self.assertEqual(estimator.num_blocks, 512)
                args = parse_arguments("inference", ["--checkpoint", str(transmitter_path), "--device", "cpu",
                                      "--gain_estimator_checkpoint", str(output_path), "--strategy", "allocated",
                                      "--diffusion_steps", "2", "--snr_db", "0", "--no-lpips", "--quiet",
                                      "--output_dir", str(Path(directory) / "inference")])
                with patch("datasets.data_processing.CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                        patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=vae), \
                        patch("inference_orthogonal_power.StaggeredReverseDiffusion", side_effect=lambda *args, **kwargs: tiny_reverse_model()):
                    result = run_single(args)
                self.assertEqual(result["receiver_mode"], "gain_estimator")
                self.assertEqual(len(result["strategies"]["allocated"]["estimated_amplitude_gain"]), 512)
                self.assertEqual(result["strategies"]["allocated"]["diffusion"]["initialization_policy"], "clip")
            finally:
                self.close_loggers()
                os.chdir(previous_directory)

    def test_unified_cityscapes_input_image_and_shared_cli_options(self):
        from inference_orthogonal_power import load_input
        config = checkpoint_config(SegmentationLayout())
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            Image.new("RGB", (20, 10), (30, 40, 50)).save(image_path)
            args = parse_arguments("inference", ["--checkpoint", "unused.pth", "--device", "cpu",
                                  "--input_image", str(image_path), "--amp", "--quiet", "--no-lpips",
                                  "--no_save_latents", "--strategy", "allocated"])
            image, label, path = load_input(args, config)
            self.assertEqual(image.shape, (1, 3, 1024, 2048))
            self.assertIsNone(label)
            self.assertEqual(path, str(image_path))
            self.assertTrue(args.amp)
            self.assertTrue(args.quiet)
            self.assertTrue(args.no_save_latents)
            self.assertFalse(args.lpips)
            self.assertEqual(args.diffusion_steps, 25)

    def test_unified_segmentation_dps_uses_dynamic_grid(self):
        layout = SegmentationLayout()
        allocator = self.allocator(layout)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pth"
            torch.save({"allocator": allocator.state_dict(), "config": checkpoint_config(layout), "epoch": 0}, checkpoint_path)
            args = parse_arguments("inference", ["--checkpoint", str(checkpoint_path), "--device", "cpu",
                                  "--restoration_method", "dps", "--diffusion_steps", "2", "--quiet", "--no-lpips",
                                  "--strategy", "allocated", "--output_dir", str(Path(directory) / "out")])
            with patch("datasets.data_processing.CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                    patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=TinyVAE().eval()), \
                    patch("inference_orthogonal_power.StaggeredReverseDiffusion", side_effect=lambda *args, **kwargs: tiny_reverse_model()):
                result = run_single(args)
            self.assertEqual(set(result["strategies"]), {"allocated"})
            self.assertEqual(result["restoration_method"], "dps")
            self.assertIn("psnr_db", result["strategies"]["allocated"]["dps_vs_input"])
            self.assertEqual(result["strategies"]["allocated"]["diffusion"]["timesteps"], [999, 499])

    def test_unified_segmentation_inference_modes_quality_and_strategy_selection(self):
        from inference_orthogonal_power import run_inference

        layout = SegmentationLayout()
        allocator = self.allocator(layout)
        config = checkpoint_config(layout)
        cases = (("rayleigh", "both", "oracle", "oracle"),
                 ("awgn", "both", "gain_estimator", "oracle"),
                 ("rayleigh", "both", "oracle", "gain_estimator"),
                 ("awgn", "allocated", "oracle", "oracle"),
                 ("awgn", "uniform", "oracle", "oracle"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint_path = directory / "latest.pth"
            # An unrelated segmenter state must never trigger an added task evaluator.
            torch.save({"allocator": allocator.state_dict(), "config": config,
                        "segmenter": {"unused": torch.zeros(1)}, "epoch": 0}, checkpoint_path)
            for channel_model, strategy, allocated_mode, uniform_mode in cases:
                with self.subTest(channel=channel_model, strategy=strategy, modes=(allocated_mode, uniform_mode)):
                    folder = directory / f"{channel_model}_{strategy}_{allocated_mode}_{uniform_mode}"
                    arguments = ["--checkpoint", str(checkpoint_path), "--device", "cpu", "--snr_db", "0",
                                 "--diffusion_steps", "2", "--output_dir", str(folder), "--channel_model", channel_model,
                                 "--receiver_mode", allocated_mode, "--uniform_receiver_mode", uniform_mode,
                                 "--strategy", strategy, "--quiet"]
                    needs_estimator = "gain_estimator" in (allocated_mode, uniform_mode)
                    if needs_estimator:
                        arguments += ["--gain_estimator_checkpoint", "cityscapes_gain.pth"]
                    if strategy == "allocated":
                        arguments += ["--no_save_latents"]
                    args = parse_arguments("inference", arguments)
                    estimator = PatchGainEstimator(4, 8, 8, 8, group_grid=(16, 32)).eval()
                    fake_lpips = lambda reference, reconstructed: (reference-reconstructed).square().mean((1, 2, 3), keepdim=True)
                    with patch("datasets.data_processing.CityscapesSemanticDataset", return_value=TinyCityscapes()), \
                            patch("inference_orthogonal_power.AutoencoderKL.from_pretrained", return_value=TinyVAE().eval()), \
                            patch("inference_orthogonal_power.StaggeredReverseDiffusion", side_effect=lambda *args, **kwargs: tiny_reverse_model()) as reverse, \
                            patch("inference_orthogonal_power.load_gain_estimator", return_value=(estimator, {})), \
                            patch("inference_orthogonal_power.load_lpips", return_value=fake_lpips), \
                            patch("train_segmentation_power_allocator.LatentSegFormerB0", side_effect=AssertionError("No task evaluator in inference")):
                        result = run_inference(args)
                    expected = ("allocated", "uniform") if strategy == "both" else (strategy,)
                    self.assertEqual(set(result["strategies"]), set(expected))
                    self.assertEqual(reverse.call_count, 1)
                    self.assertEqual(result["receiver_mode"], allocated_mode)
                    self.assertEqual(result["channel_model"], "awgn" if channel_model == "awgn" else RAYLEIGH_CHANNEL_MODEL)
                    shared_h = shared_epsilon = None
                    for name in expected:
                        row = result["strategies"][name]
                        mode = allocated_mode if name == "allocated" else uniform_mode
                        self.assertEqual(row["receiver_mode"], mode)
                        self.assertIn("psnr_db", row["direct_vs_input"])
                        self.assertIn("lpips", row["ddrm_vs_input"])
                        self.assertIsNone(row["task_direct"])
                        self.assertIsNone(row["task_diffusion"])
                        self.assertEqual(len(row["estimated_timestep"]), 512)
                        self.assertEqual(row["diffusion"]["initialization_policy"], "clip")
                        if strategy == "allocated":
                            self.assertFalse((folder / f"{name}_baseband.pt").exists())
                            continue
                        state = torch.load(folder / f"{name}_baseband.pt", weights_only=True)
                        self.assertEqual(state["Q"].shape, (512, 512))
                        self.assertEqual(state["true_timestep"].shape, (1, 512))
                        if shared_h is None:
                            shared_h, shared_epsilon = state["channel_coefficient"], state["epsilon"]
                        torch.testing.assert_close(state["channel_coefficient"], shared_h, rtol=0, atol=0)
                        torch.testing.assert_close(state["epsilon"], shared_epsilon, rtol=0, atol=0)
                        if channel_model == "awgn":
                            torch.testing.assert_close(shared_h, torch.ones_like(shared_h))
                        if name == "uniform":
                            torch.testing.assert_close(state["Q"], torch.eye(512))
                        if mode == "oracle":
                            torch.testing.assert_close(state["estimated_gain"], state["group_power"].sqrt())
                        else:
                            torch.testing.assert_close(state["estimated_gain"], torch.ones_like(state["estimated_gain"]))
                    self.assertTrue((folder / "power_allocation.png").is_file())
                    if strategy == "uniform":
                        self.assertFalse((folder / "reconstructed.png").exists())
                    if channel_model == "rayleigh" and strategy == "both" and allocated_mode == uniform_mode == "oracle":
                        state = torch.load(folder / "allocated_baseband.pt", weights_only=True)
                        transmission = prepare_segmentation_transmissions(
                            state["clean_latent"], torch.zeros(1), allocator, RayleighFadingChannel(16, 32),
                            torch.Generator().manual_seed(args.seed + 1), max_timestep=999,
                        )
                        restored, _ = restore_segmentation_transmissions(
                            transmission, OrthogonalDDRM(tiny_reverse_model(), initialization_policy="clip"),
                            torch.Generator().manual_seed(args.seed + 2),
                        )
                        torch.testing.assert_close(state["raw_received_latent"], transmission["strategies"]["allocated"]["raw_received"], rtol=0, atol=0)
                        torch.testing.assert_close(state["restored_latent"], restored["allocated"], rtol=0, atol=0)

    def test_training_cli_saves_valid_checkpoint_and_resumes_optimizer_and_segmenter(self):
        from train_segmentation_power_allocator import run_training

        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(TinyCityscapes(), [0, 1]), batch_size=1)
        validation_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(TinyCityscapes(), [2]), batch_size=1)
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                argv = ["train_segmentation_power_allocator.py", "--exp_name", "smoke", "--device", "cpu",
                        "--epochs", "1", "--hidden_dim", "8", "--num_workers", "0", "--log_interval", "1"]
                with patch("sys.argv", argv), \
                        patch("train_segmentation_power_allocator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                        patch("train_segmentation_power_allocator.LatentSegFormerB0", side_effect=lambda channels: TinyTask()), \
                        patch("train_segmentation_power_allocator.construct_cityscapes_dataloaders", return_value=(loader, validation_loader)):
                    run_training()
                checkpoint_path = Path("checkpoints_power/Cityscapes/SegFormer-B0/snr_uniform/smoke/latest.pth")
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(validate_segmentation_config(checkpoint["config"]), SegmentationLayout())
                self.assertEqual(checkpoint["global_step"], 2)
                self.assertEqual(checkpoint["scheduler"]["last_epoch"], 2)
                self.assertEqual(checkpoint["scheduler"]["warmup_steps"], 1)
                self.assertEqual(checkpoint["scheduler"]["total_steps"], 2)
                self.assertEqual(checkpoint["config"]["lr_schedule"]["interval"], "optimizer_step")
                self.assertEqual([group["lr"] for group in checkpoint["optimizer"]["param_groups"]], [.0005, .0005])
                self.assertTrue(checkpoint["optimizer"]["state"])
                previous_task = checkpoint["segmenter"]["head.weight"].clone()
                self.close_loggers()
                # Resume must preserve the saved LR instead of restarting warmup.
                argv[argv.index("--epochs") + 1] = "2"
                argv.extend(["--resume", str(checkpoint_path)])
                with patch("sys.argv", argv), \
                        patch("train_segmentation_power_allocator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                        patch("train_segmentation_power_allocator.LatentSegFormerB0", side_effect=lambda channels: TinyTask()), \
                        patch("train_segmentation_power_allocator.construct_cityscapes_dataloaders", return_value=(loader, validation_loader)):
                    run_training()
                resumed = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(resumed["epoch"], 1)
                self.assertEqual(resumed["global_step"], 4)
                self.assertEqual(resumed["scheduler"]["last_epoch"], 4)
                self.assertEqual(resumed["scheduler"]["total_steps"], 2)
                self.assertEqual([group["lr"] for group in resumed["optimizer"]["param_groups"]], [.0005, .0005])
                for value in resumed["optimizer"]["state"].values():
                    self.assertEqual(value["step"].item(), 4)
                self.assertFalse(torch.equal(resumed["segmenter"]["head.weight"], previous_task))
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
