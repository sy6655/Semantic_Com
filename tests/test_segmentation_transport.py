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
    ALLOCATOR_ARCHITECTURE, RAYLEIGH_CHANNEL_MODEL, OrthogonalPatchTransform,
    apply_complex_channel, block_map, patchify,
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
                                   "--snr_db", "0", "--batch", "2", "--num_workers", "0", "--max_samples", "3"])
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
                self.assertEqual(set(result["strategies"]), {"uniform", "allocated", "uncompressed"})
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
                                         "--snr_db", "0", "--output_dir", str(Path(directory) / "inference")])
                output = run_single(single)
                out = Path(output["output_dir"])
                for filename in ("input.png", "ground_truth.png", "power_allocation.png", "result.json",
                                 "latents.pt", "allocated_ddrm.png", "uniform_ddrm.png", "denoised_comparison.png"):
                    self.assertTrue((out / filename).is_file(), msg=filename)
                self.assertEqual(Image.open(out / "input.png").getpixel((0, 0)), (51, 51, 51))
                state = torch.load(out / "latents.pt", weights_only=True)
                self.assertEqual(state["Q"].shape, (512, 512))
                self.assertEqual(state["allocated"]["precision"].shape, (1, 512))
                self.assertEqual(output["strategies"]["allocated"]["diffusion"]["initialization_clipped_block_count"], 512)

    def test_training_cli_saves_valid_checkpoint_and_resumes_optimizer_and_segmenter(self):
        from train_segmentation_power_allocator import run_training

        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(TinyCityscapes(), [0]), batch_size=1)
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                argv = ["train_segmentation_power_allocator.py", "--exp_name", "smoke", "--device", "cpu",
                        "--epochs", "1", "--hidden_dim", "8", "--num_workers", "0", "--log_interval", "1"]
                with patch("sys.argv", argv), \
                        patch("train_segmentation_power_allocator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                        patch("train_segmentation_power_allocator.LatentSegFormerB0", side_effect=lambda channels: TinyTask()), \
                        patch("train_segmentation_power_allocator.construct_cityscapes_dataloaders", return_value=(loader, loader)):
                    run_training()
                checkpoint_path = Path("checkpoints_power/Cityscapes/SegFormer-B0/snr_uniform/smoke/latest.pth")
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(validate_segmentation_config(checkpoint["config"]), SegmentationLayout())
                self.assertEqual(checkpoint["global_step"], 1)
                self.assertTrue(checkpoint["optimizer"]["state"])
                previous_task = checkpoint["segmenter"]["head.weight"].clone()
                self.close_loggers()
                # The first run's cosine schedule reached zero LR; use its saved
                # optimizer state and check step counters as proof of resume.
                argv[argv.index("--epochs") + 1] = "2"
                argv.extend(["--resume", str(checkpoint_path)])
                with patch("sys.argv", argv), \
                        patch("train_segmentation_power_allocator.AutoencoderKL.from_pretrained", return_value=TinyVAE()), \
                        patch("train_segmentation_power_allocator.LatentSegFormerB0", side_effect=lambda channels: TinyTask()), \
                        patch("train_segmentation_power_allocator.construct_cityscapes_dataloaders", return_value=(loader, loader)):
                    run_training()
                resumed = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(resumed["epoch"], 1)
                self.assertEqual(resumed["global_step"], 2)
                self.assertEqual(resumed["scheduler"]["last_epoch"], 2)
                for value in resumed["optimizer"]["state"].values():
                    self.assertEqual(value["step"].item(), 2)
                torch.testing.assert_close(resumed["segmenter"]["head.weight"], previous_task, atol=0, rtol=0)
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
