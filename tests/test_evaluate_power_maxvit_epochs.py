import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import DDIMScheduler
from torch.utils.data import Dataset

import evaluate_power_maxvit as common
import evaluate_power_maxvit_epochs as comparison
from train_power_allocator import PowerAllocator, RayleighFadingChannel


class LabeledImages(Dataset):
    def __init__(self, root, classes=3, images_per_class=4):
        self.classes = [f"class_{index:04d}" for index in range(classes)]
        self.targets = [label for label in range(classes) for _ in range(images_per_class)]
        self.samples = [(str(Path(root) / self.classes[label] / f"{index}.jpg"), label)
                        for index, label in enumerate(self.targets)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 16, 16), float(index)), self.targets[index]


def tiny_sampler():
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
                              timestep_spacing="trailing", clip_sample=False, set_alpha_to_one=False)

    def timesteps(device):
        scheduler.set_timesteps(2, device=device)
        return scheduler.timesteps

    backend = SimpleNamespace(
        scheduler=scheduler, dtype=torch.float32, inference_timesteps=timesteps,
        _empty_prompt_embedding=lambda batch: torch.zeros(batch, 1, 1),
        unet=lambda latent, timestep, encoder_hidden_states: SimpleNamespace(sample=.1 * latent),
    )
    return common.build_restoration_sampler(backend, SimpleNamespace(restoration_method="ddrm"), receiver_mode="oracle")


class EpochComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def transmitter(self, epoch):
        allocator = PowerAllocator(4, 64, 8, orthogonal_transform=True, block_dim=16).eval()
        with torch.no_grad():
            allocator.orthogonal.upper.normal_(std=.03)
            allocator.block_allocator.logit_head[-1].weight.normal_(std=.3)
        return {"epoch": epoch, "path": Path(f"epoch_{epoch:03d}.pth"), "allocator": allocator,
                "channel": RayleighFadingChannel(8, 8), "config": {"temperature": 1.0}}

    def test_real_ddrm_matches_existing_allocated_path_and_shares_noise(self):
        transmitters = {"epoch_010": self.transmitter(10), "epoch_015": self.transmitter(15)}
        latent, snr = torch.randn(3, 4, 16, 16), torch.tensor([-5., 0., 10.])
        first = transmitters["epoch_010"]
        expected, expected_h, expected_fallback = common.restore_transmissions(
            latent, snr, first["allocator"], first["channel"], tiny_sampler(),
            torch.Generator().manual_seed(31), torch.Generator().manual_seed(32), max_timestep=999,
        )
        actual_sampler = tiny_sampler()
        diffusion_states, matrices = [], []

        class RecordingSampler:
            @property
            def last_diagnostics(self):
                return actual_sampler.last_diagnostics

            def __call__(self, observation, precision, q, generator, **kwargs):
                diffusion_states.append(generator.get_state().clone())
                matrices.append(q.clone())
                return actual_sampler(observation, precision, q, generator, **kwargs)

        with patch.object(common, "sample_raw_channel", wraps=common.sample_raw_channel) as raw, \
                patch.object(first["channel"], "sample_channel_coefficient",
                             wraps=first["channel"].sample_channel_coefficient) as sample_h:
            restored, h, fallbacks, diagnostics = comparison.restore_epochs(
                latent, snr, transmitters, RecordingSampler(), torch.Generator().manual_seed(31),
                torch.Generator().manual_seed(32), max_timestep=999,
            )
        self.assertEqual(sample_h.call_count, 1)
        self.assertEqual(raw.call_count, 2)
        for index in (2, 3):  # N0 and h.
            torch.testing.assert_close(raw.call_args_list[0].args[index], raw.call_args_list[1].args[index], rtol=0, atol=0)
        torch.testing.assert_close(raw.call_args_list[0].kwargs["epsilon"], raw.call_args_list[1].kwargs["epsilon"], rtol=0, atol=0)
        torch.testing.assert_close(diffusion_states[0], diffusion_states[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(matrices[0], matrices[1]))
        self.assertFalse(torch.equal(raw.call_args_list[0].args[0], raw.call_args_list[1].args[0]))
        torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
        torch.testing.assert_close(restored["epoch_010"], expected["allocated"], rtol=0, atol=0)
        torch.testing.assert_close(fallbacks["epoch_010"], expected_fallback)
        self.assertEqual(set(restored), set(transmitters))
        self.assertEqual(diagnostics["epoch_010"]["initialization_policy"], "clip")
        swapped, _, _, _ = comparison.restore_epochs(
            latent, snr, dict(reversed(list(transmitters.items()))), tiny_sampler(),
            torch.Generator().manual_seed(31), torch.Generator().manual_seed(32), max_timestep=999,
        )
        for name in restored:
            torch.testing.assert_close(swapped[name], restored[name], rtol=0, atol=0)

    def test_full_evaluation_only_compares_epochs_with_same_samples_at_all_snrs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LabeledImages(directory)
            for epoch in (10, 15):
                (Path(directory) / f"epoch_{epoch:03d}.pth").touch()
            args = comparison.parse_arguments([
                "--checkpoint_dir", directory, "--max_samples", "9", "--batch_size", "4",
                "--num_workers", "0", "--restoration_method", "ddrm", "--snr_db", "0", "10",
                "--output_path", str(Path(directory) / "comparison.json"),
            ])
            config = {"data_path": directory, "image_size": 16,
                      "vae_model_id": "local-test", "vae_model_revision": None}
            seen = [[], []]
            floors = []
            call_index = 0

            def accumulate(stats, reconstructed, reference, labels, *unused):
                nonlocal call_index
                strategy = call_index % 2
                ids = reference[:, 0, 0, 0].long().tolist()
                seen[strategy].extend(ids)
                torch.testing.assert_close(reconstructed, reference)
                self.assertEqual(labels.tolist(), [dataset.targets[index] for index in ids])
                count = len(ids)
                stats += stats.new_tensor([count, count * strategy, count, count * (.1 + strategy),
                                          count * (20 + strategy), count * (.3 + strategy)])
                call_index += 1

            def restore(latent, snr, transmitters, sampler, *generators, **kwargs):
                floors.append(kwargs["max_timestep"])
                ids = latent[:, 0, 0, 0].long()
                return (
                    {name: latent.clone() for name in transmitters},
                    torch.ones(len(ids), dtype=torch.complex64),
                    {name: ids % 2 == index for index, name in enumerate(transmitters)},
                    {name: {"initialization_clipped_images": int((ids % 2 == index).sum()),
                            "initialization_clipped_block_count": int((ids % 2 == index).sum()) * 64}
                     for index, name in enumerate(transmitters)},
                )

            with patch.object(common, "load_transmitter", return_value=(None, None, config, None)) as load, \
                    patch.object(common, "load_lpips", return_value=None) as lpips, \
                    patch.object(common.AutoencoderKL, "from_pretrained", return_value=torch.nn.Identity()) as vae, \
                    patch.object(common, "StaggeredReverseDiffusion", return_value=None) as diffusion, \
                    patch.object(common, "construct_model", return_value=torch.nn.Identity()) as maxvit, \
                    patch.object(common.datasets, "ImageFolder", return_value=dataset), \
                    patch.object(common, "encode_latents", side_effect=lambda images, *args: images) as encode, \
                    patch.object(common, "_decode_latent", side_effect=lambda vae, latent, scale: latent) as decode, \
                    patch.object(common, "accumulate", side_effect=accumulate), \
                    patch.object(comparison, "restore_epochs", side_effect=restore):
                result = comparison.evaluate(args, torch.device("cpu"))
            expected = common.select_balanced_indices(dataset, 9, args.seed)
            self.assertEqual(seen, [expected * 2, expected * 2])
            self.assertEqual(floors, [999] * 6)
            self.assertEqual(encode.call_count, 6)  # Three batches per SNR, shared between epochs.
            self.assertEqual(decode.call_count, 12)  # Two epoch restorations; no VAE/uniform baseline.
            for constructor in (lpips, vae, diffusion, maxvit):
                self.assertEqual(constructor.call_count, 1)
            self.assertEqual([call.args[0].name for call in load.call_args_list], ["epoch_010.pth", "epoch_015.pth"])
            self.assertEqual(result["sample_selection"]["indices"], expected)
            self.assertEqual(set(result["sample_selection"]["class_counts"].values()), {3})
            for snr_db in (0, 10):
                saved = json.loads((Path(directory) / f"comparison_snr_{snr_db}.json").read_text())
                self.assertEqual(saved["max_timestep"], 999)
                self.assertEqual(saved["channel_model"], "rayleigh_block_fading")
                self.assertEqual(saved["sample_selection"]["indices"], expected)
                self.assertEqual(set(saved["evaluations"]), {str(snr_db)})
                evaluation = saved["evaluations"][str(snr_db)]
                self.assertEqual(set(evaluation["strategies"]), {"epoch_010", "epoch_015"})
                for index, name in enumerate(("epoch_010", "epoch_015")):
                    row = evaluation["strategies"][name]
                    self.assertEqual(row["num_samples"], 9)
                    self.assertEqual(row["top1_accuracy"], index)
                    self.assertAlmostEqual(row["psnr_db"], 20 + index)
                    self.assertAlmostEqual(row["lpips"], .3 + index)
                    fallback = sum(sample % 2 == index for sample in expected)
                    self.assertEqual(evaluation["gain_floor_infeasible_images"][name], fallback)
                    self.assertEqual(evaluation["ddrm_initialization_clipped"][name],
                                     {"images": fallback, "blocks": fallback * 64})

    def test_default_count_selects_one_image_per_class_and_output_is_distinct(self):
        args = comparison.parse_arguments(["--checkpoint_dir", "/tmp/epochs", "--restoration_method", "ddrm"])
        self.assertEqual(args.epochs, [10, 15])
        self.assertEqual(args.max_samples, 1000)
        dataset = LabeledImages("/tmp/imagenet", classes=1000, images_per_class=3)
        indices = common.select_balanced_indices(dataset, args.max_samples, args.seed)
        self.assertEqual(len(indices), 1000)
        self.assertEqual(len(set(indices)), 1000)
        self.assertEqual(sorted(dataset.targets[index] for index in indices), list(range(1000)))
        args.max_timestep = common.resolve_max_timestep(args)
        transmitters = {"epoch_010": {"epoch": 10, "path": Path("/tmp/epochs/epoch_010.pth")},
                        "epoch_015": {"epoch": 15, "path": Path("/tmp/epochs/epoch_015.pth")}}
        output = comparison.evaluation_output_path(args, transmitters, 10, 1000)
        self.assertEqual(output.parent, Path("/tmp/epochs/evaluation"))
        self.assertTrue(output.name.startswith("maxvit_epochs_010_015_ddrm_rayleigh_snr_10_"))
        self.assertTrue(output.name.endswith("_balanced_n1000_maxt_999.json"))

    def test_missing_checkpoint_fails_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            args = comparison.parse_arguments(["--checkpoint_dir", directory])
            (Path(directory) / "epoch_010.pth").touch()
            with patch.object(common, "load_transmitter") as load:
                with self.assertRaisesRegex(FileNotFoundError, "epoch_015.pth"):
                    comparison.load_epochs(args, torch.device("cpu"))
                load.assert_not_called()

    def test_incompatible_shared_vae_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = comparison.parse_arguments(["--checkpoint_dir", directory])
            for epoch in (10, 15):
                (Path(directory) / f"epoch_{epoch:03d}.pth").touch()
            config = {"data_path": directory, "vae_model_id": "local-test", "vae_model_revision": None}
            with patch.object(common, "load_transmitter", side_effect=[(None, None, config, None),
                              (None, None, {**config, "image_size": 256}, None)]):
                with self.assertRaisesRegex(ValueError, "image_size"):
                    comparison.load_epochs(args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
