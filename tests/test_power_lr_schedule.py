import copy
import math
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import train_power_allocator as training
from train_power_allocator import (
    WarmupCosineLR, build_training_scheduler, optimizer_step_with_schedule,
    training_schedule_config,
)


class PowerLRScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def optimizer(self, values=None, rates=(.001, .002)):
        parameters = [torch.nn.Parameter(torch.tensor(value)) for value in (values or (1., 2.))]
        optimizer = torch.optim.AdamW([
            {"params": [parameter], "lr": rate} for parameter, rate in zip(parameters, rates)
        ])
        return parameters, optimizer

    def update(self, parameters, optimizer, scheduler, scaler):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in parameters)
        scaler.scale(loss).backward()
        return optimizer_step_with_schedule(optimizer, scheduler, scaler, parameters, 1.)

    def test_fractional_warmup_cosine_endpoints_and_parameter_group_ratios(self):
        parameters, optimizer = self.optimizer()
        scheduler = build_training_scheduler(optimizer, epochs=10, steps_per_epoch=4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.assertEqual(scheduler.total_steps, 40)
        self.assertEqual(scheduler.warmup_steps, 2)
        snapshots = {}
        for completed in range(43):
            snapshots[completed] = [group["lr"] for group in optimizer.param_groups]
            if completed < 42:
                self.update(parameters, optimizer, scheduler, scaler)
        for completed, factor in ((0, .1), (1, .55), (2, 1.), (21, .75), (40, .5), (42, .5)):
            for actual, peak in zip(snapshots[completed], (.001, .002)):
                self.assertAlmostEqual(actual, peak * factor, places=15)
        for index in range(2, 40):
            self.assertGreater(snapshots[index][0], snapshots[index + 1][0])
        self.assertEqual(training_schedule_config(scheduler)["interval"], "optimizer_step")

    def test_zero_warmup_starts_at_peak_and_reaches_requested_floor(self):
        parameters, optimizer = self.optimizer()
        scheduler = build_training_scheduler(optimizer, 2, 2, warmup_epochs=0, min_lr_ratio=.2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.assertEqual(scheduler.get_last_lr(), [.001, .002])
        for _ in range(4):
            self.update(parameters, optimizer, scheduler, scaler)
        for actual, expected in zip(scheduler.get_last_lr(), (.0002, .0004)):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_schedule_inputs_are_rejected(self):
        invalid = (
            {"total_steps": 0}, {"total_steps": 10, "warmup_steps": 10},
            {"total_steps": 10, "warmup_steps": -1},
            {"total_steps": 10, "min_lr_ratio": 1.1},
            {"total_steps": 10, "min_lr_ratio": float("nan")},
            {"total_steps": 10, "warmup_start_factor": -.1},
        )
        for options in invalid:
            with self.subTest(options=options):
                _, optimizer = self.optimizer()
                with self.assertRaises(ValueError):
                    WarmupCosineLR(optimizer, **options)
        _, optimizer = self.optimizer()
        for epochs, steps, warmup in ((0, 4, 0), (2, 0, 0), (2, 4, 2), (2, 4, float("nan"))):
            with self.assertRaises(ValueError):
                build_training_scheduler(optimizer, epochs, steps, warmup_epochs=warmup)

    def test_resume_preserves_next_update_lr_and_exact_optimization_trajectory(self):
        for split in (1, 13):  # Resume during warmup and during cosine decay.
            with self.subTest(split=split):
                parameters, optimizer = self.optimizer()
                scheduler = build_training_scheduler(optimizer, 10, 4)
                scaler = torch.amp.GradScaler("cpu", enabled=False)
                for _ in range(split):
                    self.update(parameters, optimizer, scheduler, scaler)
                saved_values = [parameter.item() for parameter in parameters]
                checkpoint = copy.deepcopy({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
                expected_lrs = []
                for _ in range(split, 40):
                    expected_lrs.append([group["lr"] for group in optimizer.param_groups])
                    self.update(parameters, optimizer, scheduler, scaler)
                resumed_parameters, resumed_optimizer = self.optimizer(saved_values, rates=(.8, .9))
                resumed = build_training_scheduler(
                    resumed_optimizer, 99, 11, warmup_epochs=3., min_lr_ratio=.1,
                    checkpoint=checkpoint,
                )
                self.assertEqual(resumed.last_epoch, split)
                self.assertEqual(resumed.total_steps, 40)
                self.assertEqual(resumed.warmup_steps, 2)
                self.assertEqual(resumed.min_lr_ratio, .5)
                actual_lrs = []
                for _ in range(split, 40):
                    actual_lrs.append([group["lr"] for group in resumed_optimizer.param_groups])
                    self.update(resumed_parameters, resumed_optimizer, resumed, scaler)
                self.assertEqual(actual_lrs, expected_lrs)
                for actual, expected in zip(resumed_parameters, parameters):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_checkpoint_keeps_epoch_cosine_and_original_horizon(self):
        parameters, optimizer = self.optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            sum(parameter.square() for parameter in parameters).backward()
            optimizer.step()
            scheduler.step()
        checkpoint = copy.deepcopy({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
        parameters, optimizer = self.optimizer()
        restored = build_training_scheduler(optimizer, 10, 4, checkpoint=checkpoint)
        self.assertIsInstance(restored, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(restored.T_max, 50)
        self.assertEqual(restored.last_epoch, 10)
        self.assertEqual(restored.get_last_lr(), checkpoint["scheduler"]["_last_lr"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], restored.get_last_lr())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        for _ in range(4):
            self.update(parameters, optimizer, restored, scaler)
        self.assertEqual(restored.last_epoch, 10)
        restored.step()  # Only the old training epoch boundary advances it.
        for actual, peak in zip(restored.get_last_lr(), (.001, .002)):
            self.assertAlmostEqual(actual, peak * (1 + math.cos(math.pi * 11 / 50)) / 2)
        self.assertEqual(training_schedule_config(restored)["interval"], "epoch")

    def test_amp_overflow_skips_both_optimizer_and_step_scheduler(self):
        parameters, optimizer = self.optimizer()
        scheduler = build_training_scheduler(optimizer, 10, 4)
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        self.assertTrue(self.update(parameters, optimizer, scheduler, scaler))
        before = [parameter.detach().clone() for parameter in parameters]
        previous_lrs = [group["lr"] for group in optimizer.param_groups]
        previous_step = scheduler.last_epoch
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(sum(parameter.square() for parameter in parameters)).backward()
        parameters[0].grad.fill_(float("inf"))
        self.assertFalse(optimizer_step_with_schedule(optimizer, scheduler, scaler, parameters, 1.))
        self.assertEqual(scheduler.last_epoch, previous_step)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], previous_lrs)
        for actual, expected in zip(parameters, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(self.update(parameters, optimizer, scheduler, scaler))
        self.assertEqual(scheduler.last_epoch, previous_step + 1)

    def test_training_defaults_use_ten_epochs_and_validation_does_not_advance_lr(self):
        class TinyVAE(torch.nn.Module):
            config = SimpleNamespace(latent_channels=4)

        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            train_loader = DataLoader(TensorDataset(torch.rand(5, 3, 64, 64), torch.arange(5) % 3), batch_size=2)
            val_loader = DataLoader(TensorDataset(torch.rand(2, 3, 64, 64), torch.arange(2)), batch_size=1)
            updates, validation_steps, used_lrs = [], [], []
            active_scheduler = None

            def step(optimizer, scheduler, scaler, parameters, grad_clip):
                nonlocal active_scheduler
                active_scheduler = scheduler
                used_lrs.append([group["lr"] for group in optimizer.param_groups])
                updated = optimizer_step_with_schedule(optimizer, scheduler, scaler, parameters, grad_clip)
                updates.append(updated)
                return updated

            def train(images, labels, vae, allocator, classifier, *args):
                loss = sum(parameter.square().sum() for parameter in list(allocator.parameters()) + list(classifier.parameters())) * .0001
                batch = len(images)
                return {
                    "loss": loss, "task_loss": loss, "top1_accuracy": loss.new_tensor(.5),
                    "mean_symbol_power": loss.new_full((batch,), 2.),
                    "mean_real_component_power": loss.new_ones(batch), "frame_energy": loss.new_full((batch,), 256.),
                    "group_power": loss.new_ones(batch, 64), "snr": loss.new_ones(batch, 64),
                    "allocation_ratio": loss.new_full((batch, 64), 1 / 64),
                }

            def validate(*args):
                validation_steps.append(active_scheduler.last_epoch)
                return {key: torch.tensor(value) for key, value in {
                    "loss": .5, "task_loss": .5, "top1_accuracy": .5, "top5_accuracy": 1.,
                    "mean_symbol_power": 2., "mean_latent_snr_db": 0.,
                }.items()}

            argv = ["train_power_allocator.py", "--exp_name", "tiny", "--device", "cpu",
                    "--image_size", "64", "--hidden_dim", "8", "--num_classes", "3",
                    "--allocator_lr", ".002", "--log_interval", "1000"]
            try:
                with patch("sys.argv", argv), \
                        patch.object(training, "initialize_distributed", return_value=(torch.device("cpu"), False, 0, 1)), \
                        patch.object(training.AutoencoderKL, "from_pretrained", return_value=TinyVAE()), \
                        patch.object(training, "construct_dataloaders", return_value=(train_loader, val_loader)), \
                        patch.object(training, "build_latent_classifier", return_value=torch.nn.Linear(2, 3)), \
                        patch.object(training, "training_step", side_effect=train), \
                        patch.object(training, "validation_step", side_effect=validate), \
                        patch.object(training, "optimizer_step_with_schedule", side_effect=step):
                    training.run_training()
                output = Path("checkpoints_power/ImageNet/snr_10.0/tiny")
                checkpoint = torch.load(output / "latest.pth", map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["epoch"], 9)
                self.assertEqual(checkpoint["global_step"], 30)
                self.assertEqual(updates, [True] * 30)
                self.assertEqual(validation_steps, [step for epoch in range(1, 11) for step in (epoch * 3, epoch * 3)])
                self.assertEqual(checkpoint["scheduler"]["last_epoch"], 30)
                self.assertEqual(checkpoint["scheduler"]["total_steps"], 30)
                self.assertEqual(checkpoint["scheduler"]["warmup_steps"], 2)
                self.assertEqual(checkpoint["config"]["warmup_epochs"], .5)
                self.assertEqual(checkpoint["config"]["lr_schedule"]["interval"], "optimizer_step")
                self.assertEqual(checkpoint["config"]["lr_schedule"]["peak_lrs"], [.002, .001])
                self.assertEqual([group["lr"] for group in checkpoint["optimizer"]["param_groups"]], [.001, .0005])
                self.assertEqual(used_lrs[0], [.0002, .0001])
                self.assertEqual(used_lrs[2], [.002, .001])
                self.assertTrue((output / "epoch_005.pth").is_file())
                self.assertTrue((output / "epoch_010.pth").is_file())
                log = (output / "train.log").read_text()
                self.assertIn("lr_allocator=", log)
                self.assertIn("lr_classifier=", log)
            finally:
                for handler in training.logging.getLogger("train_power_allocator").handlers:
                    handler.close()


if __name__ == "__main__":
    unittest.main()
