"""Shared diffusion restoration and image helpers for power allocation."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import center_crop, normalize, to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer

from datasets.data_processing import construct_dataloaders
from models.models import construct_model


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _expand_group_map(
    group_values: torch.Tensor,
    spatial_size: tuple[int, int],
    group_rows: int,
    group_cols: int,
) -> torch.Tensor:
    height, width = spatial_size
    spatial_map = torch.empty(
        group_values.shape[0],
        1,
        height,
        width,
        device=group_values.device,
        dtype=group_values.dtype,
    )
    row_indices = torch.tensor_split(
        torch.arange(height, device=group_values.device),
        group_rows,
    )
    col_indices = torch.tensor_split(
        torch.arange(width, device=group_values.device),
        group_cols,
    )
    for row, rows in enumerate(row_indices):
        for col, cols in enumerate(col_indices):
            group_index = row * group_cols + col
            spatial_map[:, :, rows[:, None], cols] = group_values[
                :, group_index, None, None, None
            ]
    return spatial_map


class StaggeredReverseDiffusion:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        local_files_only: bool,
        revision: str | None = None,
        num_inference_steps: int | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            local_files_only=local_files_only,
            revision=revision,
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            steps_offset=0,
            timestep_spacing=(
                "trailing"
                if num_inference_steps is not None
                else "leading"
            ),
            local_files_only=local_files_only,
            revision=revision,
        )
        num_train_timesteps = int(
            self.scheduler.config.num_train_timesteps
        )
        self.num_inference_steps = (
            num_train_timesteps
            if num_inference_steps is None
            else num_inference_steps
        )
        if not 1 <= self.num_inference_steps <= num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{num_train_timesteps}"
            )
        _freeze(self.unet)
        _freeze(self.text_encoder)

    def inference_timesteps(self, device: torch.device) -> torch.Tensor:
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            device=device,
        )
        return self.scheduler.timesteps

    def _empty_prompt_embedding(self, batch_size: int) -> torch.Tensor:
        tokens = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(
            tokens.input_ids.to(self.device),
        )[0]

    def __call__(
        self,
        received_latent: torch.Tensor,
        reverse_context: dict,
        generator: torch.Generator,
        verbose: bool = True,
    ) -> torch.Tensor:
        num_train_timesteps = int(
            self.scheduler.config.num_train_timesteps
        )
        assigned_timesteps = reverse_context["target_timesteps"].clone()
        missing_groups = assigned_timesteps < 0
        assigned_timesteps[missing_groups] = num_train_timesteps - 1
        timestep_map = _expand_group_map(
            assigned_timesteps,
            received_latent.shape[-2:],
            reverse_context["group_rows"],
            reverse_context["group_cols"],
        )

        reference_noise = torch.randn(
            received_latent.shape,
            device=received_latent.device,
            dtype=received_latent.dtype,
            generator=generator,
        )
        missing_map = _expand_group_map(
            missing_groups,
            received_latent.shape[-2:],
            reverse_context["group_rows"],
            reverse_context["group_cols"],
        )
        received_reference = torch.where(
            missing_map,
            reference_noise,
            received_latent,
        )
        current_latent = received_reference.clone()
        prompt_embedding = self._empty_prompt_embedding(
            received_latent.shape[0]
        )

        inference_timesteps = self.inference_timesteps(
            received_latent.device
        )
        max_timestep = int(timestep_map.max().item())
        reverse_timesteps = inference_timesteps[
            inference_timesteps <= max_timestep
        ]
        alpha_bars = self.scheduler.alphas_cumprod.to(
            device=received_latent.device,
            dtype=received_latent.dtype,
        )

        for step_index, timestep in enumerate(reverse_timesteps):
            timestep_value = int(timestep.item())
            active = timestep_map >= timestep_value
            joining = timestep_map == timestep_value

            alpha_current = alpha_bars[timestep_value]
            alpha_source = alpha_bars[timestep_map]
            alpha_ratio = (alpha_current / alpha_source).clamp(
                min=0.0,
                max=1.0,
            )
            reference_at_t = (
                torch.sqrt(alpha_ratio) * received_reference
                + torch.sqrt(1.0 - alpha_ratio) * reference_noise
            )
            active_state = torch.where(
                joining,
                received_reference,
                current_latent,
            )
            model_input = torch.where(
                active,
                active_state,
                reference_at_t,
            )
            model_input = self.scheduler.scale_model_input(
                model_input,
                timestep,
            )
            latent_height, latent_width = model_input.shape[-2:]
            pad_height = (-latent_height) % 8
            pad_width = (-latent_width) % 8
            unet_input = F.pad(
                model_input,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
            noise_prediction = self.unet(
                unet_input.to(dtype=self.dtype),
                timestep,
                encoder_hidden_states=prompt_embedding,
            ).sample[..., :latent_height, :latent_width].float()
            next_latent = self.scheduler.step(
                noise_prediction,
                timestep,
                model_input.float(),
                eta=0.0,
                generator=generator,
            ).prev_sample
            current_latent = torch.where(
                active,
                next_latent,
                reference_at_t,
            )

            if verbose and (
                step_index == 0 or (step_index + 1) % 50 == 0
            ):
                print(
                    "reverse "
                    f"step={step_index + 1}/{len(reverse_timesteps)} "
                    f"t={timestep_value}"
                )

        return current_latent


def _decode_latent(
    vae: AutoencoderKL,
    latent: torch.Tensor,
) -> torch.Tensor:
    vae_parameter = next(vae.parameters())
    latent = latent.to(
        device=vae_parameter.device,
        dtype=vae_parameter.dtype,
    )
    decoded = vae.decode(
        latent / float(vae.config.scaling_factor)
    ).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def _save_labeled_row(
    images: list[torch.Tensor],
    labels: list[str],
    output_path: Path,
) -> None:
    pil_images = [
        to_pil_image(image.detach().float().cpu().clamp(0.0, 1.0))
        for image in images
    ]
    panel_width, panel_height = pil_images[0].size
    label_height = 24
    canvas = Image.new(
        "RGB",
        (panel_width * len(pil_images), panel_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(pil_images, labels)):
        left = index * panel_width
        canvas.paste(image, (left, label_height))
        draw.text((left + 4, 5), label, fill="black")
    canvas.save(output_path)


def _load_image(
    args: argparse.Namespace,
    config: dict,
) -> tuple[torch.Tensor, int | None]:
    image_size = int(config.get("image_size", 256))
    if args.input_image is not None:
        image = Image.open(args.input_image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        return transform(image).unsqueeze(0), None

    _, validloader = construct_dataloaders(
        dataset=config["dataset"],
        data_path=args.data_path or config["data_path"],
        batch_size=1,
        normalize=False,
        image_size=image_size,
    )
    image, label = validloader.dataset[args.sample_index]
    return image.unsqueeze(0), int(label)


def _task_inference(
    image: torch.Tensor,
    device: torch.device,
    num_classes: int,
    label: int | None,
) -> dict:
    task_model = construct_model(
        "MaxViT-Large",
        num_classes=num_classes,
        pretrained=True,
    ).to(device)
    _freeze(task_model)
    task_parameter = next(task_model.parameters())
    image = image.to(
        device=task_parameter.device,
        dtype=task_parameter.dtype,
    )
    normalized_image = normalize(
        center_crop(image, [224, 224]),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    logits = task_model(normalized_image)
    probabilities = logits.softmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(5, dim=1)
    result = {
        "top5_indices": top_indices[0].cpu().tolist(),
        "top5_probabilities": top_probabilities[0].cpu().tolist(),
    }
    if label is not None:
        top1_correct = bool(top_indices[0, 0].item() == label)
        top5_correct = bool((top_indices[0] == label).any().item())
        result.update(
            {
                "label": label,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "top1_accuracy": float(top1_correct),
                "top5_accuracy": float(top5_correct),
            }
    )
    return result
