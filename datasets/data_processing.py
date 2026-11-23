import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255
CITYSCAPES_INPUT_HEIGHT = 1024
CITYSCAPES_INPUT_WIDTH = 2048

# Cityscapes labelId -> trainId. All unused and void labels stay at 255.
CITYSCAPES_TRAIN_ID = np.full(
    256,
    CITYSCAPES_IGNORE_INDEX,
    dtype=np.uint8,
)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    CITYSCAPES_TRAIN_ID[_label_id] = _train_id


class CityscapesSemanticDataset(Dataset):
    """Cityscapes RGB images paired with 19-class trainId masks."""

    def __init__(
        self,
        root,
        split,
    ):
        self.root = Path(root).expanduser()
        self.split = split

        image_root = self.root / "leftImg8bit" / split
        target_root = self.root / "gtFine" / split
        image_paths = sorted(image_root.glob("*/*_leftImg8bit.png"))
        if not image_paths:
            raise FileNotFoundError(
                f"Cityscapes images were not found under {image_root}"
            )

        self.samples = []
        for image_path in image_paths:
            base_name = image_path.name.replace("_leftImg8bit.png", "")
            city = image_path.parent.name
            train_id_path = (
                target_root
                / city
                / f"{base_name}_gtFine_labelTrainIds.png"
            )
            label_id_path = (
                target_root / city / f"{base_name}_gtFine_labelIds.png"
            )
            if train_id_path.is_file():
                self.samples.append((image_path, train_id_path, True))
            elif label_id_path.is_file():
                self.samples.append((image_path, label_id_path, False))
            else:
                raise FileNotFoundError(
                    f"No semantic label was found for {image_path}"
                )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resize(image, target):
        size = [CITYSCAPES_INPUT_HEIGHT, CITYSCAPES_INPUT_WIDTH]
        image = TF.resize(
            image,
            size,
            interpolation=InterpolationMode.BILINEAR,
        )
        target = TF.resize(
            target,
            size,
            interpolation=InterpolationMode.NEAREST,
        )
        return image, target

    def __getitem__(self, index):
        image_path, target_path, target_is_train_id = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(target_path) as target_file:
            target = target_file.convert("L")

        image, target = self._resize(image, target)

        image_tensor = TF.to_tensor(image)
        target_array = np.array(target, dtype=np.uint8, copy=True)
        if not target_is_train_id:
            target_array = CITYSCAPES_TRAIN_ID[target_array]
        target_tensor = torch.from_numpy(target_array.astype(np.int64))
        return image_tensor, target_tensor


def construct_cityscapes_dataloaders(
    data_path,
    batch_size=1,
    distributed=False,
    seed=0,
    num_workers=4,
):
    trainset = CityscapesSemanticDataset(
        data_path,
        "train",
    )
    validset = CityscapesSemanticDataset(
        data_path,
        "val",
    )
    train_sampler = (
        DistributedSampler(
            trainset,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(
            validset,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    trainloader = torch.utils.data.DataLoader(
        trainset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        shuffle=False,
        sampler=valid_sampler,
        drop_last=False,
        **loader_options,
    )
    return trainloader, validloader


def construct_dataloaders(
    dataset,
    data_path,
    batch_size=32,
    normalize=True,
    image_size=256,
    distributed=False,
    seed=0,
    num_workers=None,
):
    path = os.path.expanduser(data_path)

    if dataset == "CIFAR10":
        trainset, validset = _build_cifar10(path, normalize, image_size)
        collate_fn = None
    elif dataset == "ImageNet":
        trainset, validset = _build_imagenet(path, normalize, image_size)
        collate_fn = None
    elif dataset == "COCO":
        trainset, validset = _build_coco(path, normalize, image_size)
        collate_fn = _coco_collate_fn
    else:
        raise ValueError("dataset must be 'CIFAR10', 'ImageNet' or 'COCO'")

    train_sampler = (
        DistributedSampler(
            trainset,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(
            validset,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    num_workers = min(torch.get_num_threads(), 4) if num_workers is None else num_workers
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(batch_size, len(trainset)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=min(batch_size, len(validset)),
        shuffle=False,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return trainloader, validloader


def _build_cifar10(data_path, normalize, image_size):
    normalize_transform = (
        [transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        if normalize
        else []
    )
    resize_transform = (
        [transforms.Resize(image_size)]
        if image_size != 32
        else []
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *resize_transform,
                transforms.ToTensor(),
                *normalize_transform,
            ]
        ),
    )
    validset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                *resize_transform,
                transforms.ToTensor(),
                *normalize_transform,
            ]
        ),
    )
    return trainset, validset


def _build_imagenet(data_path, normalize, image_size):
    train_transform, valid_transform = _build_transforms(
        normalize,
        image_size,
    )
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "train"),
        transform=train_transform,
    )
    validset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "val"),
        transform=valid_transform,
    )
    return trainset, validset


def _build_coco(data_path, normalize, image_size):
    train_transform, valid_transform = _build_transforms(
        normalize,
        image_size,
    )
    trainset = torchvision.datasets.CocoDetection(
        root=os.path.join(data_path, "train2017"),
        annFile=os.path.join(
            data_path,
            "annotations",
            "instances_train2017.json",
        ),
        transform=train_transform,
    )
    validset = torchvision.datasets.CocoDetection(
        root=os.path.join(data_path, "val2017"),
        annFile=os.path.join(
            data_path,
            "annotations",
            "instances_val2017.json",
        ),
        transform=valid_transform,
    )
    return trainset, validset


def _build_transforms(normalize, image_size):
    normalize_transform = (
        [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if normalize
        else []
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            *normalize_transform,
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *normalize_transform,
        ]
    )
    return train_transform, valid_transform


def _coco_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
