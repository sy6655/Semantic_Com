import sys
from pathlib import Path

import timm
import torch.nn as nn
import torchvision.models as models


RESNETS = {
    "ResNet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "ResNet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "ResNet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

MASK2FORMER_ROOT = Path(__file__).resolve().parent / "Mask2Former"
MASK2FORMER_CONFIG = (
    MASK2FORMER_ROOT
    / "configs/cityscapes/semantic-segmentation/swin/"
    "maskformer2_swin_base_IN21k_384_bs16_90k.yaml"
)
MASK2FORMER_CHECKPOINT = (
    MASK2FORMER_ROOT / "checkpoints/model_final_1c6b65.pkl"
)
SEGFORMER_ROOT = Path(__file__).resolve().parent / "SegFormer"
SEGFORMER_CHECKPOINT = (
    SEGFORMER_ROOT
    / "checkpoints/"
    "segformer_mit-b0_8x1_1024x1024_160k_cityscapes_"
    "20211208_101857-e7f88502.pth"
)
SEGFORMER_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/"
    "segformer_mit-b0_8x1_1024x1024_160k_cityscapes/"
    "segformer_mit-b0_8x1_1024x1024_160k_cityscapes_"
    "20211208_101857-e7f88502.pth"
)

def _construct_mask2former():
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    sys.path.insert(0, str(MASK2FORMER_ROOT))
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(MASK2FORMER_CONFIG))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = str(MASK2FORMER_CHECKPOINT)
    cfg.freeze()

    network = build_model(cfg)
    DetectionCheckpointer(network).load(cfg.MODEL.WEIGHTS)
    return network.eval()


def _construct_segformer():
    import mmseg
    from mmseg.apis import init_model

    config = (
        Path(mmseg.__file__).resolve().parent
        / ".mim/configs/segformer/"
        "segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py"
    )
    checkpoint = (
        str(SEGFORMER_CHECKPOINT)
        if SEGFORMER_CHECKPOINT.is_file()
        else SEGFORMER_CHECKPOINT_URL
    )
    return init_model(
        str(config),
        checkpoint,
        device="cpu",
    ).eval()


def construct_model(model, num_classes=1000, pretrained=False):
    if model in RESNETS:
        build_resnet, weights = RESNETS[model]
        network = build_resnet(weights=weights if pretrained else None)
        network.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False,
        )
        network.fc = nn.Linear(network.fc.in_features, num_classes)
    elif model == "MaxViT-Large":
        network = timm.create_model(
            "maxvit_large_tf_224.in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model == "Mask2Former":
        network = _construct_mask2former()
    elif model == "SegFormer-B0":
        network = _construct_segformer()
    else:
        raise ValueError(
            "model must be ResNet18, ResNet34, ResNet50, MaxViT-Large, "
            "Mask2Former, or SegFormer-B0"
        )

    return network
