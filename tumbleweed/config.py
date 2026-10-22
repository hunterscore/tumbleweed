"""Default hyperparameters for training/eval, overridable via CLI args."""
from __future__ import annotations

from dataclasses import dataclass


from typing import Tuple


@dataclass
class TrainConfig:
    data_root: str = "data/speedplusv2"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    rotation_weight: float = 1.0
    translation_weight: float = 1.0
    pretrained_backbone: bool = True
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    seed: int = 0


@dataclass
class KeypointTrainConfig:
    """Keypoint + PnP pipeline. Higher resolution than the regression baseline:
    heatmap peaks have to be localized to a few pixels, which a 224x224 squash
    of a 1920x1200 frame cannot support."""

    data_root: str = "data/speedplusv2"
    image_size: Tuple[int, int] = (384, 640)
    stride: int = 4
    sigma: float = 2.0
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    peak_weight: float = 20.0
    min_confidence: float = 0.1
    pretrained_backbone: bool = True
    checkpoint_dir: str = "checkpoints/keypoint"
    device: str = "cuda"
    seed: int = 0
