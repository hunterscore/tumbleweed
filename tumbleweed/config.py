"""Default hyperparameters for training/eval, overridable via CLI args."""
from __future__ import annotations

from dataclasses import dataclass


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
