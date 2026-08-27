"""Sample-dict transforms for SpeedPlusDataset.

SpeedPlusDataset yields a dict (``image``, optionally ``quaternion`` /
``translation`` / ``filename``) rather than a bare tensor, so we wrap
torchvision's image-only transforms instead of composing them directly.
"""
from __future__ import annotations

from typing import Any, Dict

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SampleTransform:
    """Applies an image-only torchvision transform to sample["image"]."""

    def __init__(self, image_transform: T.Compose):
        self.image_transform = image_transform

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(sample)
        sample["image"] = self.image_transform(sample["image"])
        return sample


def build_transforms(image_size: int = 224, train: bool = True) -> SampleTransform:
    """Standard ImageNet-normalized resize pipeline for the ResNet18 backbone.

    Note: this resizes the image without adjusting the camera intrinsics.
    That's fine for direct pose regression (the network never sees K), but
    if you add a reprojection-based loss later, rescale ``camera_matrix``
    by the same width/height ratio applied here.
    """
    ops = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return SampleTransform(T.Compose(ops))
