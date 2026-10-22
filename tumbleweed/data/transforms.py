"""Sample-dict transforms for SpeedPlusDataset.

SpeedPlusDataset yields a dict (``image``, optionally ``quaternion`` /
``translation`` / ``filename``) rather than a bare tensor, so we wrap
torchvision's image-only transforms instead of composing them directly.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.tango_keypoints import TANGO_KEYPOINTS_BODY

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


class KeypointSampleTransform:
    """Resizes the image and renders Gaussian heatmap targets for the keypoints.

    The keypoints are projected with the *original* intrinsics, so
    ``keypoints_px`` stays in full-resolution pixel coordinates — that is the
    space PnP runs in at inference, and it keeps the camera matrix untouched.
    Only the heatmap targets live in the resized/strided space, reached via the
    separate x/y scale factors (SPEED+'s 1920x1200 does not match the network's
    aspect ratio, so one shared factor would skew the targets).
    """

    def __init__(
        self,
        camera: CameraIntrinsics,
        image_size: Tuple[int, int] = (384, 640),
        stride: int = 4,
        sigma: float = 2.0,
        keypoints_body: np.ndarray = TANGO_KEYPOINTS_BODY,
    ):
        self.camera = camera
        self.image_size = image_size
        self.stride = stride
        self.sigma = sigma
        self.keypoints_body = keypoints_body
        self.heatmap_size = (image_size[0] // stride, image_size[1] // stride)
        self.scale = (image_size[1] / camera.width, image_size[0] / camera.height)
        self.image_transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _render(self, keypoints_hm: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = self.heatmap_size
        target = np.zeros((len(keypoints_hm), height, width), dtype=np.float32)
        visible = np.zeros(len(keypoints_hm), dtype=np.float32)

        radius = int(3 * self.sigma)
        for i, (u, v) in enumerate(keypoints_hm):
            iu, iv = int(round(u)), int(round(v))
            if not (0 <= iu < width and 0 <= iv < height):
                continue
            u0, u1 = max(0, iu - radius), min(width, iu + radius + 1)
            v0, v1 = max(0, iv - radius), min(height, iv + radius + 1)
            us = np.arange(u0, u1, dtype=np.float32)
            vs = np.arange(v0, v1, dtype=np.float32)[:, None]
            target[i, v0:v1, u0:u1] = np.exp(
                -((us - u) ** 2 + (vs - v) ** 2) / (2 * self.sigma**2)
            )
            visible[i] = 1.0

        return torch.from_numpy(target), torch.from_numpy(visible)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(sample)
        sample["image"] = self.image_transform(sample["image"])

        if "quaternion" not in sample:
            return sample

        q = sample["quaternion"].numpy().astype(np.float64)
        t = sample["translation"].numpy().astype(np.float64)
        keypoints_px = project_points(self.keypoints_body, q, t, self.camera)

        scaled = keypoints_px * np.array(self.scale) / self.stride
        target, visible = self._render(scaled)

        sample["heatmaps"] = target
        sample["keypoint_visible"] = visible
        sample["keypoints_px"] = torch.from_numpy(keypoints_px.astype(np.float32))
        return sample


def build_keypoint_transforms(
    camera: CameraIntrinsics,
    image_size: Tuple[int, int] = (384, 640),
    stride: int = 4,
    sigma: float = 2.0,
) -> KeypointSampleTransform:
    return KeypointSampleTransform(camera, image_size=image_size, stride=stride, sigma=sigma)
