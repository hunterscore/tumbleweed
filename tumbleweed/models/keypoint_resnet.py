"""ResNet18 backbone with a deconvolution head predicting keypoint heatmaps."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.heatmaps import decode_heatmaps
from tumbleweed.geometry.pnp import solve_pose
from tumbleweed.geometry.tango_keypoints import NUM_KEYPOINTS


def _resnet18_backbone(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return torchvision.models.resnet18(weights=weights)
    except ImportError:  # older torchvision without the weights enum
        return torchvision.models.resnet18(pretrained=pretrained)


class KeypointResNet18(nn.Module):
    """Predicts one heatmap per keypoint at 1/4 of the input resolution.

    Unlike the regression baseline this keeps the spatial feature map: the
    backbone is cut before global pooling, then three transposed convolutions
    upsample 1/32 -> 1/4 so peaks can be localized to a few pixels.
    """

    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        pretrained: bool = True,
        deconv_channels: int = 256,
    ):
        super().__init__()
        backbone = _resnet18_backbone(pretrained)
        # Drop avgpool + fc, keeping the (N, 512, H/32, W/32) feature map.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        layers = []
        in_channels = 512
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(
                    in_channels, deconv_channels, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(deconv_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = deconv_channels
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(deconv_channels, num_keypoints, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, num_keypoints, H/4, W/4) heatmaps."""
        return self.head(self.deconv(self.backbone(images)))


class KeypointPoseEstimator(nn.Module):
    """Adapts the keypoint network to the ``(translation, quaternion)`` interface.

    ``tumbleweed.eval.evaluate`` drives any module returning that pair, so
    presenting PnP results this way makes the keypoint pipeline scoreable by the
    same SPEC metrics as the regression baseline, with no changes to the
    evaluation loop.
    """

    def __init__(
        self,
        network: KeypointResNet18,
        camera: CameraIntrinsics,
        image_size: Tuple[int, int] = (384, 640),
        stride: int = 4,
        min_confidence: float = 0.1,
    ):
        super().__init__()
        self.network = network
        self.camera = camera
        self.stride = stride
        self.min_confidence = min_confidence
        # Heatmap cell -> full-resolution pixel, undoing the resize and stride.
        self.to_full_res = np.array(
            [stride * camera.width / image_size[1], stride * camera.height / image_size[0]]
        )
        self.num_failures = 0
        self.num_solves = 0

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heatmaps = self.network(images)
        coords, confidence = decode_heatmaps(heatmaps)

        keypoints_px = coords.detach().cpu().numpy() * self.to_full_res
        confidence = confidence.detach().cpu().numpy()

        translations, quaternions = [], []
        for kp, conf in zip(keypoints_px, confidence):
            result = solve_pose(kp, self.camera, conf, min_confidence=self.min_confidence)
            self.num_solves += 1
            self.num_failures += not result.success
            translations.append(result.translation)
            quaternions.append(result.quaternion)

        device = images.device
        return (
            torch.as_tensor(np.stack(translations), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(quaternions), dtype=torch.float32, device=device),
        )

    @property
    def failure_rate(self) -> float:
        return self.num_failures / self.num_solves if self.num_solves else 0.0

    def reset_failure_stats(self) -> None:
        self.num_failures = 0
        self.num_solves = 0
