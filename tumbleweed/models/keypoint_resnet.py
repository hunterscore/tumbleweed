"""ResNet18 backbone with a deconvolution head predicting keypoint heatmaps."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.heatmaps import decode_heatmaps, find_peaks
from tumbleweed.geometry.pnp import solve_pose, select_by_prior
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
        prior_model: Optional[nn.Module] = None,
        prior_image_size: int = 224,
        num_candidates: int = 4,
        reprojection_error: float = 20.0,
    ):
        super().__init__()
        self.network = network
        self.camera = camera
        self.stride = stride
        self.min_confidence = min_confidence
        self.prior_model = prior_model
        self.prior_image_size = prior_image_size
        self.num_candidates = num_candidates
        self.reprojection_error = reprojection_error
        # Heatmap cell -> full-resolution pixel, undoing the resize and stride.
        self.to_full_res = np.array(
            [stride * camera.width / image_size[1], stride * camera.height / image_size[0]]
        )
        self.num_failures = 0
        self.num_solves = 0

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heatmaps = self.network(images)
        translations, quaternions = [], []

        if self.prior_model is None:
            coords, confidence = decode_heatmaps(heatmaps)
            
            keypoints_px = coords.detach().cpu().numpy() * self.to_full_res
            confidence = confidence.detach().cpu().numpy()

            for kp, conf in zip(keypoints_px, confidence):
                result = solve_pose(kp, self.camera, conf, min_confidence=self.min_confidence)
                self.num_solves += 1
                self.num_failures += not result.success
                translations.append(result.translation)
                quaternions.append(result.quaternion)
        else:
            prior_images = F.interpolate(images, size=self.prior_image_size, mode="bilinear", align_corners=False, antialias=True)
            prior_t, prior_q = self.prior_model(prior_images)
            prior_t = prior_t.detach().cpu().numpy()
            prior_q = prior_q.detach().cpu().numpy()
            coords, values = find_peaks(heatmaps, k=self.num_candidates)
            candidates_px = coords.detach().cpu().numpy() * self.to_full_res
            values = values.detach().cpu().numpy()

            for candidate, val, pt, pq in zip(candidates_px, values, prior_t, prior_q):
                points, point_conf = select_by_prior(candidate, val, pq, pt, self.camera)
                result = solve_pose(
                    points,
                    self.camera,
                    point_conf,
                    min_confidence=self.min_confidence,
                    reprojection_error=self.reprojection_error,
                )
                self.num_solves += 1
                self.num_failures += not result.success
                # A failed solve falls back to the prior rather than the identity placeholder.
                translations.append(result.translation if result.success else pt)
                quaternions.append(result.quaternion if result.success else pq)
    
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
