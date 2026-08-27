"""ResNet18-based direct pose regression baseline.

A single backbone feeds two linear heads: one regressing translation
(x, y, z in meters) and one regressing a quaternion (w, x, y, z), which is
L2-normalized before being returned so it's always a valid unit rotation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _resnet18_backbone(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return torchvision.models.resnet18(weights=weights)
    except ImportError:  # older torchvision without the weights enum
        return torchvision.models.resnet18(pretrained=pretrained)


class PoseResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        backbone = _resnet18_backbone(pretrained)
        feat_dim = backbone.fc.in_features
        # Keep everything up to (and including) global average pooling.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.translation_head = nn.Linear(feat_dim, 3)
        self.rotation_head = nn.Linear(feat_dim, 4)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (N, 3, H, W) normalized image batch.

        Returns:
            translation: (N, 3) meters.
            quaternion: (N, 4) unit quaternion, scalar-first (w, x, y, z).
        """
        feat = self.backbone(images).flatten(1)
        feat = self.dropout(feat)
        translation = self.translation_head(feat)
        quaternion = F.normalize(self.rotation_head(feat), p=2, dim=-1)
        return translation, quaternion
