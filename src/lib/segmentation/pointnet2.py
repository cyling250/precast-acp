"""
PointNet++ 分割模型 + Gated PointNet++.
"""

import torch
import torch.nn as nn
from typing import Optional

from lib.segmentation._base import SetAbstraction, FeaturePropagation


class PointNetPlusPlus(nn.Module):
    """(B, N, 3) → (B, N, num_classes)."""

    def __init__(self, num_classes: int = 3, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.4, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(1, None, None, 128 + 3, [128, 256, 512])
        self.fp3 = FeaturePropagation(128 + 512, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256, [128, 128])
        self.fp1 = FeaturePropagation(128, [64, 32])
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor, geo_features: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None

        if self.in_channels > 3 and geo_features is not None:
            feat_for_sa1 = geo_features.transpose(-1, -2)
        else:
            feat_for_sa1 = None
        l1_xyz, l1_f = self.sa1(xyz, features=feat_for_sa1, radius_override=r1)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f, radius_override=r2)
        l3_xyz, l3_f = self.sa3(l2_xyz, l2_f)

        fp3 = self.fp3(l2_xyz, l3_xyz, l2_f, l3_f)
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_f, fp3)
        fp1 = self.fp1(xyz, l1_xyz, None, fp2)
        logits = self.head(fp1)
        return logits.transpose(-1, -2)


class GatedPointNetPlusPlus(nn.Module):
    """门控 PointNet++: 共享 backbone → coarse_head(3类) + fine_head(2类)."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3,
                 lambda_fine: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lambda_fine = lambda_fine

        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.4, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(1, None, None, 128 + 3, [128, 256, 512])
        self.fp3 = FeaturePropagation(128 + 512, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256, [128, 128])
        self.fp1 = FeaturePropagation(128, [64, 32])

        self.coarse_head = nn.Sequential(
            nn.Conv1d(32, 32, 1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Conv1d(32, 3, 1),
        )
        self.fine_head = nn.Sequential(
            nn.Conv1d(32, 32, 1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Conv1d(32, 2, 1),
        )

    def _encode_decode(self, xyz, geo_features, scale):
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None
        if self.in_channels > 3 and geo_features is not None:
            feat_for_sa1 = geo_features.transpose(-1, -2)
        else:
            feat_for_sa1 = None
        l1_xyz, l1_f = self.sa1(xyz, features=feat_for_sa1, radius_override=r1)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f, radius_override=r2)
        l3_xyz, l3_f = self.sa3(l2_xyz, l2_f)
        fp3 = self.fp3(l2_xyz, l3_xyz, l2_f, l3_f)
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_f, fp3)
        fp1 = self.fp1(xyz, l1_xyz, None, fp2)
        return fp1

    def forward(self, xyz, geo_features=None, scale=None, labels=None):
        import torch.nn.functional as F
        fp1 = self._encode_decode(xyz, geo_features, scale)
        coarse_logits = self.coarse_head(fp1)
        fine_logits = self.fine_head(fp1)

        if self.training and labels is not None:
            return self._forward_loss(coarse_logits, fine_logits, labels, xyz.device)
        return self._forward_eval(coarse_logits, fine_logits, xyz.device)

    def _forward_loss(self, coarse_logits, fine_logits, labels, device):
        B, N = labels.shape
        coarse_labels = labels.clone()
        coarse_labels[coarse_labels == 3] = 2
        coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)

        import torch.nn.functional as F
        fine_labels = torch.full_like(labels, -1)
        fine_labels[labels == 2] = 0
        fine_labels[labels == 3] = 1
        fine_loss = F.cross_entropy(fine_logits, fine_labels, ignore_index=-1)
        total_loss = coarse_loss + self.lambda_fine * fine_loss

        B4 = torch.zeros(B, 4, N, device=device)
        B4[:, 0, :] = coarse_logits[:, 0, :]
        B4[:, 1, :] = coarse_logits[:, 1, :]
        B4[:, 2, :] = coarse_logits[:, 2, :]
        B4[:, 3, :] = coarse_logits[:, 2, :]
        metal = (labels == 2) | (labels == 3)
        B4[:, 2, :][metal] = fine_logits[:, 0, :][metal]
        B4[:, 3, :][metal] = fine_logits[:, 1, :][metal]
        return B4.permute(0, 2, 1), total_loss, coarse_loss, fine_loss

    @torch.no_grad()
    def _forward_eval(self, coarse_logits, fine_logits, device):
        B, N = coarse_logits.shape[0], coarse_logits.shape[2]
        coarse_pred = coarse_logits.argmax(dim=1)
        out = torch.zeros(B, N, 4, device=device)
        out[:, :, 0] = coarse_logits[:, 0, :]
        out[:, :, 1] = coarse_logits[:, 1, :]
        out[:, :, 2] = coarse_logits[:, 2, :]
        out[:, :, 3] = coarse_logits[:, 2, :]
        metal = (coarse_pred == 2)
        out[:, :, 2][metal] = fine_logits[:, 0, :][metal]
        out[:, :, 3][metal] = fine_logits[:, 1, :][metal]
        return out
