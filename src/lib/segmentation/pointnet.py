"""
PointNet 语义分割模型 (无 T-Net 简化版).
"""

import torch
import torch.nn as nn
from typing import Optional


class PointNetSeg(nn.Module):
    """PointNet 语义分割: (B, N, 3) + (B, N, 3) geo → (B, N, num_classes)."""

    def __init__(self, num_classes: int = 3, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 + 64, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor, geo_features: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)
        else:
            f = xyz
        f = f.transpose(-1, -2)

        local_feat = self.mlp1(f)
        global_feat = self.mlp2(local_feat)
        global_feat_pooled = global_feat.max(dim=-1, keepdim=True)[0]

        global_expanded = global_feat_pooled.expand(-1, -1, local_feat.shape[-1])
        cat_feat = torch.cat([global_expanded, local_feat], dim=1)

        feat = self.decoder(cat_feat)
        logits = self.head(feat)
        return logits.transpose(-1, -2)
