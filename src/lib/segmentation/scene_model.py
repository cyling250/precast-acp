# -*- coding: utf-8 -*-
"""场景材质分割网（PointNet++ SSG，5 类：0 地面/背景 1 混凝土 2 钢筋
3 箍筋 4 垫块）。T-037 §7.6 新架构的 DL 步骤——只做分割，不做结构推理。
"""
import torch

from lib.segmentation._base import (FeaturePropagation, SetAbstraction)

N_CLASSES = 5
CLASS_NAMES = ("背景", "混凝土", "钢筋", "箍筋", "垫块")


class SceneSegNet(torch.nn.Module):
    """输入 xyz (B,N,3)（米/毫米均可，球半径按同单位）→ logits (B,5,N)。"""

    def __init__(self, in_ch=0, n_classes=N_CLASSES):
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.15, 32, in_ch, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.40, 48, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.80, 64, 128, [128, 128, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 128])
        self.fp2 = FeaturePropagation(128 + 64, [128, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        self.head = torch.nn.Conv1d(128, n_classes, 1)

    def forward(self, xyz, extra=None):
        l0_xyz = xyz
        l0_feat = extra if extra is not None else \
            torch.zeros(xyz.shape[0], 0, xyz.shape[1], device=xyz.device)
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        f = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        f = self.fp2(l1_xyz, l2_xyz, l1_feat, f)
        f = self.fp1(l0_xyz, l1_xyz, None, f)
        return self.head(f)
