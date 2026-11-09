"""
PointMLP 语义分割模型.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from lib.segmentation._base import farthest_point_sample, _index_points, _square_distance


class GeometricAffine(nn.Module):
    """k-NN + 特征差 + 位置差 → MLP → 池化."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 24):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch + 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        B, C, N = f.shape
        sq_dist = _square_distance(p, p)
        _, idx = sq_dist.topk(self.k + 1, dim=-1, largest=False)
        idx = idx[:, :, 1:]

        grouped_p = _index_points(p, idx)
        dp = grouped_p - p.unsqueeze(2)
        dp = dp.permute(0, 3, 1, 2)

        grouped_f = _index_points(f.transpose(-1, -2), idx)
        fj = grouped_f.permute(0, 3, 1, 2)
        fi = f.unsqueeze(-1).expand(-1, -1, -1, self.k)
        df = fj - fi

        inp = torch.cat([dp, df], dim=1)
        out = self.mlp(inp)
        out, _ = out.max(dim=-1)
        return out


class ResMLPBlock(nn.Module):
    """残差 MLP 块: expand → compress + 残差."""

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        mid_ch = int(channels * expansion)
        self.conv1 = nn.Conv1d(channels, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity, inplace=True)
        return x


class PointMLPStage(nn.Module):
    """PointMLP 一级: (可选降采样) → GeometricAffine → ResMLP."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 24,
                 npoint: Optional[int] = None):
        super().__init__()
        self.npoint = npoint
        self.ga = GeometricAffine(in_ch, out_ch, k=k)
        self.res = ResMLPBlock(out_ch, expansion=2.0)
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, p: torch.Tensor, f: torch.Tensor):
        if self.npoint is not None and self.npoint < p.shape[1]:
            fps_idx = farthest_point_sample(p, self.npoint)
            new_p = _index_points(p, fps_idx.unsqueeze(-1)).squeeze(2)
            new_f = _index_points(f.transpose(-1, -2), fps_idx.unsqueeze(-1)).squeeze(2)
            new_f = new_f.transpose(-1, -2)
        else:
            new_p = p
            new_f = f

        f_ga = self.ga(new_p, new_f)
        f_skip = self.skip(new_f)
        f_out = self.res(f_ga + f_skip)
        return new_p, f_out


class PointMLPSeg(nn.Module):
    """PointMLP 语义分割模型."""

    def __init__(self, num_classes: int = 3, in_channels: int = 6,
                 widths: List[int] = None, k: int = 24,
                 nsample: List[int] = None):
        super().__init__()
        if widths is None:
            widths = [64, 128, 256]
        if nsample is None:
            nsample = [2048, 512, 128]

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.embed = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 1, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True),
        )

        encoder = []
        cur_ch = widths[0]
        for w, np_pts in zip(widths, nsample):
            encoder.append(PointMLPStage(cur_ch, w, k=k, npoint=np_pts))
            cur_ch = w
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        dec_in = widths[-1]
        skip_chs = list(reversed(widths[:-1])) + [widths[0]]
        for skip_ch in skip_chs:
            mlp_in = dec_in + skip_ch
            decoder.append(nn.Sequential(
                nn.Conv1d(mlp_in, skip_ch, 1, bias=False),
                nn.BatchNorm1d(skip_ch), nn.ReLU(inplace=True),
                nn.Conv1d(skip_ch, skip_ch, 1, bias=False),
                nn.BatchNorm1d(skip_ch), nn.ReLU(inplace=True),
            ))
            dec_in = skip_ch
        self.decoder = nn.ModuleList(decoder)

        self.head = nn.Sequential(
            nn.Conv1d(dec_in, 64, 1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Conv1d(64, num_classes, 1),
        )

    def _interpolate(self, p1, p2, f2):
        B, N1 = p1.shape[:2]
        N2 = p2.shape[1]
        if N2 == 1:
            return f2.expand(-1, -1, N1)
        sq = _square_distance(p1, p2)
        dist, idx = sq.topk(3, dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)
        feat_src = _index_points(f2.transpose(-1, -2), idx)
        interpolated = (feat_src * w.unsqueeze(-1)).sum(dim=2)
        return interpolated.transpose(-1, -2)

    def forward(self, xyz: torch.Tensor, geo_features: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)
        else:
            f = xyz
        f = f.transpose(-1, -2)
        f = self.embed(f)
        p = xyz

        p_skips, f_skips = [p], [f]
        for stage in self.encoder:
            p, f = stage(p, f)
            p_skips.append(p)
            f_skips.append(f)

        dec_f = f_skips[-1]
        for i, dec_stage in enumerate(self.decoder):
            p_target = p_skips[-(i + 2)]
            f_skip = f_skips[-(i + 2)]
            p_source = p_skips[-(i + 1)]
            f_up = self._interpolate(p_target, p_source, dec_f)
            cat_f = torch.cat([f_up, f_skip], dim=1)
            dec_f = dec_stage(cat_f)

        logits = self.head(dec_f)
        return logits.transpose(-1, -2)
