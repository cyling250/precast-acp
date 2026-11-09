"""
PointNeXt 语义分割模型.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from lib.segmentation._base import farthest_point_sample, ball_query, _index_points, _square_distance


# ── Channel mapping ──

def _map_channels(feature_type: str, in_ch: int) -> int:
    if feature_type == 'dp_fj':
        return 3 + in_ch
    elif feature_type == 'fj':
        return in_ch
    elif feature_type == 'dp':
        return 3
    elif feature_type == 'df':
        return in_ch
    elif feature_type == 'dp_fj_df':
        return 3 + in_ch * 2
    raise ValueError(f"Unknown feature_type: {feature_type}")


def _get_aggregation_features(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        return torch.cat([dp, fj], dim=1)
    elif feature_type == 'fj':
        return fj
    elif feature_type == 'dp':
        return dp
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        return torch.cat([dp, fj, df], dim=1)
    elif feature_type == 'df':
        df = fj - f.unsqueeze(-1)
        return df
    return torch.cat([dp, fj], dim=1)


# ── Conv blocks ──

class Conv1dBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x): return self.act(self.bn(self.conv(x)))


class Conv2dBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x): return self.act(self.bn(self.conv(x)))


# ── LocalAggregation ──

class LocalAggregation(nn.Module):
    def __init__(self, channels: List[int], radius: float = 0.1,
                 nsample: int = 16, feature_type: str = 'dp_fj',
                 reduction: str = 'max'):
        super().__init__()
        self.feature_type = feature_type
        self.radius = radius
        self.nsample = nsample
        ch0 = _map_channels(feature_type, channels[0])
        convs = []
        for i in range(len(channels) - 1):
            last = (i == len(channels) - 2)
            convs.append(Conv2dBN(ch0 if i == 0 else channels[i],
                                  channels[i + 1], act=not last))
        self.convs = nn.Sequential(*convs)
        if reduction == 'max':
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        elif reduction == 'mean':
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        B, N = p.shape[:2]
        idx = ball_query(self.radius, self.nsample, p, p)
        grouped_xyz = _index_points(p, idx)
        dp = grouped_xyz - p.unsqueeze(2)
        dp = dp.permute(0, 3, 1, 2)
        if f is not None:
            fj = _index_points(f.transpose(-1, -2), idx).permute(0, 3, 1, 2)
        else:
            fj = None
        agg = _get_aggregation_features(p, dp, f, fj, self.feature_type)
        out = self.convs(agg)
        out = self.pool(out)
        return out


# ── InvResMLP ──

class InvResMLP(nn.Module):
    def __init__(self, channels: int, expansion: float = 4.0,
                 radius: float = 0.1, nsample: int = 16,
                 feature_type: str = 'dp_fj'):
        super().__init__()
        mid_channels = int(channels * expansion)
        self.local = LocalAggregation(
            [channels, channels], radius=radius, nsample=nsample,
            feature_type=feature_type,
        )
        self.pw = nn.Sequential(
            Conv1dBN(channels, mid_channels, act=True),
            Conv1dBN(mid_channels, channels, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        identity = f
        f = self.local(p, f)
        f = self.pw(f)
        if f.shape == identity.shape:
            f = self.act(f + identity)
        else:
            f = self.act(f)
        return p, f


# ── PointNeXt SA ──

class PointNextSetAbstraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 radius: float = 0.1, nsample: int = 16,
                 feature_type: str = 'dp_fj', use_res: bool = False,
                 num_layers: int = 1):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.nsample = nsample
        self.feature_type = feature_type
        self.use_res = use_res
        mid_ch = out_channels // 2 if stride > 1 else out_channels
        ch = [in_channels] + [mid_ch] * (num_layers - 1) + [out_channels]
        ch[0] = _map_channels(feature_type, ch[0])
        convs = []
        for i in range(len(ch) - 1):
            convs.append(Conv2dBN(ch[i], ch[i + 1],
                                  act=not (use_res and i == len(ch) - 2)))
        self.convs = nn.Sequential(*convs)
        if use_res:
            self.skip = (Conv1dBN(in_channels, out_channels, act=False)
                         if in_channels != out_channels else nn.Identity())
            self.act = nn.ReLU(inplace=True)
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> tuple:
        B, N, _ = p.shape
        if self.stride > 1:
            npoint = N // self.stride
            fps_idx = farthest_point_sample(p, npoint)
            new_p = _index_points(p, fps_idx.unsqueeze(-1)).squeeze(2)
            idx = fps_idx
        else:
            new_p = p
            idx = None

        idx_bq = ball_query(self.radius, self.nsample, p, new_p)
        grouped_xyz = _index_points(p, idx_bq)
        dp = grouped_xyz - new_p.unsqueeze(2)
        dp = dp.permute(0, 3, 1, 2)

        if f is not None:
            fj = _index_points(f.transpose(-1, -2), idx_bq).permute(0, 3, 1, 2)
        else:
            fj = None

        agg = _get_aggregation_features(new_p, dp, f, fj, self.feature_type)
        out = self.convs(agg)
        out = self.pool(out)

        if self.use_res and idx is not None and f is not None:
            gathered = _index_points(f.transpose(-1, -2), idx.unsqueeze(-1)).squeeze(2)
            gathered = gathered.transpose(-1, -2)
            out = self.act(out + self.skip(gathered))
        return new_p, out


# ── PointNeXt FP ──

class PointNextFeaturePropagation(nn.Module):
    def __init__(self, mlp: List[int]):
        super().__init__()
        convs = []
        for i in range(len(mlp) - 1):
            convs.append(Conv1dBN(mlp[i], mlp[i + 1], act=True))
        self.convs = nn.Sequential(*convs)

    def forward(self, p1: torch.Tensor, f1: Optional[torch.Tensor],
                p2: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B, N1, _ = p1.shape
        N2 = p2.shape[1]
        if N2 == 1:
            interpolated = f2.expand(-1, -1, N1)
        else:
            sq = _square_distance(p1, p2)
            dist, idx = sq.topk(3, dim=-1, largest=False)
            w = 1.0 / (dist + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            feat_src = _index_points(f2.transpose(-1, -2), idx)
            interpolated = (feat_src * w.unsqueeze(-1)).sum(dim=2)
            interpolated = interpolated.transpose(-1, -2)

        cat = interpolated if f1 is None else torch.cat([interpolated, f1], dim=1)
        return self.convs(cat)


# ── PointNeXt Segmentation ──

class PointNeXtSeg(nn.Module):
    """PointNeXt 语义分割模型."""

    def __init__(self, num_classes: int = 3, in_channels: int = 6,
                 width: int = 32, blocks: List[int] = None,
                 strides: List[int] = None, expansion: float = 4.0,
                 nsample: int = 24, radius: float = 0.1,
                 use_res_sa: bool = False):
        super().__init__()
        if blocks is None:
            blocks = [1, 2, 3, 2, 3]
        if strides is None:
            strides = [4, 4, 4, 4]

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.blocks = blocks
        self.strides = strides
        self.radius_base = radius

        self.enc_channels = []
        encoder = []
        cur_ch = in_channels
        w = width
        for stage_idx, (n_blocks, stride) in enumerate(zip(blocks, strides)):
            if stride != 1 and stage_idx > 0:
                w *= 2
            stage_channels = w
            self.enc_channels.append(stage_channels)
            stage_modules = []
            stage_modules.append(PointNextSetAbstraction(
                in_channels=cur_ch, out_channels=stage_channels,
                stride=stride, radius=self.radius_base,
                nsample=nsample, use_res=use_res_sa, num_layers=1,
            ))
            cur_ch = stage_channels
            for _ in range(n_blocks - 1):
                stage_modules.append(InvResMLP(
                    channels=cur_ch, expansion=expansion,
                    radius=self.radius_base, nsample=nsample,
                ))
            encoder.append(nn.Sequential(*stage_modules))
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        dec_in = self.enc_channels[-1]
        dec_skip_ch = max(32, self.enc_channels[0])
        self.proj_skip = Conv1dBN(in_channels, dec_skip_ch, act=True)
        skip_list = list(reversed(self.enc_channels[:-1])) + [dec_skip_ch]
        for skip_ch in skip_list:
            mlp = [skip_ch + dec_in, skip_ch, skip_ch]
            decoder.append(PointNextFeaturePropagation(mlp))
            dec_in = skip_ch
        self.decoder = nn.ModuleList(decoder)

        head_in = dec_in
        self.head = nn.Sequential(
            Conv1dBN(head_in, 64, act=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor, geo_features: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N = xyz.shape[:2]
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)
        else:
            f = xyz
        f = f.transpose(-1, -2)

        p_orig = xyz
        f_orig = f

        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r_eff = 80.0 / scale.view(-1, 1).float()
        else:
            r_eff = None

        p_skips = [p_orig]
        f_skips = [self.proj_skip(f_orig)]

        p = xyz
        for stage in self.encoder:
            for module in stage:
                if isinstance(module, PointNextSetAbstraction):
                    if r_eff is not None:
                        module.radius = r_eff.mean().item()
                elif isinstance(module, InvResMLP):
                    if r_eff is not None:
                        module.local.radius = (r_eff.mean().item() * 0.5)
                p, f = module(p, f)
            p_skips.append(p)
            f_skips.append(f)

        f = f_skips[-1]
        for i, dec_stage in enumerate(self.decoder):
            p_target = p_skips[-(i + 2)]
            f_skip = f_skips[-(i + 2)]
            f = dec_stage(p_target, f_skip, p_skips[-(i + 1)], f)

        logits = self.head(f)
        return logits.transpose(-1, -2)
