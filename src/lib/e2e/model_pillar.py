# -*- coding: utf-8 -*-
"""BEV 检测网络：pillar 特征 → 2D CNN → CenterNet 三头（热图/偏移/规格）。

设计定案（2026-10-23）：网络只做检测——峰=实例（根数=峰数，动态数量，
无固定槽位匹配）、亚格偏移=位置、规格分类=直径/内径档；无连续回归头
（外露长度/箍筋高度由 bev_dataset.measure_* 几何量测给出）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 箍筋直径产品规格三档（生成器已规格化，2026-10-23 用户拍板）
STIRRUP_DIAM_TABLE = [8.0, 10.0, 12.0]
# 纵筋直径/套筒内径产品规格档（2026-10-23 定案自 model.py 移入；
# v0-v5 清除时 model.py 已删，常量表归并到本保留基件）
REBAR_DIAM_TABLE = [16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0]
SLEEVE_IDIAM_TABLE = [36.0, 38.0, 40.0, 42.0, 46.0, 50.0, 54.0]


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1),
                                nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1),
                                nn.BatchNorm2d(ch))

    def forward(self, x):
        return F.relu(x + self.c2(self.c1(x)), inplace=True)


class PillarBEVNet(nn.Module):
    """输入 (B,7,H,W) pillar 特征（8mm 格）→ 热图/偏移/规格三头（同分辨率）。"""

    def __init__(self, n_spec, ch=48, in_ch=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(4)])
        self.heat = nn.Conv2d(ch, 1, 1)
        self.off = nn.Conv2d(ch, 2, 1)
        self.spec = nn.Conv2d(ch, n_spec, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return {"heat": self.heat(h), "off": self.off(h), "spec": self.spec(h)}


def bev_loss(out, heat, off_t, cls_t, w_off=1.0, w_spec=0.5):
    """CenterNet focal（标准形式：正项 (1-ŷ)ᵅ，负项 (1-y)^β·ŷᵅ；α=2, β=4）
    + 峰格 L1 偏移 + 规格 CE。

    负项的 ŷᵅ 是防塌缩关键：已预测很小的格梯度趋零，否则背景 BCE 的
    总压力会把稀有正类压成全 0（bev_v1 首轮钢筋/箍筋头塌缩的根因）。
    """
    B = heat.shape[0]
    pred = torch.sigmoid(out["heat"].squeeze(1))
    pos = heat == 1.0
    alpha, beta = 2.0, 4.0
    eps = 1e-6
    pos_loss = ((1 - pred) ** alpha * -torch.log(pred.clamp(min=eps))) * pos
    neg_loss = ((1 - heat) ** beta * pred.clamp(max=1 - eps) ** alpha
                * -torch.log((1 - pred).clamp(min=eps))) * (~pos)
    loss_heat = (pos_loss.sum() + neg_loss.sum()) / max(pos.sum(), 1)
    loss_off = torch.zeros((), device=heat.device)
    loss_cls = torch.zeros((), device=heat.device)
    n_cls = 0
    for b in range(B):
        pb = pos[b]
        if pb.any():
            loss_off = loss_off + F.l1_loss(
                out["off"][b][:, pb], off_t[b][:, pb], reduction="sum")
        m = cls_t[b] >= 0
        if m.any():
            loss_cls = loss_cls + F.cross_entropy(
                out["spec"][b].permute(1, 2, 0)[m], cls_t[b][m],
                reduction="sum")
            n_cls += int(m.sum())
    loss = loss_heat + w_off * loss_off / max(pos.sum().item(), 1) \
        + w_spec * loss_cls / max(n_cls, 1)
    return loss, {"heat": float(loss_heat.detach()), "off": float(loss_off),
                  "spec": float(loss_cls)}


@torch.no_grad()
def decode(out, cell, origin=(0.0, 0.0), thresh=0.3, topk=64):
    """3×3 max-pool NMS + 阈值 → 每样本 (N,4+K)：(u_mm, v_mm, score, spec…)。

    返回坐标 = origin + (格 + 亚格偏移)×cell，即与 GT 同一的工程坐标系
    （bev_v1 首轮漏加 origin，位置整体错移一个网格原点，配对全失）。
    """
    h = torch.sigmoid(out["heat"])[:, 0]
    keep = (h == F.max_pool2d(h, 3, 1, 1)) & (h > thresh)
    offs = out["off"].permute(0, 2, 3, 1)
    specs = out["spec"].permute(0, 2, 3, 1)
    res = []
    K, H, W = h.shape
    for b in range(K):
        ys, xs = keep[b].nonzero(as_tuple=True)
        if len(xs) == 0:
            res.append(np.zeros((0, 3 + specs.shape[-1]), np.float32))
            continue
        sc = h[b, ys, xs]
        if len(xs) > topk:
            v, i = sc.topk(topk)
            ys, xs, sc = ys[i], xs[i], v
        u = origin[0] + (xs.float() + offs[b, ys, xs, 0]) * cell
        v = origin[1] + (ys.float() + offs[b, ys, xs, 1]) * cell
        sp = specs[b, ys, xs]
        res.append(torch.cat([u[:, None], v[:, None], sc[:, None],
                              torch.sigmoid(sp)], dim=1).cpu().numpy())
    return res


def match_dets(det_uv, gt_uv, gate=120.0):
    """匈牙利匹配（位置代价，>gate 不配），返回 (det_idx, gt_idx)。

    入口强制 float64——缓存对象数组可能带 object dtype 壳（2026-10-24
    屡次崩溃点），np.linalg.norm 拒收 object。
    """
    from scipy.optimize import linear_sum_assignment
    det_uv = np.asarray(det_uv, np.float64)
    gt_uv = np.asarray(gt_uv, np.float64)
    if len(det_uv) == 0 or len(gt_uv) == 0:
        return [], []
    cost = np.linalg.norm(
        det_uv[:, None, :2] - gt_uv[None, :, :2], axis=2)
    ri, ci = linear_sum_assignment(cost)
    keep = [(i, j) for i, j in zip(ri, ci) if cost[i, j] <= gate]
    return ([i for i, _ in keep], [j for _, j in keep]) if keep else ([], [])
