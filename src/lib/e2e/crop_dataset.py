# -*- coding: utf-8 -*-
"""端部裁剪数据集：合成构件 NPZ → 钢筋端/套筒端参数检测样本。

设计（用户 2026-10-23 拍板）：
- 只切钢筋/套筒端部区域进网络，混凝土身体不进（消除类不平衡）；
- 绝对尺度保留（毫米制，不平移 GT 语义、不缩放）——直径/位置是绝对量；
- 增广只扰动朝向（截面内旋转/翻转）、密度（随机丢点）、点云分布（jitter），
  不改变绝对尺度——扰乱 T-022 型捷径。
- GT 参数化（端面中心原点、轴向朝外为 +t）：
  钢筋端样本：n_rebars + 每根 (u, v, d_nominal)；
  套筒端样本：n_sleeves + 每环 (u, v, r_inner)。
"""

import glob
import json
import os

import numpy as np

# 窗口（mm，绝对尺度）：钢筋端含端面内侧 300mm 混凝土 + 外露全长（最长
# ~1500mm + 垫片）；套筒端含端面内侧 300mm + 环凸外 ~80mm。
REBAR_WIN = (-300.0, 1750.0)
SLEEVE_WIN = (-300.0, 80.0)
N_INPUT = 32768


def _load_npz(path):
    d = np.load(path, allow_pickle=False)
    pts = d["points"].astype(np.float32)
    lbls = d["labels"].astype(np.int64)
    meta = json.loads(str(d["meta_json"]))
    return pts, lbls, meta


def _end_face_center(pts, lbls, end_x):
    """端面中心 (y, z)：端面附近 50mm 带内混凝土点的中位数。"""
    band = (np.abs(pts[:, 0] - end_x) < 50.0) & (lbls == 1)
    if band.sum() < 50:  # 端面遮挡严重的兜底：全部混凝土点
        band = lbls == 1
    return float(np.median(pts[band, 1])), float(np.median(pts[band, 2]))


def make_rebar_sample(pts, lbls, meta):
    """钢筋端样本：点云（t,u,v；端面中心原点、外露朝 +t）+ 参数 GT。"""
    end_x = float(meta["rebar_end_x"])
    cy, cz = _end_face_center(pts, lbls, end_x)
    t = pts[:, 0] - end_x
    if float(meta["sleeve_end_x"]) > end_x:  # 钢筋端在负侧 → 外露朝 -x
        t = -t
    win = (t > REBAR_WIN[0]) & (t < REBAR_WIN[1])
    p = np.column_stack([t[win], pts[win, 1] - cy, pts[win, 2] - cz])
    gt = np.array([[u, v, float(meta["rebar_diam"])]
                   for (u, v) in meta["rebar_positions_yz"]], dtype=np.float32)
    return p.astype(np.float32), gt, int(meta["n_rebars"])


def make_sleeve_sample(pts, lbls, meta):
    """套筒端样本：点云（t,u,v）+ 每环 (u, v, r_inner) GT。"""
    end_x = float(meta["sleeve_end_x"])
    cy, cz = _end_face_center(pts, lbls, end_x)
    t = pts[:, 0] - end_x
    if float(meta["rebar_end_x"]) > end_x:  # 套筒端在负侧 → 外法向朝 -x
        t = -t
    win = (t > SLEEVE_WIN[0]) & (t < SLEEVE_WIN[1])
    p = np.column_stack([t[win], pts[win, 1] - cy, pts[win, 2] - cz])
    gt = np.array([[g["center_yz"][0] - cy, g["center_yz"][1] - cz,
                    g["inner_diam"] / 2.0] for g in meta["sleeve_gt"]],
                  dtype=np.float32)
    return p.astype(np.float32), gt, len(meta["sleeve_gt"])


def _endband_bbox(p, half=50.0):
    """端面带（|t|<half）点的 (u,v) 稳健包围盒（0.5/99.5 分位抗离群）。"""
    m = np.abs(p[:, 0]) < half
    if m.sum() < 50:
        m = np.ones(len(p), dtype=bool)
    u0, u1 = np.percentile(p[m, 1], [0.5, 99.5])
    v0, v1 = np.percentile(p[m, 2], [0.5, 99.5])
    return u0, u1, v0, v1


def augment(p, gt, rng, rot=True, flip=True, drop=(0.1, 0.5),
            jitter=1.5, n_input=N_INPUT):
    """保尺度增广 + 端面包围盒归一化（用户 2026-10-23 拍板）。

    流程：截面内旋转/翻转/丢点/jitter（全部保绝对尺度）→ 以端面带稳健
    包围盒为参考系：输入点云原点平移到包围盒左上角（保持 mm 制，不缩放——
    缩放会抹掉直径的绝对尺度），GT 圆心转为包围盒内归一化坐标 (0,1)，
    左上原点、v 自上而下。
    返回 (p, gt_rel, (w, h))；还原绝对位置：u = u0 + u_rel×w（w,h 随样本
    返回，评估还原与推理均可用——bbox 从输入点云即可算出，不依赖 GT）。
    """
    if rot:
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        p = p @ R.T
        gt = gt.copy()
        gt[:, 0], gt[:, 1] = (gt[:, 0] * c - gt[:, 1] * s,
                              gt[:, 0] * s + gt[:, 1] * c)
    if flip and rng.random() < 0.5:
        p[:, 2] = -p[:, 2]
        gt[:, 1] = -gt[:, 1]
    if drop is not None:
        p = p[rng.random(len(p)) > rng.uniform(*drop)]
    if jitter > 0:
        p = p + rng.normal(0, rng.uniform(0, jitter), p.shape).astype(np.float32)
    u0, u1, v0, v1 = _endband_bbox(p)
    w = max(u1 - u0, 1.0)
    h = max(v1 - v0, 1.0)
    p = p.copy()
    p[:, 1] -= u0
    p[:, 2] -= v1  # 左上角原点（v 轴向上，左上 = v_max）
    gt_rel = gt.copy()
    gt_rel[:, 0] = (gt[:, 0] - u0) / w
    gt_rel[:, 1] = (v1 - gt[:, 1]) / h
    if len(p) >= n_input:
        idx = rng.choice(len(p), n_input, replace=False)
    else:  # 不足重采样补齐（外露筋端点少时）
        idx = rng.choice(len(p), n_input, replace=True)
    return p[idx].astype(np.float32), gt_rel, (w, h)


def load_split(root, split):
    """目录下全部 NPZ 路径（训练/验证/测试同 mock/inuse 划分）。"""
    return sorted(glob.glob(os.path.join(root, split, "*.npz")))
