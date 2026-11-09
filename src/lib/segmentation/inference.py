# -*- coding: utf-8 -*-
"""场景分割模型推理（pred 复验用）：3mm 细化 + 150mm 球补丁铺片 +
PointMLP 逐补丁前向（与训练同归一），全点直接得到标签、无需 kNN 传播。"""
import glob
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from lib.segmentation import PointMLPSeg

CKPT = "out/exp1_train/seg/best.pth"
N_CLASSES = 4  # 垫块并入噪声(0)，与训练口径一致
PATCH_RADIUS_MM = 150.0
NUM_POINTS = 8192
VOXEL_THIN_MM = 3.0


def load_model(ckpt_path=None, device=None):
    path = ckpt_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), CKPT)
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    ck = torch.load(path, map_location=device)
    model = PointMLPSeg(num_classes=N_CLASSES, in_channels=6).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, device


def voxel_thin(pts, *arrays, voxel_mm=VOXEL_THIN_MM):
    keys = np.floor(pts / voxel_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return (pts[idx],) + tuple(a[idx] for a in arrays)


@torch.no_grad()
def predict_labels(pts, model, device, seed=0, batch=8, thin=True):
    """(N,3) 世界系 mm 点 → (N,) 预测材质标签（0-4）。

    batch=8：8192 点全 pairwise 距离矩阵 batch8=2.1GB，再大触发
    WDDM 显存超订阅换页（同 train_scene_seg.TRAIN_POINTS 注）。

    细化后的云按训练同款非重叠铺片；细化剔出的点用 kNN 从补丁点拿标签。"""
    rng = np.random.default_rng(seed)
    if thin:
        thin_pts, back = voxel_thin(pts, np.arange(len(pts)))
        back = np.asarray(back, dtype=np.int64)
    else:
        thin_pts, back = pts, np.arange(len(pts))
    tree = cKDTree(thin_pts)
    available = np.ones(len(thin_pts), bool)
    out = np.full(len(thin_pts), -1, np.int64)
    batch_p, batch_meta = [], []
    n_fail = 0

    def flush():
        if not batch_p:
            return
        x = torch.from_numpy(np.stack(batch_p)).to(device)
        pred = model(x, torch.zeros_like(x)).argmax(-1).cpu().numpy()
        for k, (idx_arr, _scale) in enumerate(batch_meta):
            out[idx_arr] = pred[k]
        batch_p.clear()
        batch_meta.clear()

    while available.sum() >= NUM_POINTS:
        avail_idx = np.flatnonzero(available)
        seed = int(rng.choice(avail_idx))
        ball = tree.query_ball_point(thin_pts[seed], r=PATCH_RADIUS_MM)
        cand = np.array([i for i in ball if available[i]], dtype=np.int64)
        if len(cand) >= NUM_POINTS:
            chosen = rng.choice(cand, NUM_POINTS, replace=False)
        else:
            k_need = min(NUM_POINTS * 2, len(avail_idx))
            _d, knn = tree.query(thin_pts[seed].reshape(1, -1), k=k_need)
            chosen_set = set(cand.tolist())
            for i in knn[0]:
                if available[i]:
                    chosen_set.add(int(i))
                    if len(chosen_set) >= NUM_POINTS:
                        break
            chosen = np.fromiter(chosen_set, np.int64)[:NUM_POINTS]
        if len(chosen) < NUM_POINTS:
            n_fail += 1
            if n_fail >= 50:
                break
            continue
        available[chosen] = False
        pp = thin_pts[chosen]
        pp = pp - pp.mean(0)
        scale = float(np.max(np.linalg.norm(pp, axis=1)))
        pp = pp / (scale + 1e-8)
        batch_p.append(pp.astype(np.float32))
        batch_meta.append((chosen, scale))
        if len(batch_p) >= batch:
            flush()
    flush()
    # 细化剔出的点：从最近补丁点拿标签
    uncovered = np.flatnonzero(out < 0)
    if len(uncovered):
        covered = np.flatnonzero(out >= 0)
        _d, nn = cKDTree(thin_pts[covered]).query(thin_pts[uncovered],
                                                  k=1, workers=-1)
        out[uncovered] = out[covered][nn]
    full = np.empty(len(pts), np.int64)
    full[back] = out
    return full
