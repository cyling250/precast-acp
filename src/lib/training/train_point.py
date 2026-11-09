# -*- coding: utf-8 -*-
"""点级圆检测头训练入口（2026-10-24 用户指令：GT 面口径下把检测头做到
符合并远超 GB 50204-2015 允许偏差——插筋中心线位置 5mm / 外露 +10−5mm）。

与 bev_v6d 严格同数据同口径（复用 out/exp1_train/cache/bev 的原始面点集与 GT；
该缓存存的就是点集，柱化在原入口取样时才做——本入口直接消费点集），
评估指标一一对应（count_acc/n_bias/recall/precision/pos_p50/spec_hit/
expo_p50/腿位/高度/直径命中 + 交叉负视图 + 真实 oracle-crop），可直接
A/B 对照。新增：训练集同轨评估（拟合/泛化分解，用户 2026-10-24 提问）。

v2（T-035 下一步 1「v2 打包」，2026-10-24）：cosine LR 1e-3→1e-4 + 2 轮
warmup；60 轮 + batch 4（步数 ×4）；规格平衡采样器（batch 内凑多规格，
治"记本批的径"）；w_vote=2.0；--v1 开关复现 v1 架构供基线复训。

用法（cwd=src）：python -m experiments.exp1_train point --epochs 60
（权重落盘 out/exp1_train/head_point；训练缓存 out/exp1_train/cache/bev）
"""
import json
import math
import os
import time
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import set_seed, anchor_project_path
from lib.e2e.bev_dataset import (measure_exposed, measure_height,
                                 prefilter_far, rot_flip_stirrup,
                                 _rot_flip_points, _drop_jitter,
                                 load_bev_cache, REBAR_CAP, SLEEVE_CAP)
from lib.e2e.point_head import (PointCircleNet, point_loss, detect,
                                make_seeds, assign_targets, VOTE_RANGE,
                                R_OBJ, R_OBJ_SLEEVE, RADIAL_BINS)
from lib.e2e.model_pillar import (match_dets, STIRRUP_DIAM_TABLE,
                                  REBAR_DIAM_TABLE, SLEEVE_IDIAM_TABLE)

CACHE = "out/exp1_train/cache/bev"
POS_GATE = 120.0
N_EVAL = 5
NEG_P = 0.25        # 交叉负样本（P-038：按头配比，钢筋头 0.5 曾塌）
FACE_JITTER = 25.0
SEED_VOX_END = 10.0   # 端面视图种子体素
SEED_VOX_TOP = 20.0   # 顶面视图大（3584×1024mm），种子放粗
SLEEVE_THR = 0.3    # 套筒工作点（P-048 诊断：套筒 conf 整体偏低是标定
                    # 问题非检测失败——漏检/检出 conf p50 0.37/0.69 分离
                    # 良好；thr0.3 召回 92%/误检 6%，与 bev_v6d 同档）


def _spec_id(table, diam):
    return int(np.argmin(np.abs(np.asarray(table, np.float64) - float(diam))))


def circle_iou_np(c1, r1, c2, r2):
    """圆 IoU（numpy 闭式，评估用，与 point_head.circle_iU 同式）。

    预测圆=（检出中心, 规格表直径/2）；GT 圆=（GT 中心, 名义直径/2）——
    与圆 IoU 损失、装配判废判据同口径。
    """
    d = float(np.linalg.norm(np.asarray(c1) - np.asarray(c2)))
    r1, r2 = float(r1), float(r2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return min(r1, r2) ** 2 / max(r1, r2) ** 2
    a1 = np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    inter = (r1 * r1 * a1 + r2 * r2 * a2
             - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2)
                             * (d - r1 + r2) * (d + r1 + r2)))
    return float(inter / (np.pi * r1 ** 2 + np.pi * r2 ** 2 - inter))


class SpecBalancedSampler(torch.utils.data.Sampler):
    """按规格分组的轮转采样器（T-035 病因④：v1 每件构件筋全同径，随机
    batch 内仅 1–2 个规格值 → 规格头学成"记本批的径"）。

    索引空间 = Dataset 展开空间（i → item i%N）；每组内洗牌后按组轮转
    输出，使连续 batch_size 个索引来自不同规格组。"""

    def __init__(self, n_items, spec_of, rep=2, seed=0):
        self.N, self.rep, self.seed, self._call = n_items, rep, seed, 0
        groups = {}
        for j in range(n_items):
            groups.setdefault(spec_of(j), []).append(j)
        self.groups = groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._call)
        self._call += 1
        pools = {s: rng.permutation(items).tolist() * self.rep
                 for s, items in self.groups.items()}
        order, keys = [], list(pools)
        while any(pools[s] for s in keys):
            rng.shuffle(keys)
            for s in keys:
                if pools[s]:
                    order.append(pools[s].pop())
        return iter(order)

    def __len__(self):
        return self.N * self.rep


class PtDataset(Dataset):
    """kind ∈ rebar/sleeve/stirrup；数据与增广严格对齐 train_e2e_bev 的
    BevDataset（rot/flip/drop/jitter/face-jitter/prefilter_far/交叉负样本），
    差别仅在视图→种子而非柱化。"""

    def __init__(self, z, kind, idx, train):
        self.z, self.kind, self.idx, self.train = z, kind, idx, train
        if kind == "rebar":
            for k in ("rb_pts", "rb_uv", "rb_diam", "rbn_pts"):
                setattr(self, k, z[k])
        elif kind == "sleeve":
            for k in ("sl_pts", "sl_gt", "sln_pts"):
                setattr(self, k, z[k])
        else:
            for k in ("st_pts", "st_gt", "st_topz"):
                setattr(self, k, z[k])
        self.repeat = 2 if train else 1
        self.rng = np.random.default_rng(0)
        self.table = {"rebar": REBAR_DIAM_TABLE, "sleeve": SLEEVE_IDIAM_TABLE,
                      "stirrup": STIRRUP_DIAM_TABLE}[kind]

    def __len__(self):
        return len(self.idx) * self.repeat

    def _sample(self, j, rng):
        kind = self.kind
        if kind == "rebar":
            neg = self.train and rng.random() < NEG_P
            p = (self.rbn_pts[j] if neg else self.rb_pts[j]).astype(np.float32)
            uv = self.rb_uv[j].astype(np.float32)
            diam = float(self.rb_diam[j])
            p = prefilter_far(p, 1, 2, 0)
            if self.train:
                p, uv = _rot_flip_points(p, 1, 2, rng, gt=uv, gt_uv=0)
                p = _drop_jitter(p, rng)
                p[:, 0] += rng.uniform(-FACE_JITTER, FACE_JITTER)
            specs = (np.array([_spec_id(self.table, diam) for _ in uv],
                              np.int64) if len(uv) else np.zeros(0, np.int64))
        elif kind == "sleeve":
            neg = self.train and rng.random() < NEG_P
            if neg:
                p = self.sln_pts[j].astype(np.float32)
                gt = np.zeros((0, 3), np.float32)
            else:
                p = self.sl_pts[j].astype(np.float32)
                gt = self.sl_gt[j].astype(np.float32)  # (K,3) u,v,内径
            if self.train:
                p, gt = _rot_flip_points(p, 1, 2, rng, gt=gt, gt_uv=0)
                p = _drop_jitter(p, rng)
                p[:, 0] += rng.uniform(-FACE_JITTER, FACE_JITTER)
            uv = gt[:, :2] if len(gt) else np.zeros((0, 2), np.float32)
            specs = (np.array([_spec_id(self.table, g[2]) for g in gt],
                              np.int64) if len(gt) else np.zeros(0, np.int64))
        else:
            p = self.st_pts[j].astype(np.float32)
            gt = self.st_gt[j].astype(np.float32)  # (M,6)
            top_z = float(self.st_topz[j])
            if self.train:
                p, gt = rot_flip_stirrup(p, gt, rng)
                p = _drop_jitter(p, rng)
            legs = gt[:, :4].reshape(-1, 2) if len(gt) else np.zeros((0, 2),
                                                                     np.float32)
            uv = legs
            specs = (np.repeat([_spec_id(self.table, g[5]) for g in gt], 2)
                     if len(gt) else np.zeros(0, np.int64))
        # ── 种子与目标 ──
        # 轴序约定：深度（凸出方向）必须在 0 轴——端面视图 (t,u,v) 天然
        # 满足；箍筋顶视图 (x,y,z) 须重排为 (z,x,y)，检测平面=(x,y)=轴1,2。
        vox = SEED_VOX_TOP if kind == "stirrup" else SEED_VOX_END
        p_det = (np.column_stack([p[:, 2], p[:, 0], p[:, 1]])
                 if kind == "stirrup" else p)
        # 检测平面原始点（径向剖面用，v3）：端面=(u,v)=p[:,1:3]；箍筋=(x,y)
        pts_uv = (p[:, :2].astype(np.float32) if kind == "stirrup"
                  else p[:, 1:3].astype(np.float32))
        from lib.e2e.point_head import radial_profile
        radial = (np.stack([radial_profile(pts_uv, c) for c in uv], 0)
                  if len(uv) else np.zeros((0, RADIAL_BINS), np.float32))
        seeds, nb, extra, nb2 = make_seeds(p_det, seed_vox=vox, rng=rng)
        # obj 正类盘按视图分半径：套筒孔心空、环壁 18–35mm（R_OBJ_SLEEVE），
        # 钢筋/箍筋为实心筋（24mm 收窄盘，与局部证据对齐）
        gt_idx, vote, pos_obj = assign_targets(
            seeds, uv, r_obj=(R_OBJ_SLEEVE if kind == "sleeve" else R_OBJ))
        pos = gt_idx >= 0
        spec_t = (np.where(pos, specs[np.clip(gt_idx, 0, None)], 0)
                  if len(specs) else np.zeros(len(seeds), np.int64))
        if len(uv):  # 半径=规格表直径/2（名义圆——装配判据口径）；列4=spec_id
            r_tab = np.asarray(self.table, np.float64)
            circles = np.column_stack([uv, r_tab[specs] / 2.0,
                                       specs.astype(np.float64)])
        else:
            circles = np.zeros((0, 4), np.float32)
        return {"seeds": seeds, "nb": nb, "nb2": nb2, "extra": extra,
                "pos": pos.astype(np.float32),
                "pos_obj": pos_obj.astype(np.float32),
                "valid": np.ones(len(seeds), np.float32),
                "vote_t": vote, "gt_idx": gt_idx,
                "spec_t": spec_t.astype(np.int64),
                "gt_circles": circles.astype(np.float32),
                "pts": p, "gt_uv": uv,
                "pts_uv": pts_uv,
                "radial": radial.astype(np.float32),
                "top_z": (float(self.st_topz[j]) if kind == "stirrup"
                          else 0.0),
                "expo": (float(self.z["rb_expo"][j]) if kind == "rebar"
                         else 0.0)}

    def __getitem__(self, i):
        j = self.idx[i % len(self.idx)]
        rng = self.rng if self.train else np.random.default_rng(i)
        return self._sample(j, rng)


def collate(batch):
    """种子数不齐 → 0 填充 + valid 掩码；变长 GT 圈 → 0 填充（无种子
    匹配自动跳过）。pts/gt_uv 等原始件按 list 透传（评估/量测用）。"""
    smax = max(len(b["seeds"]) for b in batch)
    k = batch[0]["nb"].shape[1] if len(batch[0]["nb"]) else 0
    k2 = batch[0]["nb2"].shape[1] if len(batch[0]["nb2"]) else 0

    def pad3(a, s):
        out = np.zeros((s,) + a.shape[1:], a.dtype)
        if len(a):
            out[:len(a)] = a
        return out

    def padk(a, s, fill=0):
        out = np.full((s,) + a.shape[1:], fill, a.dtype)
        if len(a):
            out[:len(a)] = a
        return out

    kmax = max(len(b["gt_circles"]) for b in batch)
    nmax = max(len(b["pts_uv"]) for b in batch)
    out = {key: [b[key] for b in batch] for key in ("pts", "gt_uv", "top_z",
                                                    "expo", "pts_uv")}
    out.update({
        "pts_uv_t": torch.from_numpy(np.stack(
            [pad3(b["pts_uv"], nmax) for b in batch])),
        "n_pts": torch.from_numpy(np.asarray(
            [len(b["pts_uv"]) for b in batch], np.int64)),
        "radial": torch.from_numpy(np.stack(
            [pad3(b["radial"], kmax) for b in batch])),
        "seeds": torch.from_numpy(np.stack(
            [pad3(b["seeds"], smax) for b in batch])),
        "nb": torch.from_numpy(np.stack(
            [pad3(b["nb"], smax) for b in batch])) if k else torch.zeros(
            len(batch), smax, 0, 3),
        "nb2": torch.from_numpy(np.stack(
            [pad3(b["nb2"], smax) for b in batch])) if k2 else torch.zeros(
            len(batch), smax, 0, 3),
        "extra": torch.from_numpy(np.stack(
            [pad3(b["extra"], smax) for b in batch])),
        "pos": torch.from_numpy(np.stack(
            [padk(b["pos"], smax) for b in batch])).bool(),
        "pos_obj": torch.from_numpy(np.stack(
            [padk(b["pos_obj"], smax) for b in batch])).bool(),
        "valid": torch.from_numpy(np.stack(
            [padk(b["valid"], smax) for b in batch])).bool(),
        "vote_t": torch.from_numpy(np.stack(
            [padk(b["vote_t"], smax) for b in batch])),
        "gt_idx": torch.from_numpy(np.stack(
            [padk(b["gt_idx"].astype(np.int64), smax, fill=-2)
             for b in batch])),
        "spec_t": torch.from_numpy(np.stack(
            [padk(b["spec_t"], smax) for b in batch])),
        "gt_circles": torch.from_numpy(np.stack(
            [padk(b["gt_circles"], kmax) for b in batch])),
    })
    # 损失需要的 tgt 键（seeds_uv 在设备侧由 seeds 派生）
    out["seeds_uv"] = out["seeds"][..., 1:3].clone()
    return out


# ── 评估（口径对齐 train_e2e_bev.eval_ends / eval_stirrup） ──────────────
@torch.no_grad()
def eval_ends_pt(model, z, kind, device, thr=None):
    """thr=None 时按头工作点：钢筋 0.5、套筒 SLEEVE_THR（P-048 诊断）。"""
    if thr is None:
        thr = SLEEVE_THR if kind == "sleeve" else 0.5
    table = REBAR_DIAM_TABLE if kind == "rebar" else SLEEVE_IDIAM_TABLE
    ds = PtDataset(z, kind, list(range(len(z["rb_n"]))), train=False)
    n_ok, det_n, pos_err, spec_hit, expo_err, ious = [], [], [], [], [], []
    gt_total = det_total = matched = 0
    model.eval()
    for i in range(len(ds)):
        s = ds[i]
        dets = detect(model, s["seeds"], s["nb"], s["extra"], s["nb2"],
                      table, thr=thr, device=device, pts_uv=s["pts_uv"])
        gt_uv = s["gt_uv"]
        n_ok.append(int(len(dets) == len(gt_uv)))
        det_n.append(len(dets))
        gt_total += len(gt_uv)
        det_total += len(dets)
        si, gi = match_dets(dets[:, :2], gt_uv, gate=POS_GATE)
        matched += len(si)
        for a, b in zip(si, gi):
            pos_err.append(float(np.linalg.norm(dets[a, :2] - gt_uv[b])))
            dia_hat = table[int(dets[a, 3:].argmax())]
            gt_diam = (float(z["rb_diam"][i % len(z["rb_n"])])
                       if kind == "rebar"
                       else float(z["sl_gt"][i % len(z["rb_n"])][b, 2]))
            spec_hit.append(int(abs(dia_hat - gt_diam) < 0.1))
            ious.append(circle_iou_np(dets[a, :2], dia_hat / 2.0,
                                      gt_uv[b], gt_diam / 2.0))
            if kind == "rebar":
                expo_hat = measure_exposed(s["pts"], dets[a, :2])
                expo_err.append(abs(expo_hat - s["expo"]))
    model.train()
    med = lambda x: round(float(np.median(x)), 2) if x else None
    return {"count_acc": round(sum(n_ok) / len(ds), 3),
            "n_bias": round(float(np.mean(det_n) - gt_total / len(ds)), 2),
            "recall": round(matched / max(gt_total, 1), 3),
            "precision": round(matched / max(det_total, 1), 3),
            "pos_p50": med(pos_err),
            "spec_hit": round(float(np.mean(spec_hit)), 3) if spec_hit else None,
            "iou_p50": med(ious),
            **({"expo_p50": med(expo_err)} if kind == "rebar" else {})}


@torch.no_grad()
def eval_neg_pt(model, z, kind, device, thr=None):
    if thr is None:
        thr = SLEEVE_THR if kind == "sleeve" else 0.5
    key = "rbn_pts" if kind == "rebar" else "sln_pts"
    table = REBAR_DIAM_TABLE if kind == "rebar" else SLEEVE_IDIAM_TABLE
    cnts, clean = [], 0
    model.eval()
    for j in range(len(z[key])):
        p = prefilter_far(z[key][j].astype(np.float32), 1, 2, 0)
        seeds, nb, extra, nb2 = make_seeds(p, seed_vox=SEED_VOX_END)
        dets = detect(model, seeds, nb, extra, nb2, table, thr=thr,
                      device=device, pts_uv=p[:, 1:3])
        cnts.append(len(dets))
        clean += int(len(dets) == 0)
    model.train()
    return {"neg_det_mean": round(float(np.mean(cnts)), 2),
            "neg_clean": round(clean / len(cnts), 3)}


@torch.no_grad()
def eval_stirrup_pt(model, z, device, thr=0.5):
    st_gt_all, topz_all = z["st_gt"], z["st_topz"]
    idx = [j for j in range(len(topz_all)) if len(st_gt_all[j]) > 0]
    if not idx:
        return None
    ds = PtDataset(z, "stirrup", idx, train=False)
    n_ok, pos_err, h_err, dia_hit, ious = [], [], [], [], []
    model.eval()
    for i in range(len(ds)):
        j = idx[i % len(idx)]
        s = ds[i]
        gt = z["st_gt"][j].astype(np.float32)
        legs = gt[:, :4].reshape(-1, 2)
        dets = detect(model, s["seeds"], s["nb"], s["extra"], s["nb2"],
                      STIRRUP_DIAM_TABLE, thr=thr, device=device,
                      pts_uv=s["pts_uv"])
        order = np.argsort(dets[:, 0]) if len(dets) else []
        pairs, used = [], np.zeros(len(dets), bool)
        for a in range(len(order)):
            if used[order[a]]:
                continue
            for b in range(a + 1, len(order)):
                k2 = order[b]
                if not used[k2] and abs(dets[k2, 0] - dets[order[a], 0]) < 30.0:
                    pairs.append((order[a], k2))
                    used[order[a]] = used[k2] = True
                    break
        n_ok.append(int(len(pairs) == len(gt)))
        si, gi = match_dets(dets[:, :2], legs, gate=100.0)
        for a, b in zip(si, gi):
            pos_err.append(float(np.linalg.norm(dets[a, :2] - legs[b])))
            m = b // 2
            dia_hat = STIRRUP_DIAM_TABLE[int(dets[a, 3:].argmax())]
            bar_r = dia_hat / 2
            ious.append(circle_iou_np(dets[a, :2], bar_r,
                                      legs[b], float(gt[m, 5]) / 2.0))
            h_hat = measure_height(s["pts"], dets[a, :2], s["top_z"], bar_r)
            h_err.append(abs(h_hat - float(gt[m, 4])))
            dia_hit.append(int(abs(dia_hat - gt[m, 5]) < 0.1))
    model.train()
    med = lambda x: round(float(np.median(x)), 2) if x else None
    return {"count_acc": round(n_ok.count(1) / len(ds), 3),
            "leg_pos_p50": med(pos_err), "height_p50": med(h_err),
            "iou_p50": med(ious),
            "dia_hit": round(float(np.mean(dia_hit)), 3) if dia_hit else None}


# ── 评估：真实场景（oracle-crop；视图构造逐行对齐 train_e2e_bev.eval_real，
#    仅柱化+decode 换为种子+detect） ─────────────────────────────────────
@torch.no_grad()
def eval_real_pt(model_rb, model_sl, device, thr=0.5, sleeve_thr=SLEEVE_THR):
    from lib.training.train_bev import _fit_rigid_2d
    import open3d as o3d
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from sklearn.cluster import DBSCAN
    model_rb.eval()
    model_sl.eval()
    pts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/scene_full_mm - Cloud.ply")).points).astype(np.float32)
    gts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/rebar_only.ply")).points).astype(np.float32)
    c = pts.mean(0)
    _, _, V = np.linalg.svd(pts[::20] - c, full_matrices=False)
    axis = V[0]
    t_all = (pts - c) @ axis
    t_gt = (gts - c) @ axis
    t0 = float(t_gt.min())
    band = (t_gt - t0 > -30) & (t_gt - t0 < 80)
    cy, cz = np.median(gts[band, 1] - c[1]), np.median(gts[band, 2] - c[2])
    win = (t_all > t0 - 300) & (t_all < t0 + 1750)
    p = np.column_stack([t_all[win] - t0, pts[win, 1] - c[1] - cy,
                         pts[win, 2] - c[2] - cz]).astype(np.float32)
    rng_cap = np.random.default_rng(0)
    if len(p) > REBAR_CAP:
        p = p[rng_cap.choice(len(p), REBAR_CAP, replace=False)]
    p = prefilter_far(p, 1, 2, 0)
    seeds, nb, extra, nb2 = make_seeds(p, seed_vox=SEED_VOX_END)
    dets = detect(model_rb, seeds, nb, extra, nb2, REBAR_DIAM_TABLE, thr=thr,
                  device=device)
    with open(anchor_project_path("real_resource/rebar_gt_manual.json"),
              encoding="utf-8") as f:
        circ = np.array([[q["h2"], q["h3"]]
                         for q in json.load(f)["rebars"]], dtype=np.float64)
    cl = DBSCAN(eps=60.0, min_samples=5).fit(gts - gts.mean(0))
    labs = [k for k in np.unique(cl.labels_) if k >= 0]
    sizes = sorted(((int((cl.labels_ == k).sum()), k) for k in labs),
                   reverse=True)
    if len(sizes) > len(circ):
        print(f"[real-gt] DBSCAN {len(sizes)} 簇（含碎片），按大小取前 "
              f"{len(circ)}；簇点数 {[_s for _s, _ in sizes]}", flush=True)
    gt_uv, gt_expo = [], []
    for _, k in sizes[:len(circ)]:
        gk = gts[cl.labels_ == k]
        tk = (gk - c) @ axis
        bk = (tk - t0 > -30) & (tk - t0 < 80)
        gt_uv.append([(gk[bk, 1] - c[1]).mean() - cy,
                      (gk[bk, 2] - c[2]).mean() - cz])
        gt_expo.append(float(tk.max() - t0))
    gt_uv = np.array(gt_uv, dtype=np.float32)
    Rm, tm = np.eye(2), np.zeros(2)
    for _ in range(2):
        ri, ci = linear_sum_assignment(cdist(circ @ Rm.T + tm, gt_uv))
        Rm, tm = _fit_rigid_2d(circ[ri], gt_uv[ci])
    fit_res = float(np.linalg.norm(
        (circ @ Rm.T + tm)[ri] - gt_uv[ci], axis=1).mean())
    if fit_res < 30.0:
        gt_uv = (circ @ Rm.T + tm).astype(np.float32)
        gt_n = len(circ)
    else:
        gt_n = len(gt_uv)
    pos_err, expo_err = [], []
    si, gi = match_dets(dets[:, :2], gt_uv, gate=POS_GATE)
    for a, b in zip(si, gi):
        pos_err.append(float(np.linalg.norm(dets[a, :2] - gt_uv[b])))
        expo_err.append(abs(measure_exposed(p, dets[a, :2]) - gt_expo[b]))
    # 套筒端
    t1 = float(t_all.max())
    wins = (t_all > t1 - 300) & (t_all < t1 + 80)
    band2 = np.abs(t_all[wins] - t1) < 50
    scy, scz = (np.median(pts[wins][band2, 1] - c[1]),
                np.median(pts[wins][band2, 2] - c[2]))
    ps = np.column_stack([t_all[wins] - t1, pts[wins, 1] - c[1] - scy,
                          pts[wins, 2] - c[2] - scz]).astype(np.float32)
    if len(ps) > SLEEVE_CAP:
        ps = ps[rng_cap.choice(len(ps), SLEEVE_CAP, replace=False)]
    seeds_s, nb_s, extra_s, nb2_s = make_seeds(ps, seed_vox=SEED_VOX_END)
    dets_s = detect(model_sl, seeds_s, nb_s, extra_s, nb2_s,
                    SLEEVE_IDIAM_TABLE, thr=sleeve_thr, device=device)
    with open(anchor_project_path("real_resource/sleeve_gt_manual.json"),
              encoding="utf-8") as f:
        scirc = np.array([[q["h2"], q["h3"]]
                          for q in json.load(f)["circles"]], dtype=np.float64)
    s_uv = scirc @ Rm.T + tm - np.array([cy - scy, cz - scz])
    si_s, gi_s = match_dets(dets_s[:, :2], s_uv.astype(np.float32),
                            gate=POS_GATE)
    spos = [float(np.linalg.norm(dets_s[a, :2] - s_uv[b]))
            for a, b in zip(si_s, gi_s)]
    model_rb.train()
    model_sl.train()
    med = lambda x: round(float(np.median(x)), 2) if x else None
    return {"钢筋GT根数": gt_n, "钢筋检出": len(dets), "钢筋匹配": len(si),
            "钢筋标注拟合残差mm": round(fit_res, 1),
            "钢筋位置p50mm": med(pos_err),
            "钢筋外露量测误差p50mm": med(expo_err),
            "套筒GT环数": len(s_uv), "套筒检出": len(dets_s),
            "套筒匹配": len(si_s), "套筒位置p50mm": med(spos)}


def _pct(x):
    return f"{x:.0%}" if x is not None else "—"


# ── 模块基准台：部署视图臂（阶段 1a，2026-10-28） ────────────────────────
@torch.no_grad()
def eval_faceview_pt(model_rb, model_sl, device, n_files=0):
    """部署视图基准：GT 面(oracle_faces)+GT 材质带点 → face_view（与
    e2e_fullchain 逐行同构，含 u 轴手性对齐）→ make_seeds → detect。

    与 eval_ends_pt（缓存视图臂）同一权重同一 test 件对照，分解
    "缓存视图 vs 部署视图"的域差（用户模块路线：小模块先拉指标）。
    指标口径与 eval_ends_pt 一致（count/recall/precision/pos/spec/expo）。
    """
    import glob
    from lib.e2e.face_sem import face_band_labels
    from lib.e2e.bev_dataset import (REBAR_WIN, SLEEVE_WIN, REBAR_CAP,
                                     SLEEVE_CAP)
    from lib.pipeline.single_chain import face_view
    files = sorted(glob.glob(anchor_project_path("mock/inuse/test/*.npz")))
    if n_files:
        files = files[:n_files]
    rng = np.random.default_rng(0)

    def run_arm(model, kind, thr):
        pos_err, spec_hit, expo_err = [], [], []
        det_total = gt_total = matched = n_ok = 0
        model.eval()
        for f in files:
            d = np.load(f, allow_pickle=False)
            pts = d["points"].astype(np.float32)
            lbls = d["labels"]
            meta = json.loads(str(d["meta_json"]))
            c0 = pts.mean(0)
            p = pts - c0
            from lib.pipeline.single_chain import oracle_faces
            faces = oracle_faces(p, meta, c0)
            lab = face_band_labels(pts, lbls, meta)
            face = faces["rebar" if kind == "rebar" else "sleeve"]
            pq = p[lab == (1 if kind == "rebar" else 2)]
            if len(pq) < 50:
                continue
            if kind == "rebar":
                pv, basis = face_view(pq, face, REBAR_WIN, REBAR_CAP, rng)
            else:
                pv, basis = face_view(pq, face, SLEEVE_WIN, SLEEVE_CAP,
                                      rng, rebar_far=False)
            e1, e2 = basis
            c3 = face["c3"]
            seeds, nb, extra, nb2 = make_seeds(pv, seed_vox=SEED_VOX_END,
                                               rng=rng)
            table = REBAR_DIAM_TABLE if kind == "rebar" \
                else SLEEVE_IDIAM_TABLE
            dets = detect(model, seeds, nb, extra, nb2, table, thr=thr,
                          device=device)
            if kind == "rebar":
                gt_yz = (np.asarray(meta["rebar_positions_yz"], np.float32)
                         - c0[1:])          # 世界系 → 质心系（套筒臂同源）
                gt_uv = np.column_stack([(gt_yz - c3[1:]) @ e1[1:],
                                         (gt_yz - c3[1:]) @ e2[1:]])
                gt_diam = float(meta["rebar_diam"])
                g0 = meta["rebar_gt"][0]
                end_x = float(meta["rebar_end_x"])
                flip = -1.0 if float(meta["sleeve_end_x"]) > end_x else 1.0
                gt_expo = flip * (float(g0["p1"][0]) - end_x)
            else:
                gts = meta.get("sleeve_gt") or []
                if not gts:
                    continue
                gt_w = np.asarray([[g["center_yz"][0] - c0[1],
                                    g["center_yz"][1] - c0[2]]
                                   for g in gts], np.float32)
                gt_uv = np.column_stack([(gt_w - c3[1:]) @ e1[1:],
                                         (gt_w - c3[1:]) @ e2[1:]])
                gt_diam = None
                gt_expo = None
            n_ok += int(len(dets) == len(gt_uv))
            det_total += len(dets)
            gt_total += len(gt_uv)
            si, gi = match_dets(dets[:, :2], gt_uv, gate=POS_GATE)
            matched += len(si)
            for a, b in zip(si, gi):
                pos_err.append(float(np.linalg.norm(
                    dets[a, :2] - gt_uv[b])))
                dia_hat = table[int(dets[a, 3:].argmax())]
                if kind == "rebar":
                    spec_hit.append(int(abs(dia_hat - gt_diam) < 0.1))
                    expo_err.append(abs(measure_exposed(pv, dets[a, :2])
                                        - gt_expo))
                else:
                    gt_d = float(gts[b]["inner_diam"])
                    spec_hit.append(int(abs(dia_hat - gt_d) < 0.1))
        model.train()
        med = lambda x: round(float(np.median(x)), 2) if x else None
        return {"count_acc": round(n_ok / max(len(files), 1), 3),
                "recall": round(matched / max(gt_total, 1), 3),
                "precision": round(matched / max(det_total, 1), 3),
                "pos_p50": med(pos_err),
                "spec_hit": round(float(np.mean(spec_hit)), 3)
                if spec_hit else None,
                **({"expo_p50": med(expo_err)} if kind == "rebar" else {})}

    return {"rebar": run_arm(model_rb, "rebar", 0.5),
            "sleeve": run_arm(model_sl, "sleeve", SLEEVE_THR)}


def fmt(tag, s, kind):
    line = (f"    [{tag}] 根数全对 {_pct(s['count_acc'])} | "
            f"根数平均偏差 {s['n_bias']:+.2f} 根 | 检出召回 {_pct(s['recall'])} | "
            f"误检率 {_pct(1 - s['precision'])} | 位置p50 {s['pos_p50']}mm | "
            f"规格命中 {_pct(s['spec_hit'])} | 圆IoU p50 {s.get('iou_p50')}")
    if kind == "rebar":
        line += f" | 外露量测误差p50 {s.get('expo_p50')}mm"
    return line


# ── 主流程 ──────────────────────────────────────────────────────────────
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-ends", type=int, default=4)
    ap.add_argument("--batch-st", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-vote", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tag", type=str, default="head_point")
    ap.add_argument("--v1", action="store_true",
                    help="复现 v1 架构（无 BN/粗尺度/精修/fg），基线复训用")
    ap.add_argument("--ref-radial", type=int, default=0,
                    help="精修头径向剖面维数（v3=18；0=v2b 兼容）")
    ap.add_argument("--eval-only", type=str, default="",
                    help="只评估不训练：权重路径（out/exp1_train/<tag>/best.pth）。"
                         "跑缓存视图臂+部署视图臂（--bench-faceview）")
    ap.add_argument("--bench-faceview", action="store_true",
                    help="评估时附部署视图基准臂（GT 面+GT 带点，与全链"
                         " face_view 同构）")
    args = ap.parse_args(argv)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    if args.eval_only:
        pt = torch.load(anchor_project_path(args.eval_only),
                        map_location=device)
        flags = dict(bn=not args.v1, coarse=not args.v1, refine=not args.v1,
                     fg=not args.v1)
        flags["in_extra"] = 6 if args.v1 else 7
        flags["ref_radial"] = (args.ref_radial
                               or int(pt.get("ref_radial", 0)))
        m_rb = PointCircleNet(len(REBAR_DIAM_TABLE), **flags).to(device)
        m_sl = PointCircleNet(len(SLEEVE_IDIAM_TABLE), **flags).to(device)
        m_st = PointCircleNet(len(STIRRUP_DIAM_TABLE), **flags).to(device)
        m_rb.load_state_dict(pt["rb"])
        m_sl.load_state_dict(pt["sl"])
        m_st.load_state_dict(pt["st"])
        zte = load_bev_cache(anchor_project_path(CACHE), "test")
        s_rb = eval_ends_pt(m_rb, zte, "rebar", device)
        s_sl = eval_ends_pt(m_sl, zte, "sleeve", device)
        s_st = eval_stirrup_pt(m_st, zte, device)
        print("[缓存视图臂] 钢筋 " + fmt("", s_rb, "rebar"))
        print("[缓存视图臂] 套筒 " + fmt("", s_sl, "sleeve"))
        if s_st:
            print(f"[缓存视图臂] 箍筋 榀数全对 {s_st['count_acc']:.0%} | "
                  f"腿位p50 {s_st['leg_pos_p50']}mm | 高度p50 "
                  f"{s_st['height_p50']}mm | 直径命中 {s_st['dia_hit']:.0%}")
        if args.bench_faceview:
            fv = eval_faceview_pt(m_rb, m_sl, device)
            for kind in ("rebar", "sleeve"):
                s = fv[kind]
                line = (f"[部署视图臂] {kind} 根数全对 {_pct(s['count_acc'])}"
                        f" | 检出召回 {_pct(s['recall'])} | 误检率 "
                        f"{_pct(1 - s['precision'])} | 位置p50 {s['pos_p50']}mm"
                        f" | 规格命中 {_pct(s['spec_hit'])}")
                if kind == "rebar":
                    line += f" | 外露量测误差p50 {s.get('expo_p50')}mm"
                print(line)
        return
    ztr = load_bev_cache(anchor_project_path(CACHE), "train")
    zte = load_bev_cache(anchor_project_path(CACHE), "test")
    zva = load_bev_cache(anchor_project_path(CACHE), "val")
    all_idx = list(range(len(ztr["rb_n"])))
    st_idx = [j for j in all_idx if len(ztr["st_gt"][j]) > 0]
    print(f"train 件数 {len(all_idx)}（含箍筋 {len(st_idx)}）", flush=True)

    ds_rb = PtDataset(ztr, "rebar", all_idx, True)
    ds_sl = PtDataset(ztr, "sleeve", all_idx, True)
    ds_st = PtDataset(ztr, "stirrup", st_idx, True)

    def spec_of(kind, j):
        if kind == "rebar":
            return _spec_id(REBAR_DIAM_TABLE, float(ztr["rb_diam"][j]))
        if kind == "sleeve":
            g = ztr["sl_gt"][j]
            return (_spec_id(SLEEVE_IDIAM_TABLE, float(g[0, 2]))
                    if len(g) else -1)
        g = ztr["st_gt"][j]
        return (_spec_id(STIRRUP_DIAM_TABLE, float(g[0, 5]))
                if len(g) else -1)

    dl_rb = DataLoader(ds_rb, args.batch_ends, num_workers=0,
                       collate_fn=collate,
                       sampler=SpecBalancedSampler(
                           len(all_idx), lambda j: spec_of("rebar", j),
                           rep=ds_rb.repeat))
    dl_sl = DataLoader(ds_sl, args.batch_ends, num_workers=0,
                       collate_fn=collate,
                       sampler=SpecBalancedSampler(
                           len(all_idx), lambda j: spec_of("sleeve", j),
                           rep=ds_sl.repeat))
    dl_st = DataLoader(ds_st, args.batch_st, num_workers=0,
                       collate_fn=collate,
                       sampler=SpecBalancedSampler(
                           len(st_idx),
                           lambda j: spec_of("stirrup", st_idx[j]),
                           rep=ds_st.repeat))
    flags = dict(bn=not args.v1, coarse=not args.v1, refine=not args.v1,
                 fg=not args.v1)
    flags["in_extra"] = 6 if args.v1 else 7   # v1 权重 = 6 特征口径
    flags["ref_radial"] = args.ref_radial
    model_rb = PointCircleNet(len(REBAR_DIAM_TABLE), **flags).to(device)
    model_sl = PointCircleNet(len(SLEEVE_IDIAM_TABLE), **flags).to(device)
    model_st = PointCircleNet(len(STIRRUP_DIAM_TABLE), **flags).to(device)
    opt = torch.optim.AdamW(sum([list(m.parameters()) for m in
                                 (model_rb, model_sl, model_st)], []),
                            lr=args.lr)

    def lr_fn(ep):  # 0-based：warmup 线性升 → cosine 降到 0.1×base（1e-4）
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    ckpt_dir = anchor_project_path(f"out/exp1_train/{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    best_score = -1.0

    tables = {"rb": REBAR_DIAM_TABLE, "sl": SLEEVE_IDIAM_TABLE,
              "st": STIRRUP_DIAM_TABLE}
    n_steps = max(len(dl_rb), len(dl_sl), len(dl_st))
    for ep in range(1, args.epochs + 1):
        t0, tot, n_upd = time.time(), 0.0, 0
        acc = {}
        nan_skips = {}
        it_rb, it_sl, it_st = iter(dl_rb), iter(dl_sl), cycle(dl_st)
        model_rb.train()
        for _ in range(n_steps):
            opt.zero_grad()
            for it, model, key in ((it_rb, model_rb, "rb"),
                                   (it_sl, model_sl, "sl"),
                                   (it_st, model_st, "st")):
                try:
                    b = next(it)
                except StopIteration:
                    continue
                nb = b["nb"].to(device)
                extra = b["extra"].to(device)
                out = model(nb, extra, b["nb2"].to(device),
                            b["valid"].to(device))
                tgt = {k2: b[k2].to(device) for k2 in
                       ("pos", "pos_obj", "valid", "vote_t", "gt_idx",
                        "spec_t", "gt_circles", "seeds_uv", "radial",
                        "pts_uv_t", "n_pts")}
                tgt["pts_uv"] = tgt.pop("pts_uv_t")
                loss, parts = point_loss(out, tgt, tables[key], model=model,
                                         w_vote=args.w_vote, v1=args.v1)
                if not torch.isfinite(loss):
                    # 非有限损失守卫：跳过该步不进权重（P-046 家族教训：
                    # NaN 一旦入权重整轮作废），计数留痕供定位
                    opt.zero_grad()
                    nan_skips[key] = nan_skips.get(key, 0) + 1
                    continue
                loss.backward()
                tot += float(loss)
                n_upd += 1
                for kk, vv in parts.items():
                    if kk != "n_gt":
                        acc[kk] = acc.get(kk, 0.0) + vv
            torch.nn.utils.clip_grad_norm_(
                sum([list(m.parameters()) for m in
                     (model_rb, model_sl, model_st)], []), 10.0)
            opt.step()
        sched.step()
        parts_msg = " ".join(f"{k} {v / max(n_upd, 1):.3f}"
                             for k, v in sorted(acc.items()))
        msg = (f"[ep {ep:3d}/{args.epochs}] loss {tot / n_steps:.4f} "
               f"({time.time() - t0:.0f}s) lr {sched.get_last_lr()[0]:.2e} "
               f"| {parts_msg}"
               + (f" | nan跳步 {nan_skips}" if nan_skips else ""))
        if ep % N_EVAL == 0 or ep == args.epochs:
            s_rb = eval_ends_pt(model_rb, zte, "rebar", device)
            s_rb.update(eval_neg_pt(model_rb, zte, "rebar", device))
            s_sl = eval_ends_pt(model_sl, zte, "sleeve", device)
            s_sl.update(eval_neg_pt(model_sl, zte, "sleeve", device))
            s_st = eval_stirrup_pt(model_st, zte, device)
            r = eval_real_pt(model_rb, model_sl, device)
            # 训练集同轨（拟合/泛化分解仪表，用户 2026-10-24 提问落码）
            tr_rb = eval_ends_pt(model_rb, ztr, "rebar", device)
            v_rb = eval_ends_pt(model_rb, zva, "rebar", device)
            v_sl = eval_ends_pt(model_sl, zva, "sleeve", device)
            v_st = eval_stirrup_pt(model_st, zva, device)
            score = (v_rb["recall"] + v_rb["count_acc"]
                     + v_sl["count_acc"] + (v_st["count_acc"]
                                            if v_st else 0.0))
            msg += ("\n" + fmt("合成-钢筋", s_rb, "rebar")
                    + f" | 负视图 {s_rb['neg_det_mean']}"
                      f"({s_rb['neg_clean']:.0%}全零)"
                    + "\n" + fmt("合成-套筒", s_sl, "sleeve")
                    + f" | 负视图 {s_sl['neg_det_mean']}"
                      f"({s_sl['neg_clean']:.0%}全零)"
                    + "\n    [合成-箍筋] 榀数全对 "
                    + (f"{s_st['count_acc']:.0%} | 腿位p50 {s_st['leg_pos_p50']}mm"
                       f" | 高度p50 {s_st['height_p50']}mm"
                       f" | 圆IoU p50 {s_st.get('iou_p50')}"
                       f" | 直径命中 {s_st['dia_hit']:.0%}" if s_st else "无样本")
                    + "\n    [训练集-钢筋] " + fmt("", tr_rb, "rebar")[22:]
                    + f"\n    [真实场景] 钢筋 {r['钢筋检出']}/{r['钢筋GT根数']}"
                      f"(匹配{r['钢筋匹配']}) 位置p50 {r['钢筋位置p50mm']}mm | "
                      f"套筒 {r['套筒检出']}/{r['套筒GT环数']}"
                      f"(匹配{r['套筒匹配']}) 位置p50 {r['套筒位置p50mm']}mm"
                    + f"\n    [val] score {score:.3f}"
                      f"（钢筋召回 {v_rb['recall']} 全对 {v_rb['count_acc']}"
                      f" | 套筒全对 {v_sl['count_acc']}）")
            history.append({"epoch": ep, "rebar": s_rb, "sleeve": s_sl,
                            "stirrup": s_st, "real": r,
                            "train_rebar": tr_rb,
                            "val_score": round(score, 4),
                            "loss_parts": {k: round(v / max(n_upd, 1), 4)
                                           for k, v in acc.items()}})
            if score > best_score:
                best_score = score
                torch.save({"rb": model_rb.state_dict(),
                            "sl": model_sl.state_dict(),
                            "st": model_st.state_dict(), "epoch": ep,
                            "val_score": score,
                            "ref_radial": args.ref_radial},
                           os.path.join(ckpt_dir, "best.pth"))
                msg += " ← best"
        print(msg, flush=True)
        torch.save({"rb": model_rb.state_dict(),
                    "sl": model_sl.state_dict(),
                    "st": model_st.state_dict(), "epoch": ep,
                    "ref_radial": args.ref_radial},
                   os.path.join(ckpt_dir, "last.pth"))
    with open(os.path.join(ckpt_dir, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    print(f"完成 → {ckpt_dir}")


if __name__ == "__main__":
    main()
