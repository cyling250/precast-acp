# -*- coding: utf-8 -*-
"""
BEV 端到端装配参数检测——v6 训练入口（PointPillars + CenterNet）。

v6（2026-10-24 与用户定稿，见 docs/handover.md T-033）：面视图检测阶段 B。
训练视图仍按 GT 面投影（缓存口径同 bev_v1/v2，保证与其指标可比），新增：
  1) 双端交叉负样本——钢筋头看套筒端视图、套筒头看出筋端视图（热图全 0），
     端别由检测头自决（部署时两头都跑两端面，正确端才点火）；
  2) face-jitter——视图 t 原点随机 ±25mm（部署面参数误差量级：经典面
     偏移 p50 7-13mm），只扰动深度通道、不改绝对尺度。
评估三轨：合成 test（oracle 面，对照 bev_v1/v2）+ 交叉负视图误检（端别
自决能力）+ 真实场景 oracle-crop。

用法（cwd=src）：python -m experiments.exp1_train bev [--epochs 30]
（权重落盘 out/exp1_train/head_bev；训练缓存 out/exp1_train/cache/bev）
"""
import json
import os
import time
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import set_seed, anchor_project_path
from lib.e2e.crop_dataset import load_split
from lib.e2e.bev_dataset import (CELL, END_ORIGIN, END_WH, ST_ORIGIN, ST_WH,
                                 build_split_cache, load_bev_cache, pillarize,
                                 encode_gt, measure_exposed, measure_height,
                                 prefilter_far, rot_flip_stirrup,
                                 _rot_flip_points, _drop_jitter,
                                 REBAR_CAP, SLEEVE_CAP)
from lib.e2e.model_pillar import (PillarBEVNet, bev_loss, decode, match_dets,
                                  STIRRUP_DIAM_TABLE, REBAR_DIAM_TABLE,
                                  SLEEVE_IDIAM_TABLE)

CACHE = "out/exp1_train/cache/bev"
POS_GATE = 120.0
N_EVAL = 5
FACE_JITTER = 25.0  # 部署面偏移误差量级（经典面 p50 7-13mm）
NEG_P = 0.25  # 交叉负视图概率（0.5 曾致钢筋头正类曝光减半而漏检塌缩，
#             召回 29%/漏检 5.8 根；套筒头不受影响——其视图窗小、峰密集）


# ── 数据 ────────────────────────────────────────────────────────────────
def preprocess():
    for split in ("train", "val", "test"):
        out = anchor_project_path(os.path.join(CACHE, f"{split}.npz"))
        if os.path.exists(out):
            continue
        paths = load_split(anchor_project_path("mock/inuse"), split)
        build_split_cache(paths, out)
        print(f"[cache] {split}: {len(paths)} 件 → {out}", flush=True)


class BevDataset(Dataset):
    """kind ∈ rebar/sleeve/stirrup；train 每 epoch 重复采样（免费扩增）。

    rebar/sleeve 训练时以 0.5 概率取交叉负视图（rbn/sln，热图全 0）——
    v6 端别自决。NpzFile 惰性解压：所需数组必须 __init__ 一次性物化
    （P-033：逐样本访问曾致 epoch 慢 46 倍）。
    """

    _KEYS = {"rebar": ("rb_pts", "rb_uv", "rb_diam", "rbn_pts"),
             "sleeve": ("sl_pts", "sl_gt", "sln_pts"),
             "stirrup": ("st_pts", "st_gt", "st_topz")}

    def __init__(self, z, kind, idx, train, repeat=None):
        self.z, self.kind, self.idx, self.train = z, kind, idx, train
        for k in self._KEYS[kind]:
            setattr(self, k, z[k])
        self.repeat = repeat or (2 if train else 1)
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.idx) * self.repeat

    def _sample(self, j, rng):
        if self.kind == "rebar":
            neg = self.train and rng.random() < NEG_P
            p = (self.rbn_pts[j] if neg
                 else self.rb_pts[j]).astype(np.float32)
            uv = self.rb_uv[j].astype(np.float32)
            if not neg:
                spec = np.full(len(uv),
                               int(np.argmin(np.abs(
                                   np.array(REBAR_DIAM_TABLE)
                                   - float(self.rb_diam[j])))),
                               dtype=np.int64)
            else:
                uv = np.zeros((0, 2), np.float32)
                spec = None
            p = prefilter_far(p, 1, 2, 0)
            if self.train:
                p, uv = _rot_flip_points(p, 1, 2, rng, gt=uv, gt_uv=0)
                p = _drop_jitter(p, rng)
                p[:, 0] += rng.uniform(-FACE_JITTER, FACE_JITTER)
            feat = pillarize(p, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
            heat, off, cls = encode_gt(uv, END_ORIGIN, END_WH, spec_ids=spec)
            return feat, heat, off, cls, (p, uv)
        if self.kind == "sleeve":
            neg = self.train and rng.random() < NEG_P
            if neg:
                p = self.sln_pts[j].astype(np.float32)
                gt = np.zeros((0, 3), np.float32)
            else:
                p = self.sl_pts[j].astype(np.float32)
                gt = self.sl_gt[j].astype(np.float32)  # (K,3): u,v,内径mm
            if self.train:
                p, gt = _rot_flip_points(p, 1, 2, rng, gt=gt, gt_uv=0)
                p = _drop_jitter(p, rng)
                p[:, 0] += rng.uniform(-FACE_JITTER, FACE_JITTER)
            feat = pillarize(p, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
            spec = (np.array([int(np.argmin(np.abs(
                np.array(SLEEVE_IDIAM_TABLE) - g[2]))) for g in gt],
                dtype=np.int64) if len(gt) else None)
            heat, off, cls = encode_gt(gt[:, :2] if len(gt)
                                       else np.zeros((0, 2)),
                                       END_ORIGIN, END_WH, spec_ids=spec)
            return feat, heat, off, cls, (p, gt[:, :2] if len(gt) else gt)
        p = self.st_pts[j].astype(np.float32)
        gt = self.st_gt[j].astype(np.float32)  # (M,6)
        top_z = float(self.st_topz[j])
        if self.train:
            p, gt = rot_flip_stirrup(p, gt, rng)
            p = _drop_jitter(p, rng)
        feat = pillarize(p, 0, 1, 2, ST_ORIGIN, ST_WH, d_shift=top_z)
        legs = gt[:, :4].reshape(-1, 2)
        spec = np.repeat([int(np.argmin(np.abs(
            np.array(STIRRUP_DIAM_TABLE) - g[5]))) for g in gt], 2)
        heat, off, cls = encode_gt(legs, ST_ORIGIN, ST_WH, spec_ids=spec)
        return feat, heat, off, cls, (p, legs, gt)

    def __getitem__(self, i):
        j = self.idx[i % len(self.idx)]
        rng = self.rng if self.train else np.random.default_rng(i)
        feat, heat, off, cls, extra = self._sample(j, rng)
        return (torch.from_numpy(feat), torch.from_numpy(heat),
                torch.from_numpy(off.transpose(2, 0, 1).copy()),
                torch.from_numpy(cls), extra)


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    heats = torch.stack([b[1] for b in batch])
    offs = torch.stack([b[2] for b in batch])
    clss = torch.stack([b[3] for b in batch])
    return feats, heats, offs, clss, [b[4] for b in batch]


# ── 评估：合成（oracle 面，对照 bev_v1/v2） ─────────────────────────────
@torch.no_grad()
def eval_ends(model, z, kind, device):
    """钢筋/套筒端：根数 acc、匹配位置 p50、规格命中、（钢筋）外露量测 p50。"""
    model.eval()
    table = REBAR_DIAM_TABLE if kind == "rebar" else SLEEVE_IDIAM_TABLE
    idx = list(range(len(z["rb_n"])))
    diam_arr = z["rb_diam"] if kind == "rebar" else None
    expo_arr = z["rb_expo"] if kind == "rebar" else None
    sl_gt_all = z["sl_gt"] if kind == "sleeve" else None
    ds = BevDataset(z, kind, idx, train=False)
    n_ok = []
    pos_err, spec_hit, expo_err = [], [], []
    det_n, gt_total, det_total, matched = [], 0, 0, 0
    for i in range(len(ds)):
        j = idx[i % len(idx)]
        feat, _, _, _, (p, gt_uv) = ds[i]
        dets = decode(model(feat[None].to(device)), CELL, END_ORIGIN, topk=32)[0]
        n_ok.append(int(len(dets) == len(gt_uv)))
        det_n.append(len(dets))
        gt_total += len(gt_uv)
        det_total += len(dets)
        gt_diam = (np.full(len(gt_uv), float(diam_arr[j])) if kind == "rebar"
                   else sl_gt_all[j].astype(np.float32)[:, 2])
        si, gi = match_dets(dets[:, :2], gt_uv, gate=POS_GATE)
        matched += len(si)
        for a, b in zip(si, gi):
            pos_err.append(float(np.linalg.norm(dets[a, :2] - gt_uv[b])))
            dia_hat = table[int(dets[a, 3:].argmax())]
            spec_hit.append(int(abs(dia_hat - gt_diam[b]) < 0.1))
            if kind == "rebar":
                expo_hat = measure_exposed(p, dets[a, :2])
                expo_err.append(abs(expo_hat - float(expo_arr[j])))
    model.train()
    med = lambda x: round(float(np.median(x)), 2) if x else None
    out = {"count_acc": round(sum(n_ok) / len(ds), 3),
           "n_bias": round(float(np.mean(det_n) - gt_total / len(ds)), 2),
           "recall": round(matched / max(gt_total, 1), 3),
           "precision": round(matched / max(det_total, 1), 3),
           "pos_p50": med(pos_err),
           "spec_hit": round(float(np.mean(spec_hit)), 3) if spec_hit else None}
    if kind == "rebar":
        out["expo_p50"] = med(expo_err)
    return out


@torch.no_grad()
def eval_neg_views(model, z, kind, device, thresh=0.3):
    """交叉负视图误检（端别自决能力）：对端视图检出数（0 为满分）。

    统计：件均检出数 / 全零件占比（thresh 与部署一致，另报 0.5 高阈）。
    """
    model.eval()
    key = "rbn_pts" if kind == "rebar" else "sln_pts"
    arr = z[key]
    cnts, clean = [], 0
    for j in range(len(arr)):
        p = arr[j].astype(np.float32)
        p = prefilter_far(p, 1, 2, 0)
        feat = pillarize(p, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
        dets = decode(model(torch.from_numpy(feat[None]).to(device)),
                      CELL, END_ORIGIN, thresh=thresh, topk=32)[0]
        cnts.append(len(dets))
        if len(dets) == 0:
            clean += 1
    model.train()
    return {"neg_det_mean": round(float(np.mean(cnts)), 2),
            "neg_clean": round(clean / len(cnts), 3)}


@torch.no_grad()
def eval_stirrup(model, z, device):
    """箍筋：配对数 acc、腿位置 p50、高度量测 p50、直径命中。"""
    model.eval()
    st_gt_all, topz_all = z["st_gt"], z["st_topz"]
    idx = [j for j in range(len(topz_all)) if len(st_gt_all[j]) > 0]
    if not idx:
        return None
    ds = BevDataset(z, "stirrup", idx, train=False)
    n_ok = []
    pos_err, h_err, dia_hit = [], [], []
    for i in range(len(ds)):
        j = idx[i % len(idx)]
        feat, _, _, _, (p, legs, gt) = ds[i]
        dets = decode(model(feat[None].to(device)), CELL, ST_ORIGIN, topk=96)[0]
        # 配对：同箍两腿 x 相同（|Δx|<30mm）
        xs = dets[:, 0] if len(dets) else np.zeros(0)
        order = np.argsort(xs)
        pairs, used = [], np.zeros(len(dets), bool)
        for a in range(len(order)):
            if used[order[a]]:
                continue
            for b in range(a + 1, len(order)):
                k = order[b]
                if not used[k] and abs(dets[k, 0] - dets[order[a], 0]) < 30.0:
                    pairs.append((order[a], k))
                    used[order[a]] = used[k] = True
                    break
        n_ok.append(int(len(pairs) == len(gt)))
        si, gi = match_dets(dets[:, :2], legs, gate=100.0)
        for a, b in zip(si, gi):
            pos_err.append(float(np.linalg.norm(dets[a, :2] - legs[b])))
            m = b // 2  # GT 腿→箍行
            bar_r = STIRRUP_DIAM_TABLE[int(dets[a, 3:].argmax())] / 2
            h_hat = measure_height(p, dets[a, :2], float(topz_all[j]), bar_r)
            h_err.append(abs(h_hat - float(gt[m, 4])))
            dia_hit.append(int(abs(STIRRUP_DIAM_TABLE[int(dets[a, 3:].argmax())]
                                   - gt[m, 5]) < 0.1))
    model.train()
    med = lambda x: round(float(np.median(x)), 2) if x else None
    return {"count_acc": round(n_ok.count(1) / len(ds), 3),
            "leg_pos_p50": med(pos_err), "height_p50": med(h_err),
            "dia_hit": round(float(np.mean(dia_hit)), 3) if dia_hit else None}


# ── 评估：真实（oracle-crop；GT=人工标注 12 根筋/12 套筒，2026-10-24 定）──
def _fit_rigid_2d(src, dst):
    """2D 刚体（旋转+平移）最小二乘：src (N,2) → dst (N,2)。"""
    cs, cd = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - cd
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


@torch.no_grad()
def eval_real(model_rb, model_sl, device):
    model_rb.eval()
    model_sl.eval()
    import open3d as o3d
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from sklearn.cluster import DBSCAN
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
    # 端面中心：GT 筋端带 (y,z) 中位数（≈截面中心，oracle 口径）
    band = (t_gt - t0 > -30) & (t_gt - t0 < 80)
    cy, cz = np.median(gts[band, 1] - c[1]), np.median(gts[band, 2] - c[2])
    win = (t_all > t0 - 300) & (t_all < t0 + 1750)
    p = np.column_stack([t_all[win] - t0, pts[win, 1] - c[1] - cy,
                         pts[win, 2] - c[2] - cz]).astype(np.float32)
    rng_cap = np.random.default_rng(0)
    if len(p) > REBAR_CAP:  # 与训练同口径下采样，对齐两域密度（T-022 教训）
        p = p[rng_cap.choice(len(p), REBAR_CAP, replace=False)]
    p = prefilter_far(p, 1, 2, 0)
    feat = pillarize(p, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
    dets = decode(model_rb(torch.from_numpy(feat[None]).to(device)),
                  CELL, END_ORIGIN, topk=32)[0]
    # 钢筋 GT：手筛 12 根全云 3D 分簇（根数 GT=标注数；端带簇心=插入位置，
    # 簇内最大 t−t0=外露 GT）。eps60 可能带出小碎片簇 → 按大小取前 12。
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
    # 人工标注圆心 (h2,h3) → 本框架：12 对对应两轮匈牙利 + 2D 刚体拟合
    Rm, tm = np.eye(2), np.zeros(2)
    for _ in range(2):
        ri, ci = linear_sum_assignment(cdist(circ @ Rm.T + tm, gt_uv))
        Rm, tm = _fit_rigid_2d(circ[ri], gt_uv[ci])
    fit_res = float(np.linalg.norm(
        (circ @ Rm.T + tm)[ri] - gt_uv[ci], axis=1).mean())
    if fit_res < 30.0:  # 拟合可信 → 位置/数量 GT 以人工标注为准
        gt_uv = (circ @ Rm.T + tm).astype(np.float32)
        gt_n = len(circ)
    else:
        gt_n = len(gt_uv)
    pos_err, expo_err = [], []
    si, gi = match_dets(dets[:, :2], gt_uv, gate=POS_GATE)
    for a, b in zip(si, gi):
        pos_err.append(float(np.linalg.norm(dets[a, :2] - gt_uv[b])))
        expo_err.append(abs(measure_exposed(p, dets[a, :2]) - gt_expo[b]))
    # 套筒端：构件另一端，面心=端带全点中位数（该端无筋束）
    t1 = float(t_all.max())
    wins = (t_all > t1 - 300) & (t_all < t1 + 80)
    band2 = np.abs(t_all[wins] - t1) < 50
    scy, scz = (np.median(pts[wins][band2, 1] - c[1]),
                np.median(pts[wins][band2, 2] - c[2]))
    ps = np.column_stack([t_all[wins] - t1, pts[wins, 1] - c[1] - scy,
                          pts[wins, 2] - c[2] - scz]).astype(np.float32)
    if len(ps) > SLEEVE_CAP:
        ps = ps[rng_cap.choice(len(ps), SLEEVE_CAP, replace=False)]
    feats = pillarize(ps, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
    dets_s = decode(model_sl(torch.from_numpy(feats[None]).to(device)),
                    CELL, END_ORIGIN, topk=32)[0]
    # 套筒 GT：sleeve_gt_manual.json 12 圆心，同一刚体变换 + 两端横向中心差
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
            "外露量测值mm": [round(v, 1) for v in
                             sorted(measure_exposed(p, d) for d in dets[:, :2])],
            "外露GTmm": [round(v, 1) for v in sorted(gt_expo)],
            "套筒GT环数": len(s_uv), "套筒检出": len(dets_s),
            "套筒匹配": len(si_s), "套筒位置p50mm": med(spos)}


def _pct(x):
    return f"{x:.0%}" if x is not None else "—"


LEGEND = ("指标定义：根数全对=检出峰数==GT数(严格匹配)；根数偏差=mean(检出-GT)；"
          "召回=120mm门限内匹配上的GT占比；误检率=1-精确率(匹配/全部检出)；"
          "位置p50=匹配对距离中位数；规格命中=匹配对直径/内径档完全一致占比；"
          "外露/高度误差=几何量测值与GT之差的中位数(量测精度，不是外露长度本身)；"
          "负视图检出=端别自决能力(钢筋头看套筒端/套筒头看出筋端，应全 0)")


def fmt_ends(tag, s):
    line = (f"    [{tag}] 根数全对 {_pct(s['count_acc'])} | "
            f"根数平均偏差 {s['n_bias']:+.2f} 根 | 检出召回 {_pct(s['recall'])} | "
            f"误检率 {_pct(1 - s['precision'])} | 位置p50 {s['pos_p50']}mm")
    if s.get("spec_hit") is not None:
        line += f" | 规格命中 {_pct(s['spec_hit'])}"
    if s.get("expo_p50") is not None:
        line += f" | 外露量测误差p50 {s['expo_p50']}mm"
    if s.get("neg_det_mean") is not None:
        line += (f" | 负视图件均检出 {s['neg_det_mean']}"
                 f"(全零率 {_pct(s['neg_clean'])})")
    return line


def fmt_stirrup(s):
    if s is None:
        return "    [合成-箍筋] 无箍筋样本"
    return (f"    [合成-箍筋] 榀数全对 {_pct(s['count_acc'])} | "
            f"腿位p50 {s['leg_pos_p50']}mm | "
            f"高度量测误差p50 {s['height_p50']}mm | 直径命中 {_pct(s['dia_hit'])}")


def fmt_real(r):
    return (f"    [真实场景] 钢筋 检出{r['钢筋检出']}/GT{r['钢筋GT根数']}根"
            f"(匹配{r['钢筋匹配']}) 位置p50 {r['钢筋位置p50mm']}mm | "
            f"外露量测误差p50 {r['钢筋外露量测误差p50mm']}mm\n"
            f"               外露量测值mm {r['外露量测值mm']}\n"
            f"               外露GT mm   {r['外露GTmm']}\n"
            f"               套筒 检出{r['套筒检出']}/GT{r['套筒GT环数']}环"
            f"(匹配{r['套筒匹配']}) 位置p50 {r['套筒位置p50mm']}mm | "
            f"标注↔点云拟合残差 {r['钢筋标注拟合残差mm']}mm")


# ── 主流程 ──────────────────────────────────────────────────────────────
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-ends", type=int, default=4)
    ap.add_argument("--batch-st", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", type=str, default="head_bev")
    args = ap.parse_args(argv)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    preprocess()
    ztr = load_bev_cache(anchor_project_path(CACHE), "train")
    zte = load_bev_cache(anchor_project_path(CACHE), "test")
    zva = load_bev_cache(anchor_project_path(CACHE), "val")
    all_idx = list(range(len(ztr["rb_n"])))
    st_gt_all = ztr["st_gt"]
    st_idx = [j for j in all_idx if len(st_gt_all[j]) > 0]
    print(f"train 件数 {len(all_idx)}（含箍筋 {len(st_idx)}）", flush=True)

    dl_rb = DataLoader(BevDataset(ztr, "rebar", all_idx, True),
                       args.batch_ends, shuffle=True, num_workers=0,
                       collate_fn=collate)
    dl_sl = DataLoader(BevDataset(ztr, "sleeve", all_idx, True),
                       args.batch_ends, shuffle=True, num_workers=0,
                       collate_fn=collate)
    dl_st = DataLoader(BevDataset(ztr, "stirrup", st_idx, True),
                       args.batch_st, shuffle=True, num_workers=0,
                       collate_fn=collate)
    model_rb = PillarBEVNet(len(REBAR_DIAM_TABLE), in_ch=8).to(device)
    model_sl = PillarBEVNet(len(SLEEVE_IDIAM_TABLE), in_ch=8).to(device)
    model_st = PillarBEVNet(len(STIRRUP_DIAM_TABLE)).to(device)
    opt = torch.optim.AdamW(sum([list(m.parameters()) for m in
                                 (model_rb, model_sl, model_st)], []),
                            lr=args.lr)
    ckpt_dir = anchor_project_path(f"out/exp1_train/{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    best_score = -1.0

    n_steps = max(len(dl_rb), len(dl_sl), len(dl_st))
    for ep in range(1, args.epochs + 1):
        t0, tot = time.time(), 0.0
        it_rb, it_sl, it_st = iter(dl_rb), iter(dl_sl), cycle(dl_st)
        model_rb.train()
        for _ in range(n_steps):
            loss_sum = 0.0
            opt.zero_grad()
            for it, model, w_off in ((it_rb, model_rb, 1.0),
                                     (it_sl, model_sl, 1.0),
                                     (it_st, model_st, 1.0)):
                try:
                    feats, heats, offs, clss, _ = next(it)
                except StopIteration:
                    continue
                feats, heats = feats.to(device), heats.to(device)
                offs, clss = offs.to(device), clss.to(device)
                loss, _ = bev_loss(model(feats), heats, offs, clss)
                loss.backward()
                loss_sum += float(loss)
            opt.step()
            tot += loss_sum
        msg = (f"[ep {ep:3d}/{args.epochs}] loss {tot / n_steps:.4f} "
               f"({time.time() - t0:.0f}s)")
        if ep % N_EVAL == 0 or ep == args.epochs:
            s_rb = eval_ends(model_rb, zte, "rebar", device)
            s_rb.update(eval_neg_views(model_rb, zte, "rebar", device))
            s_sl = eval_ends(model_sl, zte, "sleeve", device)
            s_sl.update(eval_neg_views(model_sl, zte, "sleeve", device))
            s_st = eval_stirrup(model_st, zte, device)
            r = eval_real(model_rb, model_sl, device)
            # val 综合分 → best.pth（治训练后期退化：v6c 钢筋召回 ep15
            # 88% → ep30 44%，固定取最后 epoch 不可靠）
            v_rb = eval_ends(model_rb, zva, "rebar", device)
            v_sl = eval_ends(model_sl, zva, "sleeve", device)
            v_st = eval_stirrup(model_st, zva, device)
            score = (v_rb["recall"] + v_rb["count_acc"]
                     + v_sl["count_acc"] + (v_st["count_acc"]
                                            if v_st else 0.0))
            msg += (f"\n{LEGEND}\n" if ep == N_EVAL else "\n")
            msg += (fmt_ends("合成-钢筋", s_rb) + "\n" + fmt_ends("合成-套筒", s_sl)
                    + "\n" + fmt_stirrup(s_st) + "\n" + fmt_real(r)
                    + f"\n    [val] score {score:.3f}"
                    f"（钢筋召回 {v_rb['recall']} 全对 {v_rb['count_acc']}"
                    f" | 套筒全对 {v_sl['count_acc']}）")
            history.append({"epoch": ep, "rebar": s_rb, "sleeve": s_sl,
                            "stirrup": s_st, "real": r,
                            "val_score": round(score, 4)})
            if score > best_score:
                best_score = score
                torch.save({"rb": model_rb.state_dict(),
                            "sl": model_sl.state_dict(),
                            "st": model_st.state_dict(), "epoch": ep,
                            "val_score": score},
                           os.path.join(ckpt_dir, "best.pth"))
                msg += f" ← best"
        print(msg, flush=True)
        torch.save({"rb": model_rb.state_dict(), "sl": model_sl.state_dict(),
                    "st": model_st.state_dict(), "epoch": ep},
                   os.path.join(ckpt_dir, "last.pth"))
    with open(os.path.join(ckpt_dir, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    print(f"完成 → {ckpt_dir}")


if __name__ == "__main__":
    main()
