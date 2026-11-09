# -*- coding: utf-8 -*-
"""
v6 阶段 A 训练入口——连接面语义分类（mini PointNet → 迭代 RANSAC）。

验收口径（docs/handover.md T-033）：三连接面（出筋/套筒/顶）检出率、
法向角误差、偏移误差；基线=经典版（本机复核：出筋 teeth 15/20·0.87°·
13.1mm；套筒经典全失 0/20；顶面 100%·0.61°·7.0mm）——打平或超过才有
存在意义，套筒面是经典法的结构缺口。

用法（cwd=src）：python -m experiments.exp1_train face [--epochs 60]
（权重落盘 out/exp1_train/face_sem；训练缓存 out/exp1_train/cache/face）
"""
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import set_seed, anchor_project_path
from lib.e2e.crop_dataset import load_split
from lib.e2e.face_sem import (N_CLASS, MiniPointNet, face_band_labels,
                              voxelize_with_labels, predict_bands,
                              faces_from_bands, refine_face_planes)

CACHE = "out/exp1_train/cache/face"
VOX = 16.0  # v5：32→16mm（面片判别特征分辨率翻倍）


# ── 缓存：体素心 + 多数标签（一次） ─────────────────────────────────────
def preprocess():
    for split in ("train", "val", "test"):
        out = anchor_project_path(os.path.join(CACHE, f"{split}.npz"))
        if os.path.exists(out):
            continue
        paths = load_split(anchor_project_path("mock/inuse"), split)
        pts_all, lab_all, nrm_all, geo_all = [], [], [], []
        for f in paths:
            d = np.load(f, allow_pickle=False)
            pts = d["points"].astype(np.float32)
            lbls = d["labels"].astype(np.int64)
            meta = json.loads(str(d["meta_json"]))
            lab = face_band_labels(pts, lbls, meta)
            cents, vlab, nrm, geo = voxelize_with_labels(pts, lab, VOX)
            cents = cents - cents.mean(0)  # 质心系（与推理一致）
            pts_all.append(cents.astype(np.float16))
            lab_all.append(vlab.astype(np.int8))
            nrm_all.append(nrm.astype(np.float16))
            geo_all.append(geo.astype(np.float16))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out,
                            pts=np.array(pts_all, dtype=object),
                            lab=np.array(lab_all, dtype=object),
                            nrm=np.array(nrm_all, dtype=object),
                            geo=np.array(geo_all, dtype=object))
        print(f"[cache] {split}: {len(paths)} 件 → {out}", flush=True)


class FaceBandDataset(Dataset):
    MAXN = 32768  # 定长（v5 体素 16mm 点数翻倍；-1 标签被 loss 忽略）

    def __init__(self, z, train):
        self.pts = z["pts"]  # __init__ 物化（P-033）
        self.lab = z["lab"]
        self.nrm = z["nrm"]
        self.geo = z["geo"]
        self.train = train
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, i):
        p = self.pts[i].astype(np.float32)
        n = self.nrm[i].astype(np.float32)
        g = self.geo[i].astype(np.float32)
        lab = self.lab[i].astype(np.int64)
        if self.train:  # 绕 z 旋转（坐标与法向同步）+ 丢点
            th = self.rng.uniform(0, 2 * np.pi)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], np.float32)
            p = p @ R.T
            n = n @ R.T
            keep = self.rng.random(len(p)) > 0.1
            p, n, g, lab = p[keep], n[keep], g[keep], lab[keep]
        from lib.e2e.face_sem import axial_features
        ax_f = axial_features(p)  # 旋转后重算（部署同序）
        m = len(p)
        if m > self.MAXN:  # 随机截断（等价丢点增广）
            idx = self.rng.choice(m, self.MAXN, replace=False)
            p, n, g, ax_f, lab, m = (p[idx], n[idx], g[idx], ax_f[idx],
                                     lab[idx], self.MAXN)
        x = np.zeros((self.MAXN, 10), np.float32)
        lp = np.full(self.MAXN, -1, np.int64)
        x[:m, :3] = p / 1000.0
        x[:m, 3:6] = n
        x[:m, 6:8] = ax_f
        x[:m, 8] = g[:, 0]
        x[:m, 9] = g[:, 1] / 50.0
        lp[:m] = lab
        return torch.from_numpy(x), torch.from_numpy(lp)


# ── 验收：三连接面 vs GT ────────────────────────────────────────────────
@torch.no_grad()
def eval_faces(model, files, device, n_max=56, verbose=False):
    model.eval()
    stat = {"rb": [0, 0, []], "sl": [0, 0, []], "top": [0, 0, []]}
    # 部署路径指标（T-036：评价指标对齐全链 sem_faces 真实消费量）——
    # 掩码支撑（查全/查准）+ refine RANSAC 平面角/偏移。16mm 体素
    # RANSAC 的"偏移"是评估路径伪影（2026-10-25 诊断：p50 124-135mm
    # vs 部署 4-12mm），保留仅作历史对照。
    deploy = {t: {"mask": [0, 0, 0], "plane": [0, []]}
              for t in ("rb", "sl", "top")}
    times, ious = [], {c: [] for c in range(N_CLASS)}
    for f in files[:n_max]:
        d = np.load(f, allow_pickle=False)
        pts = d["points"].astype(np.float32)
        lbls = d["labels"].astype(np.int64)
        meta = json.loads(str(d["meta_json"]))
        lab = face_band_labels(pts, lbls, meta)
        cents_gt, vlab, _, _ = voxelize_with_labels(
            pts, lab, VOX, rng=np.random.default_rng(0))
        t0 = time.time()
        cents, nrm_pred, prob = predict_bands(model, pts, device, vox=VOX)
        times.append(time.time() - t0)
        pred = prob.argmax(1)
        for c in range(N_CLASS):
            inter = ((pred == c) & (vlab == c)).sum()
            union = ((pred == c) | (vlab == c)).sum()
            if union:
                ious[c].append(inter / union)
        c0 = pts.mean(0)
        # 部署路径（sem_faces 同序：质心系 kNN 传播 → 5mm 加权法向约束）
        from scipy.spatial import cKDTree as _KDTree
        p_c = pts - c0
        _, ixp = _KDTree(cents).query(p_c, k=1, workers=-1)
        prob_pt = prob[ixp]
        for cls, tag in ((1, "rb"), (2, "sl")):
            mk = prob_pt[:, cls] > 0.5
            ms = deploy[tag]["mask"]
            ms[0] += int((mk & (lab == cls)).sum())
            ms[1] += int(mk.sum())
            ms[2] += int((lab == cls).sum())
            pls = refine_face_planes(p_c, mk, cents, nrm_pred, prob, cls)
            if pls:
                pf = pls[0]
                n = pf["n"] / np.linalg.norm(pf["n"])
                ang = float(np.degrees(np.arccos(
                    np.clip(abs(n @ np.array([1.0, 0, 0])), -1, 1))))
                ex_c = float(meta[f"{'rebar' if tag == 'rb' else 'sleeve'}"
                                  f"_end_x"]) - c0[0]
                sgn = 1.0 if tag == "rb" else -1.0
                off = float(sgn * (pf["d"] * float(np.sign(n[0])) - ex_c))
                deploy[tag]["plane"][0] += 1
                deploy[tag]["plane"][1].append((ang, abs(off)))
        faces_all = faces_from_bands(cents, prob, nrm=nrm_pred)
        faces = {k: (v[0] if v else None) for k, v in faces_all.items()}
        gts = {"rb": (float(meta["rebar_end_x"]), np.array([1.0, 0, 0])),
               "sl": (float(meta["sleeve_end_x"]), np.array([-1.0, 0, 0]))}
        for tag, (end_x, gn) in gts.items():
            band = np.abs(pts[:, 0] - end_x) < 60.0
            fc = pts[band].mean(0) - c0
            gd = float(fc @ gn)
            stat[tag][0] += 1
            pf = faces["rebar" if tag == "rb" else "sleeve"]
            if pf is not None:
                ang, dd = _plane_err(pf, gn, gd)
                if ang < 10.0 and dd < 200.0:
                    stat[tag][1] += 1
                    stat[tag][2].append((ang, dd))
        hps = meta.get("hairpin_gt") or []
        if hps:
            gn = np.array([0.0, 0, 1.0])
            gd = float(hps[0]["top_z"] - c0[2])
            stat["top"][0] += 1
            mk = prob_pt[:, 3] > 0.5
            ms = deploy["top"]["mask"]
            ms[0] += int((mk & (lab == 3)).sum())
            ms[1] += int(mk.sum())
            ms[2] += int((lab == 3).sum())
            pls = refine_face_planes(p_c, mk, cents, nrm_pred, prob, 3)
            if pls:
                pf2 = pls[0]
                n = pf2["n"] / np.linalg.norm(pf2["n"])
                ang = float(np.degrees(np.arccos(
                    np.clip(abs(n @ gn), -1, 1))))
                off = abs(float(pf2["d"] * float(np.sign(n[2]))) - gd)
                deploy["top"]["plane"][0] += 1
                deploy["top"]["plane"][1].append((ang, off))
            pf = faces["top"]
            if pf is not None:
                ang, dd = _plane_err(pf, gn, gd)
                if ang < 10.0 and dd < 200.0:
                    stat["top"][1] += 1
                    stat["top"][2].append((ang, dd))
    model.train()
    out = {"耗时/件s": round(float(np.mean(times)), 2)}
    names = {0: "其他", 1: "纵筋面", 2: "套筒面", 3: "箍筋面"}
    out["IoU"] = {names[c]: round(float(np.mean(v)), 3)
                  for c, v in ious.items() if v}
    med = lambda x: round(float(np.median(x)), 2) if x else None
    for tag, name in (("rb", "出筋面"), ("sl", "套筒面"), ("top", "顶面")):
        tot, hit, errs = stat[tag]
        out[name] = {"检出": f"{hit}/{tot}", "角p50°": med([e[0] for e in errs]),
                     "偏移p50mm": med([e[1] for e in errs])}
        m, pl = deploy[tag]["mask"], deploy[tag]["plane"]
        out[f"部署_{name}"] = {
            "掩码查全": round(m[0] / max(m[2], 1), 3),
            "掩码查准": round(m[0] / max(m[1], 1), 3),
            "平面检出": f"{pl[0]}/{tot}",
            "角p50°": med([e[0] for e in pl[1]]),
            "偏移p50mm": med([e[1] for e in pl[1]])}
    return out


def _plane_err(pf, gn, gd):
    """预测面 vs GT 面：角误差（法向无向）+ 偏移误差（规范到 gn 方向）。"""
    n = pf["n"] / np.linalg.norm(pf["n"])
    d = pf["d"]
    if n @ gn < 0:
        n, d = -n, -d
    ang = float(np.degrees(np.arccos(np.clip(abs(n @ gn), -1, 1))))
    return ang, abs(d - gd)


def fmt(r):
    lines = [f"    [面验收] 耗时/件 {r['耗时/件s']}s | IoU {r['IoU']}"]
    for name in ("出筋面", "套筒面", "顶面"):
        s, sd = r[name], r[f"部署_{name}"]
        lines.append(f"    [面验收] {name}: 检出 {s['检出']} | "
                     f"角 p50 {s['角p50°']}° | 偏移 p50 {s['偏移p50mm']}mm"
                     f" ‖ 部署: 掩码 R{sd['掩码查全']}/P{sd['掩码查准']} |"
                     f" 平面 {sd['平面检出']}·{sd['角p50°']}°·"
                     f"{sd['偏移p50mm']}mm")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", type=str, default="face_sem")
    args = ap.parse_args(argv)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    preprocess()
    ztr = np.load(anchor_project_path(os.path.join(CACHE, "train.npz")),
                  allow_pickle=True)
    files_te = sorted(glob.glob(anchor_project_path(
        "mock/inuse/test/*.npz")))
    ds = FaceBandDataset(ztr, True)
    dl = DataLoader(ds, args.batch, shuffle=True, num_workers=0)

    # 类权重（逆频率开方——带类占比 ~5-15%，防其他类淹没）
    all_lab = np.concatenate([l.astype(np.int64) for l in ztr["lab"]])
    freq = np.bincount(all_lab, minlength=N_CLASS) / len(all_lab)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    w = w / w.mean()
    print(f"类频率 {np.round(freq, 4)} → 权重 {np.round(w, 2)}", flush=True)

    model = MiniPointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * len(dl))
    ckpt_dir = anchor_project_path(f"out/exp1_train/{args.tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        model.train()
        for p, lab in dl:
            p, lab = p.to(device), lab.to(device)
            logit = model(p)
            loss = torch.nn.functional.cross_entropy(
                logit.reshape(-1, N_CLASS), lab.reshape(-1),
                weight=torch.from_numpy(w).float().to(device),
                ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            sched.step()
        msg = (f"[ep {ep:3d}/{args.epochs}] loss {tot / max(nb, 1):.4f} "
               f"({time.time() - t0:.0f}s)")
        if ep % 20 == 0 or ep == args.epochs:
            r = eval_faces(model, files_te, device, n_max=20)
            msg += "\n" + fmt(r)
            history.append({"epoch": ep, "faces": r})
        print(msg, flush=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
    with open(os.path.join(ckpt_dir, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    print(f"完成 → {ckpt_dir}")


if __name__ == "__main__":
    main()
