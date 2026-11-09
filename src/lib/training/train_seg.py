# -*- coding: utf-8 -*-
"""场景材质分割训练（场景适配配方，合成 test mIoU 0.956）。

配方（与构件级原版一致，勿改）：
- PointMLP 骨干，150mm 球查询补丁、每补丁 8192 点；
- 补丁生成前 3mm 体素细化（密度均衡）；
- 非重叠铺片（available 掩码 + 球采样 + kNN 补足），单位球归一；
- 类均衡：含筋(2/3)补丁全保留，纯混凝土补丁按 POS_RATIO 补足；
- 训练期旋转增广（绕 z 全角 + ±10° 倾斜）——防朝向捷径（合成件转 90°
  纵筋 IoU 0.949→0.300 的教训）。

场景差异：5 类（0 背景 1 混凝土 2 钢筋 3 箍筋 4 垫块，0/4 降权 0.5）、
每场景补丁数封顶（场景点量 ~1M 级）、补丁池一次预计算落盘。

用法（cwd=src）：python -m experiments.exp1_train seg --train --epochs 100
（补丁预计算 --patches / 类均衡合并 --merge / test 评估 --eval；
权重落盘 out/exp1_train/seg，补丁池 out/exp1_train/cache/seg_patches）
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from lib.segmentation import PointMLPSeg

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ROOT = os.path.join(PROJ, "mock", "scene", "v1")
PATCH_DIR = os.path.join(PROJ, "out", "exp1_train", "cache",
                         "seg_patches")
PATCH_RADIUS_MM = 150.0
NUM_POINTS = 8192
VOXEL_THIN_MM = 3.0
PATCHES_PER_SCENE = 32
# 4 类 = §7.6 架构口径（混凝土/钢筋/箍筋/噪声）：场景标签的垫块(4)
# 并入噪声桶(0)——它是构件下的垫木，对管线就是噪声；留作独立类会把
# 前景 mIoU 上限压到 0.75（ep11 前垫块 IoU 恒 0，2026-10-28）。
POS_RATIO = 0.8
N_CLASSES = 4
CLASS_NAMES = ("背景/噪声", "混凝土", "钢筋", "箍筋")
LR = 1e-3
WEIGHT_DECAY = 1e-4


def scene_files():
    out = []
    for mode in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, mode)
        if os.path.isdir(d):
            out += sorted(os.path.join(d, f) for f in os.listdir(d)
                          if f.endswith(".npz"))
    return out


def voxel_thin(pts, *arrays, voxel_mm=VOXEL_THIN_MM):
    keys = np.floor(pts / voxel_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return (pts[idx],) + tuple(a[idx] for a in arrays)


def tile_patches(pts, lbl, rng, cap=PATCHES_PER_SCENE):
    """非重叠铺片：球采样 + kNN 补足，单位球归一。返回 (P,8192,3),(P,8192)。"""
    tree = cKDTree(pts)
    available = np.ones(len(pts), bool)
    out_p, out_l = [], []
    n_fail = 0
    while available.sum() >= NUM_POINTS and len(out_p) < cap:
        avail_idx = np.flatnonzero(available)
        seed = int(rng.choice(avail_idx))
        ball = tree.query_ball_point(pts[seed], r=PATCH_RADIUS_MM)
        cand = np.array([i for i in ball if available[i]], dtype=np.int64)
        if len(cand) >= NUM_POINTS:
            chosen = rng.choice(cand, NUM_POINTS, replace=False)
        else:
            k_need = min(NUM_POINTS * 2, len(avail_idx))
            _d, knn = tree.query(pts[seed].reshape(1, -1), k=k_need)
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
        n_fail = 0
        available[chosen] = False
        pp = pts[chosen]
        pp = pp - pp.mean(0)
        scale = float(np.max(np.linalg.norm(pp, axis=1)))
        pp = pp / (scale + 1e-8)
        out_p.append(pp.astype(np.float32))
        out_l.append(lbl[chosen].astype(np.int64))
    if not out_p:
        return np.empty((0, NUM_POINTS, 3), np.float32), \
            np.empty((0, NUM_POINTS), np.int64)
    return np.stack(out_p), np.stack(out_l)


def cmd_patches(shard=0, num_shards=1):
    """一次性补丁预计算（3mm 细化 + 铺片）→ out/exp1_train/cache/seg_patches/all/*.npz
    + manifest.json（stem→split）；类均衡合并单独跑 --merge。"""
    os.makedirs(os.path.join(PATCH_DIR, "all"), exist_ok=True)
    files = scene_files()[shard::num_shards]
    print(f"补丁预计算 shard {shard}/{num_shards}: {len(files)} 场景",
          flush=True)
    t0 = time.time()
    manifest = {}
    mf_path = os.path.join(PATCH_DIR, "manifest.json")
    if os.path.exists(mf_path):
        manifest = json.load(open(mf_path, encoding="utf-8"))
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in manifest:
            z = np.load(f, allow_pickle=True)
            vis = z["visible"].astype(bool)
            pts, lbl = voxel_thin(z["points"][vis].astype(np.float32),
                                  z["class_id"][vis].astype(np.int64))
            meta = json.loads(str(z["scene_meta_json"]))
            rng = np.random.default_rng(0)
            P, L = tile_patches(pts, lbl, rng)
            np.savez(os.path.join(PATCH_DIR, "all", stem + ".npz"),
                     pts=P, label=L)
            manifest[stem] = {"split": meta.get("split") or "train",
                              "mode": meta["mode"], "n_patch": int(len(P))}
        if (i + 1) % 5 == 0:
            json.dump(manifest, open(mf_path, "w", encoding="utf-8"))
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
    json.dump(manifest, open(mf_path, "w", encoding="utf-8"))
    print(f"shard 完成 {len(files)} 场景 {time.time()-t0:.0f}s", flush=True)


def cmd_merge():
    """类均衡合并（按 manifest 的 split）→ pool_{split}.pt。"""
    manifest = json.load(open(os.path.join(PATCH_DIR, "manifest.json"),
                              encoding="utf-8"))
    for s in ("train", "val", "test"):
        stems = [k for k, v in manifest.items() if v["split"] == s]
        pts_l, lbl_l, pos_l = [], [], []
        for stem in stems:
            z = np.load(os.path.join(PATCH_DIR, "all", stem + ".npz"))
            P, L = z["pts"], z["label"]
            for j in range(len(P)):
                pos = float(((L[j] == 2) | (L[j] == 3)).mean())
                pts_l.append(P[j]); lbl_l.append(L[j]); pos_l.append(pos)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(pts_l))
        n_pos = sum(1 for p in pos_l if p > 0)
        n_con = len(pos_l) - n_pos
        n_con_keep = max(0, min(n_con, int(n_pos / POS_RATIO - n_pos)))
        kept, seen_con = [], 0
        for j in order:
            if pos_l[j] > 0:
                kept.append(j)
            elif seen_con < n_con_keep:
                kept.append(j)
                seen_con += 1
        kept = np.array(sorted(kept))
        pts = np.stack([pts_l[j] for j in kept])
        lbl = np.stack([lbl_l[j] for j in kept]).astype(np.int32)
        out = os.path.join(PATCH_DIR, f"pool_{s}.pt")
        torch.save({"pts": torch.from_numpy(pts),
                    "label": torch.from_numpy(lbl)}, out)
        print(f"[{s}] {len(stems)} 场景 {len(pts_l)} 补片 → 保留 "
              f"{len(kept)} (含筋 {n_pos}, 纯混凝土 {seen_con}) → {out} "
              f"({os.path.getsize(out)/1e9:.2f}GB)", flush=True)




def _batch_random_rotate(pts, max_tilt_deg=10.0):
    """训练增广：逐样本绕 z 全角 + 绕随机水平轴倾斜 ±max_tilt。"""
    B = pts.shape[0]
    dev = pts.device
    ang = torch.rand(B, device=dev) * (2 * math.pi)
    cz, sz = torch.cos(ang), torch.sin(ang)
    z0, z1 = torch.zeros_like(ang), torch.ones_like(ang)
    Rz = torch.stack([cz, -sz, z0, sz, cz, z0, z0, z0, z1],
                     dim=1).view(B, 3, 3)
    tilt = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_tilt_deg)
    az = torch.rand(B, device=dev) * (2 * math.pi)
    k = torch.stack([torch.cos(az), torch.sin(az),
                     torch.zeros_like(az)], dim=1)
    ct, st = torch.cos(tilt), torch.sin(tilt)
    eye = torch.eye(3, device=dev).unsqueeze(0).expand(B, 3, 3)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    Rt = eye + st.view(B, 1, 1) * K + (1 - ct).view(B, 1, 1) * (K @ K)
    return torch.einsum("bij,bnj->bni", Rz @ Rt, pts)


class SceneWeightedCE(torch.nn.Module):
    """原版 PerSampleWeightedCE 的 5 类适配：w_c = N/(5×n_c) 截断 15，
    背景(0)/垫块(4) 降权 0.5。st_boost>1 时箍筋类(3) 权重再乘（T-040
    修复清单#2：类 3 IoU 0.757 最弱，发夹细杆占比小、逐补丁归一后被
    max_weight 截断吃掉倾斜量）。"""

    def __init__(self, max_weight=15.0, st_boost=1.0):
        super().__init__()
        self.max_weight = max_weight
        self.st_boost = st_boost

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none")
        B, N = targets.shape
        onehot = torch.nn.functional.one_hot(targets, N_CLASSES).float()
        cnt = onehot.sum(1)                                  # (B,C)
        present = cnt > 0
        w = N / (N_CLASSES * cnt.clamp(min=1))
        w = torch.where(present, w, torch.ones_like(w))
        w[:, 0] *= 0.5                                       # 背景降权
        if self.st_boost != 1.0:
            w[:, 3] = w[:, 3] * self.st_boost                # 箍筋类倾斜
        w = w.clamp(max=self.max_weight)
        weight_map = torch.gather(w, 1, targets)
        # 原配方尺度：Σ(ce·w)/N——比 mean 大 N 倍。尺度影响 Adam 中
        # weight_decay 与数据梯度的相对强度（mean 尺度下 decay 相对
        # 强 4096 倍 → 权重塌向零 → 退化成全混凝土，2026-10-28 实锤）。
        return (ce * weight_map).sum(dim=1).mean() / targets.shape[1]


TRAIN_POINTS = 4096      # 8192 补片裁半：8192 点全 pairwise 距离矩阵
                         # batch16=4.3GB 触发 WDDM 显存超订阅换页（实测
                         # GPU util 26%）；4096 点下矩阵缩 4 倍不溢出。


def load_pool(split, device):
    d = torch.load(os.path.join(PATCH_DIR, f"pool_{split}.pt"),
                   map_location="cpu")
    lbl = d["label"][:, :TRAIN_POINTS].long()
    lbl = torch.where(lbl == 4, torch.zeros_like(lbl), lbl)   # 垫块→噪声
    return d["pts"][:, :TRAIN_POINTS], lbl


@torch.no_grad()
def eval_patches(model, pts, lbl, device, n=4000, rng=None):
    """补丁池上的逐类 IoU。"""
    model.eval()
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    idx = rng.permutation(len(pts))[:n] if rng else np.arange(n)
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        x = pts[b].to(device)
        pred = model(x, torch.zeros_like(x)).argmax(-1).cpu().numpy()
        y = lbl[b].numpy()
        for c in range(N_CLASSES):
            inter[c] += int(((pred == c) & (y == c)).sum())
            union[c] += int(((pred == c) | (y == c)).sum())
    iou = inter / np.maximum(union, 1)
    return float(iou[1:].mean()), iou   # 前景3类均值


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    t0 = time.time()
    tr_pts, tr_lbl = load_pool("train", device)
    va_pts, va_lbl = load_pool("val", device)
    print(f"train {len(tr_pts)} / val {len(va_pts)} 补片 "
          f"(载入 {time.time()-t0:.0f}s) → {device}", flush=True)
    model = PointMLPSeg(num_classes=N_CLASSES, in_channels=6).to(device)
    print(f"参数 {sum(p.numel() for p in model.parameters())}")
    crit = SceneWeightedCE(st_boost=args.st_boost)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(0)
    vrng = np.random.default_rng(42)
    srng = np.random.default_rng(7)
    # 箍筋补丁过采样（T-040 修复清单#2）：类均衡池只保证"含筋(2/3)"
    # 整体保留，箍筋补丁是其中少数——按含箍筋点 ≥1% 加权重复采样。
    st_frac = (tr_lbl == 3).float().mean(1).numpy()
    samp_w = np.where(st_frac >= 0.01, args.st_oversample, 1.0)
    samp_w = samp_w / samp_w.sum()
    print(f"箍筋补丁(≥1%) {int((st_frac >= 0.01).sum())}/{len(st_frac)} "
          f"过采样 ×{args.st_oversample}；st_boost={args.st_boost}",
          flush=True)
    best = -1.0
    out_dir = os.path.join(PROJ, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        t_ep = time.time()
        if args.st_oversample > 1.0:
            order = torch.from_numpy(
                srng.choice(len(tr_pts), size=args.samples,
                            replace=True, p=samp_w).astype(np.int64))
        else:
            order = torch.randperm(len(tr_pts))[:args.samples]
        losses = []
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x = tr_pts[idx].to(device)
            if args.aug:
                x = _batch_random_rotate(x)
            y = tr_lbl[idx].to(device)
            _ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if args.amp else torch.amp.autocast("cuda", enabled=False))
            with _ctx:
                logits = model(x, torch.zeros_like(x))
                loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        is_val_ep = (ep + 1) % args.val_every == 0 or ep == args.epochs - 1
        miou, iou = (eval_patches(model, va_pts, va_lbl, device,
                                  n=args.val_n, rng=vrng)
                     if is_val_ep else (best, np.zeros(N_CLASSES)))
        if not is_val_ep:
            torch.save({"model": model.state_dict(), "miou": best,
                        "ep": ep},
                       os.path.join(out_dir, "last.pth"))
            print(f"ep{ep}: loss={np.mean(losses):.3f} "
                  f"({time.time()-t_ep:.0f}s)", flush=True)
            continue
        print(f"ep{ep}: loss={np.mean(losses):.3f} val mIoU={miou:.3f} "
              + " ".join(f"{c}:{v:.2f}"
                         for c, v in zip(CLASS_NAMES, iou)), flush=True)
        ck = {"model": model.state_dict(), "miou": miou, "ep": ep}
        torch.save(ck, os.path.join(out_dir, "last.pth"))
        if miou > best:
            best = miou
            torch.save(ck, os.path.join(out_dir, "best.pth"))
    print(f"best val mIoU={best:.3f}")


def eval_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(PROJ, args.out_dir, args.ckpt_name),
                    map_location=device)
    model = PointMLPSeg(num_classes=N_CLASSES, in_channels=6).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    te_pts, te_lbl = load_pool("test", device)
    rng = np.random.default_rng(1)
    miou, iou = eval_patches(model, te_pts, te_lbl, device,
                             n=args.val_n, rng=rng)
    print(f"test mIoU(前景4类)={miou:.3f}  ep={ck.get('ep')} "
          f"val_mIoU={ck.get('miou'):.3f}")
    for c, v in zip(CLASS_NAMES, iou):
        print(f"  {c}: {v:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--val-n", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--amp", type=int, default=0)  # bf16 下模型塌向全混凝土（fp32 立刻逃出，2026-10-28 决断实验），禁用
    ap.add_argument("--out-dir", default="out/exp1_train/seg",
                    help="权重落盘目录（v2 消融用独立目录，不覆盖基线）")
    ap.add_argument("--st-boost", type=float, default=1.0,
                    help="箍筋类(3) CE 权重乘子（T-040 修复清单#2）")
    ap.add_argument("--st-oversample", type=float, default=1.0,
                    help="含箍筋点≥1%% 补丁的过采样倍数（1=关）")
    ap.add_argument("--ckpt-name", default="best.pth")
    args = ap.parse_args(argv)
    if args.patches:
        cmd_patches(args.shard, args.num_shards)
    if args.merge:
        cmd_merge()
    if args.train:
        train(args)
    if args.eval:
        eval_test(args)
    if not (args.patches or args.train or args.eval):
        print(__doc__)


if __name__ == "__main__":
    main()
