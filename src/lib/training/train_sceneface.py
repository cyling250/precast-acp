"""
超点面网络训练（T-037 M2，docs/t037_scene_face_pipeline.md §6/§8）。

数据：out/exp1_train/cache/scene_cache/sc_*.npz（build_scene_cache 产物，超点级精简
缓存）；`__init__` 一次性物化（P-033），feat 以 f16 驻内存（16G 机器
全量 ~1.5GB），batch 时转 f32。

协议（§6）：D1:D2 batch 内 50:50；AdamW 1e-3 + cosine；100 epoch；
batch=8 场景；增广=绕 Z 旋转（质心/法向/extent 精确变换）+ 丢超点
10% + 密度抖动。

评估（§7.1/§8 M2 DoD）：前景 F1（超点级，每 epoch）；实例检出率
（§7.1 匹配判据，按可见度分桶 整面/部分/缝隙，每 eval_every 轮 +
终评 test）。

CLI：
  python -m experiments.exp1_train sceneface --tag sceneface
  python -m experiments.exp1_train sceneface --tag smoke --smoke  # 冒烟
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lib.scene import face_net as fn

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEF_CACHE = os.path.join(_PROJ, "out", "exp1_train", "cache",
                         "scene_cache")
DEF_CKPT = os.path.join(_PROJ, "out", "exp1_train", "sceneface")
MIN_FACE_PTS = 30     # GT 面计入指标的最小点数（与 M1 QA 对齐）
MIN_FACE_VIS = 30     # 计入检出率的最小可见点数（低于=证据不足）


# ── 数据集 ─────────────────────────────────────────────────────────


class SceneItem:
    __slots__ = ("feat", "adj", "w", "fg", "inst", "cent", "nrm",
                 "n_pts", "n_face", "pair_sp", "pair_face", "pair_cnt",
                 "sp_member", "member_types", "n_members", "type_tab",
                 "face_table", "meta", "domain", "split")

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.feat = z["feat"].astype(np.float16)
        self.adj = z["adj"].astype(np.int64)
        self.w = np.log1p(z["adj_w"].astype(np.float64)).astype(np.float16)
        self.fg = z["fg"].astype(np.int64)
        self.inst = z["face_key"].astype(np.int64)   # face_table 行号
        self.cent = z["cent"].astype(np.float32)
        self.nrm = z["nrm"].astype(np.float32)
        self.n_pts = z["n_pts"].astype(np.float32)
        self.n_face = z["n_face"].astype(np.float32)
        self.pair_sp = z["pair_sp"].astype(np.int64)
        self.pair_face = z["pair_face"].astype(np.int64)
        self.pair_cnt = z["pair_cnt"].astype(np.float32)
        self.face_table = z["face_table"]
        self.meta = json.loads(str(z["meta_json"]))
        self.domain = self.meta.get("domain", "d2")
        self.split = self.meta.get("split", "train")
        # ── 构件/类型标签（M3 段②③，全部由现有字段派生，零重生成）──
        ft = self.face_table
        n_mem = int(ft[:, 0].max()) + 1 if len(ft) else 0
        self.n_members = n_mem
        self.sp_member = np.where(self.inst >= 0,
                                  ft[np.clip(self.inst, 0, None), 0],
                                  -1).astype(np.int64)
        # 截面边数 = **去重后的 body 面数**（face_table 第3列 type：
        # 0=body 侧面 / 1=出筋端面 / 2=套筒端面 / 3=箍筋顶面）。旧派生
        # `行数−2` 把端面之外的 type1/2/3 和零点面/重复面都算成边
        # （2026-10-27 实锤：与真值一致率仅 45%，且 body 面表里存在
        # n/d 完全相同的重复行 + npts=0 影子面——同一物理面记两次）。
        self.member_types = np.array(
            [fn.type_bucket_of(_member_edges(ft, m)) for m in range(n_mem)],
            np.int64)
        self.type_tab = np.full(64, -1, np.int64)
        self.type_tab[:min(n_mem, 64)] = self.member_types[:min(n_mem, 64)]


def _member_edges(ft: np.ndarray, member: int) -> int:
    """构件截面边数：去重后的 body(type=0) 面数。重复面判据=法向
    |n·n'|>0.999 且 |d∓d'|<1.5mm（d 符号随法向翻转）。"""
    rows = ft[ft[:, 0] == member]
    body = rows[rows[:, 2] == 0]
    kept = []
    for r in body:
        n, d = r[3:6], float(r[6])
        if not any(abs(float(n @ n0)) > 0.999 and
                   (abs(d - d0) if float(n @ n0) > 0 else abs(d + d0)) < 1.5
                   for n0, d0 in kept):
            kept.append((n, d))
    return len(kept)


def load_items(cache_dir: str) -> list[SceneItem]:
    paths = sorted(
        os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
        if f.startswith("sc_") and f.endswith(".npz"))
    if not paths:
        raise FileNotFoundError(f"{cache_dir} 下无 sc_*.npz 缓存")
    return [SceneItem(p) for p in paths]


def augment(it: SceneItem, rng: np.random.Generator) -> dict:
    """绕 Z 旋转（质心 xy/法向 xy/extent 精确变换，z 量不变）→ 丢超点
    10% → 密度抖动。返回训练所需张量的 numpy 视图。"""
    feat = it.feat.astype(np.float32).copy()
    th = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(th), np.sin(th)
    x, y = feat[:, 0] - 0.5, feat[:, 1] - 0.5
    feat[:, 0] = c * x - s * y + 0.5
    feat[:, 1] = s * x + c * y + 0.5
    nx, ny = feat[:, 9].copy(), feat[:, 10].copy()
    feat[:, fn.SL_NRM.start] = c * nx - s * ny
    feat[:, fn.SL_NRM.start + 1] = s * nx + c * ny
    ex = np.expm1(7.0 * feat[:, 4])
    ey = np.expm1(7.0 * feat[:, 5])
    feat[:, 4] = np.log1p(ex * abs(c) + ey * abs(s)) / 7.0
    feat[:, 5] = np.log1p(ex * abs(s) + ey * abs(c)) / 7.0

    K = len(feat)
    keep = rng.random(K) >= 0.10
    if keep.sum() < max(2, int(0.5 * K)):
        keep[:] = True                       # 极小图不丢
    remap = np.full(K, -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))

    feat[:, 3] += rng.uniform(-0.3, 0.3)
    feat[:, 13] += rng.uniform(-0.3, 0.3)

    em = keep[it.adj[:, 0]] & keep[it.adj[:, 1]]
    adj = remap[it.adj[em]]
    # cent 不随旋转（空间推离只消费两两距离，旋转不变量），仅随丢点过滤
    return {"feat": feat[keep], "adj": adj,
            "w": it.w[em].astype(np.float32),
            "fg": it.fg[keep], "inst": it.inst[keep],
            "cent": it.cent[keep], "memb": it.sp_member[keep],
            "n_pts": it.n_pts[keep], "type_tab": it.type_tab}


def as_eval_arrays(it: SceneItem) -> dict:
    return {"feat": it.feat.astype(np.float32),
            "adj": it.adj, "w": it.w.astype(np.float32),
            "fg": it.fg, "inst": it.inst, "cent": it.cent,
            "nrm": it.nrm, "n_pts": it.n_pts,
            "memb": it.sp_member, "type_tab": it.type_tab,
            "sp_member": it.sp_member,
            "pair_sp": it.pair_sp, "pair_face": it.pair_face,
            "pair_cnt": it.pair_cnt, "face_table": it.face_table}


def collate(items: list[dict], device: torch.device) -> dict:
    """多场景拼一张图（块对角，无跨场景边）。**inst 加场景偏移**——
    不同场景的 face_table 行号空间独立，全局拉聚前必须错开。"""
    xs, eis, ws, fgs, insts, cents = [], [], [], [], [], []
    membs, npts, type_tab = [], [], []
    off, soff = 0, 0
    for d in items:
        K = len(d["fg"])
        xs.append(d["feat"])
        eis.append(d["adj"] + off)
        ws.append(d["w"])
        fgs.append(d["fg"])
        insts.append(d["inst"] + soff * 100000)
        cents.append(d["cent"])
        membs.append(d["memb"] + soff * 64)      # 构件 gid：场景×64 槽
        npts.append(d["n_pts"].astype(np.float32))
        type_tab.append(d["type_tab"])           # (64,) gid→类型档，-1 空
        off += K
        soff += 1
    return {
        "x": torch.from_numpy(np.concatenate(xs)).float().to(device),
        "edge_index": torch.from_numpy(
            np.concatenate(eis)).t().contiguous().long().to(device),
        "w": torch.from_numpy(np.concatenate(ws)).float().to(device),
        "fg": torch.from_numpy(np.concatenate(fgs)).long().to(device),
        "inst": torch.from_numpy(np.concatenate(insts)).long().to(device),
        "cent": torch.from_numpy(np.concatenate(cents)).float().to(device),
        "memb": torch.from_numpy(np.concatenate(membs)).long().to(device),
        "n_pts": torch.from_numpy(np.concatenate(npts)).float().to(device),
        "type_tab": torch.from_numpy(
            np.concatenate(type_tab)).long().to(device),
    }


# ── 评估 ───────────────────────────────────────────────────────────


@torch.no_grad()
def forward_item(model, d: dict, device: torch.device):
    x = torch.from_numpy(d["feat"]).float().to(device)
    ei = torch.from_numpy(d["adj"]).t().contiguous().long().to(device)
    w = torch.from_numpy(d["w"]).float().to(device)
    lg, emb, memb, trunk = model(x, ei, w)
    fg_prob = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
    return fg_prob, emb.cpu().numpy(), memb.cpu().numpy(), trunk.cpu().numpy()


@torch.no_grad()
def eval_member_type(model, items: list[SceneItem], device,
                     tau: float = 0.6) -> dict:
    """②③逐头验收：构件聚类 pairwise P/R/F1（列联表精确式，逐场景解码
    后跨场景累计分子分母）+ 类型档准确率。构件簇=构件嵌入 τ 合并
    （种子同 decode 语义：邻接∧emb<τ 连通）。

    tau 可调：构件嵌入的尺度与面嵌入不同（面级 τ=0.6 不能假设适用），
    τ 应在 val 上选、test 上报告（指标纪律）。"""
    from math import comb
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    model.eval()
    tp = pd = rd = 0
    n_type = ok_type = 0
    for it in items:
        d = as_eval_arrays(it)
        fg_prob, _emb, memb_np, trunk = forward_item(model, d, device)
        # GT 构件标签（仅前景超点）
        gt = d.get("member_gt")
        if gt is None:
            gt = np.where(d["fg"] > 0, it.sp_member, -1)
        fg = fg_prob >= 0.5
        # 预测簇：邻接 ∧ memb 距 < tau 连通（前景内）
        a, b = d["adj"][:, 0], d["adj"][:, 1]
        eidx = []
        m1 = fg[a] & fg[b]
        if m1.any():
            dist = np.linalg.norm(memb_np[a[m1]] - memb_np[b[m1]], axis=1)
            eidx.append(np.column_stack([a[m1], b[m1]])[dist < tau])
        pred = np.full(len(fg), -1, np.int64)
        if eidx:
            e = np.unique(np.vstack(eidx), axis=0)
            e = np.vstack([e, np.column_stack([np.flatnonzero(fg)] * 2)])
            g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                           shape=(len(fg), len(fg)))
            _n, comp = connected_components(g, directed=False)
            pred = np.where(fg, comp, -1)
        # 列联表（前景内，GT≥0 且 pred≥0）
        sel = (gt >= 0) & (pred >= 0)
        if sel.any():
            ck = pred[sel].astype(np.int64) * 100000 + gt[sel]
            u, cnt = np.unique(ck, return_counts=True)
            tp += sum(int(comb(int(c), 2)) for c in cnt)
            pc = np.bincount(pred[sel].astype(np.int64))
            gk = np.bincount(gt[sel].astype(np.int64))
            pd += sum(int(comb(int(c), 2)) for c in pc if c > 1)
            rd += sum(int(comb(int(c), 2)) for c in gk if c > 1)
        # 类型 acc：预测簇池化 → type_head；GT 构件的众数簇对齐
        gm = np.unique(gt[gt >= 0])
        if len(gm) == 0:
            continue
        gid_map = {int(g): k for k, g in enumerate(gm)}
        selm = np.isin(gt, gm)
        pooled = np.zeros((len(gm), trunk.shape[1]), np.float32)
        wv = d["n_pts"][selm]
        for k, g in enumerate(gm):
            pooled[k] = (trunk[selm][gt[selm] == g]
                         * wv[gt[selm] == g, None]).sum(0)
        pooled /= np.maximum(
            np.array([wv[gt[selm] == g].sum() for g in gm])[:, None], 1e-9)
        xt = torch.from_numpy(pooled).float().to(device)
        tl = model.type_head(xt).argmax(1).cpu().numpy()
        for k, g in enumerate(gm):
            m = gt == g
            if not m.any():
                continue
            pc_pred = pred[m]
            pc_pred = pc_pred[pc_pred >= 0]
            if len(pc_pred) == 0:
                n_type += 1
                continue
            dom = np.bincount(pc_pred).argmax()
            n_type += 1
            # 众数簇的池化特征（重池化预测簇）
            wv2 = d["n_pts"][m & (pred == dom)]
            if wv2.sum() <= 0:
                continue
            feat2 = (trunk[m & (pred == dom)] * wv2[:, None]).sum(0)                 / wv2.sum()
            tl2 = model.type_head(
                torch.from_numpy(feat2[None]).float().to(device))
            ok_type += int(int(tl2.argmax(1)) == int(it.member_types[g]))
    p = tp / max(pd, 1)
    r = tp / max(rd, 1)
    return {"pair_p": p, "pair_r": r,
            "pair_f1": 2 * p * r / max(p + r, 1e-9),
            "type_acc": ok_type / max(n_type, 1), "n_members": n_type}


@torch.no_grad()
def eval_fg(model, items: list[SceneItem], device) -> dict:
    """前景 F1（类 1）：批量前向（每批 8 图，无增广、无解码）。"""
    tp = fp = fn = 0
    model.eval()
    for i in range(0, len(items), 8):
        chunk = [as_eval_arrays(it) for it in items[i:i + 8]]
        b = collate(chunk, device)
        lg, _e, _m, _t = model(b["x"], b["edge_index"], b["w"])
        pred = lg.argmax(1).cpu().numpy()
        gts = np.concatenate([c["fg"] for c in chunk])
        tp += int(((pred == 1) & (gts == 1)).sum())
        fp += int(((pred == 1) & (gts == 0)).sum())
        fn += int(((pred == 0) & (gts == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"f1": 2 * p * r / max(p + r, 1e-9), "p": p, "r": r,
            "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def eval_instances(model, items: list[SceneItem], device) -> dict:
    """逐场景：解码实例 + §7.1 匹配 → 分桶检出率（DoD 口径）。"""
    model.eval()
    face_rows, face_rows_strict = [], []
    f1 = eval_fg(model, items, device)
    n_pred = 0
    for it in items:
        d = as_eval_arrays(it)
        fg_prob, emb, memb_np, trunk_np = forward_item(model, d, device)
        inst = fn.decode_instances(fg_prob, emb, d["cent"], d["nrm"],
                                   d["adj"], len(d["fg"]),
                                   n_pts=d["n_pts"])
        n_pred += int(inst.max()) + 1 if (inst >= 0).any() else 0
        # D17 终版：主口径=支撑覆盖∧法向角（名义法向参照，翘曲免疫）；
        # 偏移降为诊断分位数；严格 §7.1（含 30mm 偏移门）并列报告
        # buckets_strict 交用户裁定
        rows = fn.match_faces(inst, d["pair_sp"], d["pair_face"],
                              d["pair_cnt"], d["cent"], d["nrm"],
                              d["face_table"][:, 3:6],
                              d["face_table"][:, 6])
        rows_strict = fn.match_faces(inst, d["pair_sp"], d["pair_face"],
                                     d["pair_cnt"], d["cent"], d["nrm"],
                                     d["face_table"][:, 3:6],
                                     d["face_table"][:, 6],
                                     offset_gate=True)
        face_rows.append(rows)
        face_rows_strict.append(rows_strict)
    # 分桶聚合：可见率/可见点数取自 face_table 全量域口径（ft[:,16/17]）
    det, by_type, by_band, excl, no_support, _np, offs, angs = _aggregate(
        face_rows, items)
    det_strict, _, _, _, _, _, offs_strict, _ = _aggregate(
        face_rows_strict, items)
    det_strict_agg = det_strict
    out = {"fg_f1": f1["f1"], "n_pred_total": n_pred, "buckets": {},
           "by_type": {}, "by_band": by_band, "excluded": excl,
           "no_support": no_support,
           "offset_p50_p90": [round(float(np.percentile(offs, q)), 2)
                              for q in (50, 90)] if offs else None,
           "angle_p50_p90": [round(float(np.percentile(angs, q)), 3)
                             for q in (50, 90)] if angs else None,
           "buckets_strict": {b: {"n": n, "detected": h,
                                  "rate": round(h / max(n, 1), 4)}
                              for b, (n, h) in det_strict.items()},
           "offset_p50_p90_strict":
               [round(float(np.percentile(offs_strict, q)), 2)
                for q in (50, 90)] if offs_strict else None}
    for b, (n, hit) in det.items():
        out["buckets"][b] = {"n": n, "detected": hit,
                             "rate": round(hit / max(n, 1), 4)}
    for t, (n, hit) in by_type.items():
        out["by_type"][t] = {"n": n, "rate": round(hit / max(n, 1), 4)}
    return out


def _aggregate(face_rows, items):
    det = {"whole": [0, 0], "part": [0, 0], "gap": [0, 0]}
    excl = {"whole": 0, "part": 0, "gap": 0, "tiny": 0}
    n_pred = 0
    by_type = {t: [0, 0] for t in (0, 1, 2, 3)}
    by_band = {"30-100": [0, 0], "100-300": [0, 0], "300-": [0, 0]}
    no_support = 0
    offs = []          # 诊断：最佳候选偏移分布（翘曲 vs 30mm 判据线）
    angs = []
    for rows, it in zip(face_rows, items):
        ft = it.face_table
        for r in rows:
            fi = r["face_idx"]
            n_pts, n_vis, t = ft[fi, 16], ft[fi, 17], ft[fi, 2]
            if r["no_support"]:
                no_support += 1
                continue
            if n_pts < MIN_FACE_PTS:
                excl["tiny"] += 1
                continue
            vr = n_vis / max(n_pts, 1)
            b = "whole" if vr >= 0.8 else ("part" if vr >= 0.3 else "gap")
            if n_vis < MIN_FACE_VIS:
                excl[b] += 1
                continue
            det[b][0] += 1
            det[b][1] += int(r["matched"])
            offs.append(r["offset"])
            angs.append(r["angle"])
            by_type[int(t)][0] += 1
            by_type[int(t)][1] += int(r["matched"])
            band = "30-100" if n_pts < 100 else \
                ("100-300" if n_pts < 300 else "300-")
            by_band[band][0] += 1
            by_band[band][1] += int(r["matched"])
    return det, by_type, by_band, excl, no_support, n_pred, offs, angs


# ── 训练主循环 ─────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="sceneface")
    ap.add_argument("--cache-dir", default=DEF_CACHE)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="场景/批")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：子集+2 轮+eval_every=1")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，加载 best.pth 直接终评（修评估代码后复算用）")
    ap.add_argument("--init", default=None,
                    help="热启动权重（strict=False，缺的新头随机初始化）")
    ap.add_argument("--w-mem", type=float, default=2.0,
                    help="构件嵌入损失权重（M3b：新头欠压，诊断后上调）")
    ap.add_argument("--memb-pull", type=float, default=None,
                    help="构件 hinge pull 目标（默认用 face_net.MEMB_PULL_MARGIN）")
    ap.add_argument("--memb-margin", type=float, default=None,
                    help="构件推离 margin（默认用 face_net.MEMB_PUSH_MARGIN）")
    ap.add_argument("--memb-hard-neg", type=float, default=0.5,
                    help="推离难负样本保留比例（1.0=全用）")
    ap.add_argument("--memb-w-pull", type=float, default=3.0,
                    help="pull 项权重（pull 对数远多于 push）")
    ap.add_argument("--memb-tau", type=float, default=0.6,
                    help="构件聚类评估 τ（解码连通阈值）")
    ap.add_argument("--tau-sweep", default=None,
                    help="逗号分隔 τ 列表：逐 τ 报告 pairwise F1（val 选参）")
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.eval_every = 2, 1

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    items = load_items(args.cache_dir)
    if args.smoke:
        items = items[:24]
    pools = {"train": [], "val": [], "test": []}
    for idx, it in enumerate(items):
        pools[it.split].append(idx)
    tr_d1 = [i for i in pools["train"] if items[i].domain == "d1"]
    tr_d2 = [i for i in pools["train"] if items[i].domain == "d2"]
    va = pools["val"] or pools["test"] or pools["train"]
    # 训练池条目剥离评估专用数组（nrm/n_pts/face_table 等）——cent 保留
    # （空间推离损失要用），全量物化的内存预算（P-033 纪律下仍需 GB 级）
    for i in pools["train"]:
        it = items[i]
        it.nrm = it.n_face = it.face_table = None
    print(f"数据：train d1={len(tr_d1)} d2={len(tr_d2)} "
          f"val={len(pools['val'])} test={len(pools['test'])} "
          f"（加载 {time.time()-t0:.0f}s）→ {device}", flush=True)
    if not tr_d1 or not tr_d2:
        raise SystemExit("D1/D2 训练池为空——先建齐两类缓存")

    ckpt_dir = os.path.join(DEF_CKPT, args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    net = fn.SceneFaceNet().to(device)
    if args.init:
        ck0 = torch.load(args.init, map_location=device, weights_only=True)
        miss, unexp = net.load_state_dict(ck0["net"], strict=False)
        print(f"热启动 {args.init}: missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"SceneFaceNet 参数 {n_par}（红线 <100k）", flush=True)
    assert n_par < 100_000
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    half = args.batch // 2
    steps = max(1, int(np.ceil(max(len(tr_d2), 1) / half)))
    if args.smoke:
        steps = min(steps, 3)
    history = []

    def sample_batch(epoch: int, step: int) -> list[SceneItem]:
        # D1:D2 50:50：每池独立洗牌、循环取用（池小者跨 epoch 复用）
        pick = []
        for pool in (tr_d1, tr_d2):
            perm = rng.permutation(pool)
            k = (epoch * steps + step) * half
            pick.extend([items[j] for j in perm[k % len(pool)
                                               :(k % len(pool)) + half]])
        return pick

    for epoch in range(0 if args.eval_only else args.epochs):
        net.train()
        ep_fg = ep_emb = ep_mem = ep_type = 0.0
        te = time.time()
        kw = {}
        if args.memb_pull is not None:
            kw["pull_margin"] = args.memb_pull
        if args.memb_margin is not None:
            kw["margin"] = args.memb_margin
        for step in range(steps):
            batch = [augment(it, rng) for it in sample_batch(epoch, step)]
            b = collate(batch, device)
            lg, emb, memb, trunk = net(b["x"], b["edge_index"], b["w"])
            l_fg = fn.foreground_loss(lg, b["fg"])
            l_emb = fn.embedding_loss(emb, b["inst"], b["edge_index"],
                                      cent=b["cent"])
            # ②构件判别（M3b：hinge pull 对齐解码条件 + margin 2.5 推离，
            #   诊断实锤方差 pull 被大构件稀释、margin 1.5 压死解码 τ）
            l_mem = fn.member_embedding_loss(
                memb, b["memb"], b["edge_index"], cent=b["cent"],
                hard_neg_frac=args.memb_hard_neg, w_pull=args.memb_w_pull,
                **kw) * args.w_mem
            # ③类型档：GT 构件分组池化 → MLP → CE（teacher-forcing 池化，
            # 梯度经 index_add 流回主干；推理期换预测簇，ROI 池化范式）
            gid, inv = torch.unique(b["memb"][b["memb"] >= 0],
                                    return_inverse=True)
            wsel = b["n_pts"][b["memb"] >= 0]
            pooled = torch.zeros(len(gid), trunk.size(1),
                                 device=trunk.device)
            pooled.index_add_(0, inv,
                              trunk[b["memb"] >= 0] * wsel.unsqueeze(1))
            pooled = pooled / torch.bincount(
                inv, weights=wsel, minlength=len(gid)).clamp_min(1e-9
                ).unsqueeze(1)
            type_logits = net.type_head(pooled)
            type_labels = b["type_tab"][gid]
            ok = type_labels >= 0
            l_type = nn.functional.cross_entropy(
                type_logits[ok], type_labels[ok]) if ok.any()                 else type_logits.sum() * 0.0
            loss = l_fg + l_emb + l_mem + l_type
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_fg += float(l_fg.detach())
            ep_emb += float(l_emb.detach())
            ep_mem += float(l_mem.detach())
            ep_type += float(l_type.detach())
        sched.step()
        f1 = eval_fg(net, [items[i] for i in va], device)
        rec = {"epoch": epoch, "loss_fg": round(ep_fg / steps, 4),
               "loss_emb": round(ep_emb / steps, 4),
               "loss_mem": round(ep_mem / steps, 4),
               "loss_type": round(ep_type / steps, 4),
               "val_fg_f1": round(f1["f1"], 4),
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - te)}
        history.append(rec)
        line = (f"ep{epoch:03d} fg={rec['loss_fg']:.3f} "
                f"emb={rec['loss_emb']:.3f} valF1={f1['f1']:.4f} "
                f"({rec['sec']}s)")
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            det = eval_instances(net, [items[i] for i in va], device)
            mt = eval_member_type(net, [items[i] for i in va], device,
                                  tau=args.memb_tau)
            rec["val_det"] = det["buckets"]
            rec["val_member_pair_f1"] = round(mt["pair_f1"], 4)
            rec["val_member_p"] = round(mt["pair_p"], 4)
            rec["val_member_r"] = round(mt["pair_r"], 4)
            rec["val_type_acc"] = round(mt["type_acc"], 4)
            line += (f" | mem={mt['pair_p']:.3f}/{mt['pair_r']:.3f}"
                     f"/{mt['pair_f1']:.3f} type={mt['type_acc']:.3f}"
                     " | " + " ".join(
                         f"{b}={det['buckets'][b]['rate']:.3f}"
                         f"({det['buckets'][b]['n']})"
                         for b in ("whole", "part", "gap")))
        print(line, flush=True)
        torch.save({"net": net.state_dict(), "epoch": epoch,
                    "args": vars(args), "history": history},
                   os.path.join(ckpt_dir, "last.pth"))
        # best 选择：M3 核心 DoD 是构件聚类 F1（fg 已达标且稳定），
        # 按 fg F1 选会停在无关紧要的波动上
        best = max(((r.get("val_member_pair_f1", 0.0), r["val_fg_f1"], e)
                    for e, r in enumerate(history)))
        if best[2] == epoch:
            torch.save({"net": net.state_dict(), "epoch": epoch,
                        "args": vars(args)},
                       os.path.join(ckpt_dir, "best.pth"))
        with open(os.path.join(ckpt_dir, "history.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"history": history, "n_params": n_par,
                       "data": {"train_d1": len(tr_d1),
                                "train_d2": len(tr_d2),
                                "val": len(pools["val"]),
                                "test": len(pools["test"])}},
                      f, ensure_ascii=False, indent=1)

    # ── 终评：test 分桶（DoD） ──────────────────────────────────
    net.load_state_dict(torch.load(
        os.path.join(ckpt_dir, "best.pth"),
        map_location=device, weights_only=True)["net"])
    # τ 选择纪律：val 上扫 → 取 val 最优 → test 上报告该 τ（超参选择不算
    # 指标作弊，但 τ 必须在 val 定、test 只读一次）
    taus = ([float(t) for t in args.tau_sweep.split(",")]
            if args.tau_sweep else [args.memb_tau])
    tau_sel = args.memb_tau
    tau_val = {}
    if len(taus) > 1 and pools["val"]:
        va_items = [items[i] for i in pools["val"]]
        for t in taus:
            mt_t = eval_member_type(net, va_items, device, tau=t)
            tau_val[t] = mt_t["pair_f1"]
        tau_sel = max(tau_val, key=lambda k: tau_val[k])
        print(f"τ 扫描(val): {[(k, round(v, 4)) for k, v in tau_val.items()]}"
              f" → 选 τ={tau_sel}", flush=True)
    for name, pool in (("val", pools["val"]), ("test", pools["test"])):
        if not pool:
            continue
        det = eval_instances(net, [items[i] for i in pool], device)
        mt = eval_member_type(net, [items[i] for i in pool], device,
                              tau=tau_sel)
        out = {"tag": args.tag, "split": name, "n_scenes": len(pool),
               "member_pair": {k: round(v, 4) for k, v in mt.items()
                               if k != "n_members"},
               "n_members": mt["n_members"], "tau": tau_sel,
               "tau_sweep_val": tau_val or None,
               "fg_f1": det["fg_f1"], "n_pred_total": det["n_pred_total"],
               "buckets": det["buckets"], "by_type": det["by_type"],
               "by_band": det["by_band"], "excluded": det["excluded"],
               "no_support": det["no_support"],
               "offset_p50_p90": det["offset_p50_p90"],
               "angle_p50_p90": det["angle_p50_p90"],
               "buckets_strict": det["buckets_strict"],
               "offset_p50_p90_strict": det["offset_p50_p90_strict"],
               "dod": {
                   "fg_f1_ge_0.95": det["fg_f1"] >= 0.95,
                   "member_pair_f1_ge_0.95": mt["pair_f1"] >= 0.95,
                   "type_acc_ge_0.95": mt["type_acc"] >= 0.95,
                   "whole_rate_ge_0.90":
                       det["buckets"]["whole"]["rate"] >= 0.90},
               }
        path = os.path.join(ckpt_dir, f"eval_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[{name}] F1={det['fg_f1']:.4f} 整面="
              f"{det['buckets']['whole']['rate']:.4f}"
              f"({det['buckets']['whole']['n']}) 部分="
              f"{det['buckets']['part']['rate']:.4f} 缝隙="
              f"{det['buckets']['gap']['rate']:.4f}"
              f"({det['buckets']['gap']['n']}) mem_pair="
              f"{mt['pair_f1']:.4f}(P{mt['pair_p']:.3f}/R{mt['pair_r']:.3f})"
              f" type={mt['type_acc']:.4f} τ={tau_sel} → {path}", flush=True)


if __name__ == "__main__":
    main()
