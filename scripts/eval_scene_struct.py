# -*- coding: utf-8 -*-
"""
T-037 M3 段②③④全量复评（docs/handover/t037_m3_struct_wip.md §4.3）。

管线：构件嵌入解码簇 → 逐簇 fit_member 面假设 → 面检出评估。

指标（全部并列 GT 上限，指标纪律 §1.6）：
- 构件聚类 pairwise P/R/F1（列联表精确式，跨场景累计分子分母）；
- 面检出率（双向 50% 判据，整面/部分/缝隙分桶）：
    预测面 f ↔ GT 面 fi 同时满足
      ① f 携带 fi 的可见承载 ≥50%：Σ_{s∈f} pair_cnt(s,fi) / denom(fi)
      ② fi 在 f 的总承载中占比 ≥50%：Σ_{s∈f} pair_cnt(s,fi) / Σ_{fi'} 同
    另：GT 面须 n_pts≥30 且 n_vis≥30（证据充分，与 M2 口径一致）；
- 逐超点面标签一致率（主指标，点级精确分摊）：预测面 f 的 GT 标签 =
  其检出（双向 50%）GT 面中承载最大者；超点承载点按 pair_cnt 分摊，
  落在其面标签上的点计分子，全部承载计分母；
- GT 上限并列：**GT 构件簇直接拟合**（同评价管线，隔离聚类误差 →
  纯拟合质量上限）；
- 诊断：预测面 vs GT 名义面的角度/偏移 p50/p90。

CLI：
  python -m scripts.eval_scene_struct \
      --ckpt ../out/exp1_train/sceneface/best.pth --tau 1.0 --split test
"""

import argparse
import json
import os
from math import comb

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from lib.training.train_sceneface import as_eval_arrays, load_items
from lib.scene import face_net as fn
from lib.scene.struct_fit import fit_member

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MIN_FACE_PTS = 30      # GT 面计入的最小点数（与 M2 口径对齐）
MIN_FACE_VIS = 30      # 计入的最小可见点数
LABEL_DEG = 10.0       # 预测面→GT 面标签匹配：法向角门
LABEL_MM = 60.0        # 预测面→GT 面标签匹配：偏移门（翘曲 σ40 + 拟合）


def decode_clusters(memb_np: np.ndarray, fg: np.ndarray, adj: np.ndarray,
                    tau: float) -> np.ndarray:
    """构件簇 = 邻接 ∧ 嵌入距<τ 连通（前景内），非前景 -1。"""
    pred = np.full(len(fg), -1, np.int64)
    a, b = adj[:, 0], adj[:, 1]
    m1 = fg[a] & fg[b]
    if not m1.any():
        return pred
    aa, bb = a[m1], b[m1]
    dist = np.linalg.norm(memb_np[aa] - memb_np[bb], axis=1)
    e = np.column_stack([aa[dist < tau], bb[dist < tau]])
    e = np.unique(e, axis=0)
    e = np.vstack([e, np.column_stack([np.flatnonzero(fg)] * 2)])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                   shape=(len(fg), len(fg)))
    _n, comp = connected_components(g, directed=False)
    return np.where(fg, comp, -1)


def _new_core() -> dict:
    return {"buckets": {"whole": [0, 0], "part": [0, 0], "gap": [0, 0]},
            "point_hit": 0.0, "point_tot": 0.0, "angles": [], "offsets": [],
            "n_faces_pred": 0, "n_members": 0,
            "f_tp": 0, "f_pred": 0, "f_gt": 0}


def face_eval_core(assign: np.ndarray, faces: list, pair_sp: np.ndarray,
                   pair_face: np.ndarray, pair_cnt: np.ndarray,
                   face_table: np.ndarray, sp_sel: np.ndarray,
                   eval_sp: np.ndarray = None) -> dict:
    """单构件组的面检出 + 点级命中（分母留给场景级统一口径）。

    assign：(M,) 组内各超点的预测面序号（-1 未归属），M=len(sp_sel)；
    faces ：fit_member 输出的面（世界系 n/d）；拟合失败传空列表。
    sp_sel：组内超点在**全场景超点数组**中的索引。
    eval_sp：(n_sp,) 布尔——评估域（GT 前景）。pred 组可能含 GT 非前景
    超点（fg 假阳性），其承载若进分子而分母只算 GT 前景，点级会反超
    GT 上限（0.668 vs 0.576 实锤）。两口径同域后唯一变量才是分组质量。
    """
    out = _new_core()
    F = len(face_table)
    M = len(assign)
    K = len(faces)
    out["n_faces_pred"] = K
    pc = pair_cnt.astype(np.float64)
    denom = np.bincount(pair_face, weights=pc, minlength=F)
    # pair 行 → 组内局部位置（-1 = 不在该组）
    lim = int(max(pair_sp.max(initial=0), sp_sel.max(initial=0))) + 1
    loc_of = np.full(lim, -1, np.int64)
    loc_of[sp_sel] = np.arange(M)
    loc = loc_of[pair_sp]
    grp = np.where(loc >= 0, assign[np.clip(loc, 0, None)], -1)
    in_grp = grp >= 0
    if eval_sp is not None:
        in_grp &= eval_sp[pair_sp]          # 只统计评估域内的承载
    # 承载矩阵 W[k, fi]（预测面 → GT 面）——只用于面检出判据
    W = np.zeros((K, F), np.float64)
    if K and in_grp.any():
        kk, pf, ppc = grp[in_grp], pair_face[in_grp], pc[in_grp]
        flat = np.bincount(kk * F + pf, weights=ppc, minlength=K * F)
        W = flat.reshape(K, F)
    f_tot = W.sum(1)
    # 面检出（双向 50%）+ 分桶：该 GT 面是否被某个预测面接住
    for fi in range(F):
        if denom[fi] <= 0 or K == 0:
            continue
        cov = W[:, fi] / denom[fi]
        pur = W[:, fi] / np.maximum(f_tot, 1e-9)
        hit = np.flatnonzero((cov >= 0.5) & (pur >= 0.5))
        if len(hit) == 0:
            continue
        n_pts_f, n_vis_f = face_table[fi, 16], face_table[fi, 17]
        if n_pts_f < MIN_FACE_PTS or n_vis_f < MIN_FACE_VIS:
            continue
        vr = n_vis_f / max(n_pts_f, 1)
        bk = "whole" if vr >= 0.8 else ("part" if vr >= 0.3 else "gap")
        out["buckets"][bk][0] += 1
        out["buckets"][bk][1] += 1
        # 诊断：取承载最大的检出面，与 GT 名义面比角/偏
        best = int(hit[np.argmax(W[hit, fi])])
        gn = face_table[fi, 3:6].astype(np.float64)
        gd = float(face_table[fi, 6])
        pn = np.asarray(faces[best]["n"], np.float64)
        pd_ = float(faces[best]["d"])
        dot = float(pn @ gn)
        if dot < 0:                    # 法向反向 → 平面参数同步翻转
            pn, pd_, dot = -pn, -pd_, -dot
        out["angles"].append(float(np.degrees(
            np.arccos(np.clip(dot, 0.0, 1.0)))))
        out["offsets"].append(abs(pd_ - gd))
    # 点级标签一致率 + 面级一对一 P/R
    lab_of_face = match_faces_to_gt(faces, face_table, one_to_one=True)
    if in_grp.any():
        lab = lab_of_face[grp[in_grp]]
        pf = pair_face[in_grp]
        ok = (lab >= 0) & (lab == pf)
        out["point_hit"] = float(pc[in_grp][ok].sum())
    # 面级：只统计有承载的预测面（无支撑面不是预测）；GT 分母在
    # 场景级只累加一次（按组累加会×组数，首版 41034 vs 1146 实锤）
    sup = f_tot > 0
    out["f_pred"] = int(sup.sum())
    out["f_tp"] = int(((lab_of_face >= 0) & sup).sum())
    out["n_members"] = 1
    return out


def scene_point_tot(pair_sp: np.ndarray, pair_cnt: np.ndarray,
                    sp_sel: np.ndarray) -> float:
    """场景级点级分母：**全部 GT 前景超点**的承载总量（与组划分无关）。

    首版用"组内超点承载"当分母，预测簇漏超点时（前景漏检/簇外超点）
    分母变小，pred 的 point_acc 反而高于 GT 上限（0.755 > 0.660 实锤）。
    固定分母后 pred ≤ GT 上限（同样的标签准则 + 已知更准的分组）。"""
    if len(sp_sel) == 0:
        return 0.0
    in_fg = np.isin(pair_sp, sp_sel)
    return float(pair_cnt[in_fg].sum())


def match_faces_to_gt(faces: list, face_table: np.ndarray,
                      ang_deg: float = LABEL_DEG,
                      off_mm: float = LABEL_MM,
                      one_to_one: bool = True) -> np.ndarray:
    """预测面 → GT 面标签匹配，(K,) 索引或 -1。

    为什么不用"承载投票"定标签：承载投票（argmax W[k,·]）是贪心的，
    且点级一致率**奖励过分割**（小簇拟合更准→点级更高；实测 GT 构件
    分组拟合 0.578 < 预测过分割簇 0.626）。平面参数匹配是面假设质量
    的直接度量：预测面若真是某个物理面，其 (n,d) 必然同时贴住该 GT
    面的法向与偏移；角部弦/错误归属线匹配不上任何 GT 面 → -1。

    one_to_one=True：按 score 升序贪心，每个 GT 面只接受一个预测面
    ——多余的预测面（过分割）计为 -1，成为面级 precision 的惩罚项。"""
    K = len(faces)
    if K == 0 or len(face_table) == 0:
        return np.full(K, -1, np.int64)
    pn = np.array([np.asarray(f["n"], np.float64) for f in faces])
    pd_ = np.array([float(f["d"]) for f in faces], np.float64)
    gn = face_table[:, 3:6].astype(np.float64)
    gd = face_table[:, 6].astype(np.float64)
    dots = pn @ gn.T                                  # (K,F) 有符号
    angs = np.degrees(np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))
    sgn = np.sign(dots)
    sgn[sgn == 0] = 1.0
    offs = np.abs(pd_[:, None] - sgn * gd[None, :])
    score = angs / ang_deg + offs / off_mm
    score[(angs > ang_deg) | (offs > off_mm)] = np.inf
    lab = np.full(K, -1, np.int64)
    F = len(face_table)
    if not one_to_one:
        j = score.argmin(1)
        ok = np.isfinite(score[np.arange(K), j])
        lab[ok] = j[ok]
        return lab
    order = np.argsort(score, axis=None)
    used = np.zeros(F, bool)
    for idx in order:
        k, fi = divmod(int(idx), F)
        if not np.isfinite(score[k, fi]):
            break
        if lab[k] >= 0 or used[fi]:
            continue
        lab[k] = fi
        used[fi] = True
    return lab


def merge_core(dst: dict, src: dict) -> None:
    for b in dst["buckets"]:
        dst["buckets"][b][0] += src["buckets"][b][0]
        dst["buckets"][b][1] += src["buckets"][b][1]
    dst["point_hit"] += src["point_hit"]
    dst["point_tot"] += src["point_tot"]
    dst["angles"] += src["angles"]
    dst["offsets"] += src["offsets"]
    dst["n_faces_pred"] += src["n_faces_pred"]
    dst["n_members"] += src["n_members"]
    for k in ("f_tp", "f_pred", "f_gt"):
        dst[k] += src[k]


def core_summary(core: dict) -> dict:
    out = {}
    for b, (a, h) in core["buckets"].items():
        out[b] = {"n": a, "hit": h, "rate": round(h / max(a, 1), 4)}
    out["n_elig"] = sum(v[0] for v in core["buckets"].values())
    fp_, fr_ = (core["f_tp"] / max(core["f_pred"], 1),
                core["f_tp"] / max(core["f_gt"], 1))
    out["face_p"] = round(fp_, 4)
    out["face_r"] = round(fr_, 4)
    out["face_f1"] = round(2 * fp_ * fr_ / max(fp_ + fr_, 1e-9), 4)
    out["face_counts"] = {"tp": core["f_tp"], "pred": core["f_pred"],
                          "gt": core["f_gt"]}
    out["point_acc"] = round(core["point_hit"] / max(core["point_tot"], 1), 4)
    out["angle_p50_p90"] = [round(float(np.percentile(core["angles"], q)), 2)
                            for q in (50, 90)] if core["angles"] else None
    out["offset_p50_p90"] = [round(float(np.percentile(core["offsets"], q)), 1)
                             for q in (50, 90)] if core["offsets"] else None
    out["n_faces_pred"] = core["n_faces_pred"]
    out["n_members"] = core["n_members"]
    return out


def fit_group(d: dict, sel: np.ndarray, rng, max_lines: int = 14):
    """对一组超点做结构拟合；返回 (assign_full, faces, ok)。

    assign_full 在**全场景超点索引域**（组外 -1）。"""
    assign_full = np.full(len(d["fg"]), -1, np.int64)
    res = fit_member(d["cent"][sel], d["nrm"][sel], d["n_pts"][sel],
                     max_lines=max_lines, rng=rng)
    if not res.get("ok"):
        return assign_full, [], res
    a = np.asarray(res["assign"])
    a = a[:len(sel)] if len(a) >= len(sel) else np.pad(
        a, (0, len(sel) - len(a)), constant_values=-1)
    assign_full[sel] = a
    return assign_full, res["faces"], res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--cache-dir", default=os.path.join(
        _PROJ, "out", "exp1_train", "cache", "scene_cache"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [it for it in load_items(args.cache_dir)
             if it.split == args.split]
    if args.max_scenes:
        items = items[:args.max_scenes]
    net = fn.SceneFaceNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device,
                                   weights_only=True)["net"])
    net.eval()
    rng = np.random.default_rng(0)

    tp = pd = rd = 0
    core_pred, core_gt = _new_core(), _new_core()
    n_fit_ok = n_fit_tot = 0
    for si, it in enumerate(items):
        d = as_eval_arrays(it)
        x = torch.from_numpy(d["feat"]).float().to(device)
        ei = torch.from_numpy(d["adj"]).t().contiguous().long().to(device)
        w = torch.from_numpy(d["w"]).float().to(device)
        _lg, _fe, memb_np, _tr = net(x, ei, w)
        # 解码用**预测前景**（部署条件）；d["fg"] 是 GT 前景，用它解码
        # 等于把真值喂进解码（首版把整面检出刷到 100% 的实锤）
        fg_prob = torch.softmax(_lg, dim=1)[:, 1].cpu().numpy()
        fg_pred = fg_prob >= 0.5
        gt = it.sp_member
        pred = decode_clusters(memb_np.cpu().numpy(), fg_pred, d["adj"],
                               args.tau)
        # pairwise（跨场景累计分子分母）
        selp = (gt >= 0) & (pred >= 0)
        if selp.any():
            ck = pred[selp].astype(np.int64) * 100000 + gt[selp]
            _u, cnt = np.unique(ck, return_counts=True)
            tp += sum(comb(int(c), 2) for c in cnt)
            pd += sum(comb(int(c), 2)
                      for c in np.bincount(pred[selp]) if c > 1)
            rd += sum(comb(int(c), 2)
                      for c in np.bincount(gt[selp]) if c > 1)
        # 预测簇逐簇拟合（只在前景内拟合；**统一到 GT 前景域**——
        # 预测前景含 fg 假阳性超点，其承载会凭空进分子而分母只算 GT
        # 前景，导致 pred 点级反超 GT 上限（0.668 vs 0.576 实锤）。
        # 两口径同域，唯一变量才是"分组质量"。）
        fg_gt = it.fg > 0
        n_pred_clusters = 0
        for c in np.unique(pred[pred >= 0]):
            sel = np.flatnonzero((pred == c) & fg_gt)
            if len(sel) == 0:
                continue
            n_pred_clusters += 1
            n_fit_tot += 1
            assign, faces, res = fit_group(d, sel, rng)
            n_fit_ok += int(res.get("ok", False))
            ok = res.get("ok", False)
            a = assign[sel] if ok else np.full(len(sel), -1, np.int64)
            merge_core(core_pred, face_eval_core(
                a, faces if ok else [], d["pair_sp"], d["pair_face"],
                d["pair_cnt"], it.face_table, sel, eval_sp=fg_gt))
        # GT 上限：GT 构件簇直接拟合（隔离聚类误差）
        for g in np.unique(gt[gt >= 0]):
            sel = np.flatnonzero(gt == g)
            assign, faces, res = fit_group(d, sel, rng)
            a = assign[sel] if res.get("ok") else np.full(len(sel), -1,
                                                          np.int64)
            merge_core(core_gt, face_eval_core(
                a, faces if res.get("ok") else [], d["pair_sp"],
                d["pair_face"], d["pair_cnt"], it.face_table, sel,
                eval_sp=fg_gt))
        # 场景级统一分母（GT 前景全部潜点的承载 + 证据充分的 GT 面数）
        fg_sel = np.flatnonzero(it.fg > 0)
        tot = scene_point_tot(d["pair_sp"], d["pair_cnt"], fg_sel)
        core_pred["point_tot"] += tot
        core_gt["point_tot"] += tot
        n_gt_elig = int(((it.face_table[:, 16] >= MIN_FACE_PTS) &
                         (it.face_table[:, 17] >= MIN_FACE_VIS)).sum())
        core_pred["f_gt"] += n_gt_elig
        core_gt["f_gt"] += n_gt_elig
        if (si + 1) % 40 == 0:
            print(f"  {si+1}/{len(items)}", flush=True)

    p = tp / max(pd, 1)
    r = tp / max(rd, 1)
    out = {
        "ckpt": args.ckpt, "tau": args.tau, "split": args.split,
        "n_scenes": len(items),
        "member_pair": {"p": round(p, 4), "r": round(r, 4),
                        "f1": round(2 * p * r / max(p + r, 1e-9), 4)},
        "fit_ok_rate": round(n_fit_ok / max(n_fit_tot, 1), 4),
        "pred": core_summary(core_pred),
        "gt_fit_upper": core_summary(core_gt),
    }
    path = args.out or os.path.join(
        os.path.dirname(args.ckpt), f"eval_struct_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {path}")


if __name__ == "__main__":
    main()
