"""
场景训练缓存构建（T-037 M2，docs/t037_scene_face_pipeline.md §4.1/§5.2/§6）。

scene npz（mock/scene/v1，全量点+visible 掩码）→ **visible 过滤**
（部署域=扫描仪只见可见点，2026-10-26 快验证实 D14 参数在可见域纯度
0.90-0.98 与全量域逐位相当）→ 超点（superpoint.build，D14 默认）→
逐超点 18 维特征 + 前景/实例标签 → 精简缓存 `out/exp1_train/cache/scene_cache/sc_*.npz`
（超点级，无体素/原始点数组——16G 内存机器全量物化预算 ~2GB，P-033）。

标签口径（§5.2）：
- 前景正类 = 混凝土面点占比 >50% 的超点（多数票，负类=地面/垫块/筋/
  箍筋/噪声合并为一桶）；
- 实例标签 face_key = face_table 行号（-1 非前景）——匹配/判型统一按
  行号索引；
- n_face = 该超点承载的多数面可见点数（§7.1 覆盖权重）。

CLI：
  python -m scripts.build_scene_cache --shard 0 --num-shards 4
  （按排序文件列表分片；多进程并行时 --kdtree-workers 传 cores//nproc）
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from lib.scene import superpoint as sp
from lib.scene.face_net import (FEAT_DIM, SL_CENT, SL_EXT, SL_NRM,
                                SL_ZQ)

CACHE_VER = 3


def _group_majority(sps: np.ndarray, keys: np.ndarray, n_sp: int):
    """逐超点 (键, 键+1 平移后) 多数票 → (多数键数组, 多数计数, 总数)。
    键可为 -1（负桶）——内部 +1 平移，0=负桶。"""
    pair = sps * (2**22) + (keys + 1)
    u, cnt = np.unique(pair, return_counts=True)
    s, b = u // (2**22), u % (2**22)
    order = np.lexsort((cnt, s))
    s, cnt, b = s[order], cnt[order], b[order]
    last = np.ones(len(s), bool)
    last[:-1] = s[1:] != s[:-1]
    total = np.bincount(s, cnt, minlength=n_sp)
    maj_key = np.full(n_sp, -1, np.int64)
    maj_cnt = np.zeros(n_sp, np.int64)
    maj_key[s[last]] = b[last] - 1
    maj_cnt[s[last]] = cnt[last]
    return maj_key, maj_cnt, total


def _grouped_axis_extent(p: np.ndarray, sps: np.ndarray, n_sp: int):
    """逐超点三轴范围（log 前的原始 mm 值）：按 (sp, 值) 排序取组首尾。"""
    ext = np.zeros((n_sp, 3), np.float32)
    for a in range(3):
        order = np.lexsort((p[:, a], sps))
        v, s = p[order, a], sps[order]
        first = np.searchsorted(s, np.arange(n_sp), "left")
        last = np.searchsorted(s, np.arange(n_sp), "right") - 1
        has = last >= first
        ext[has, a] = v[last[has]] - v[first[has]]
    return ext


def _grouped_z_quantiles(p: np.ndarray, sps: np.ndarray, n_sp: int):
    """逐超点 z 分位 p10/p90（原始 mm）：组内排序取位。"""
    order = np.lexsort((p[:, 2], sps))
    z, s = p[order, 2], sps[order]
    first = np.searchsorted(s, np.arange(n_sp), "left")
    cnt = np.searchsorted(s, np.arange(n_sp), "right") - first
    out = np.zeros((n_sp, 2), np.float32)
    for j, q in enumerate((0.10, 0.90)):
        idx = np.clip(first + (cnt * q).astype(np.int64), 0,
                      np.maximum(first + cnt - 1, 0))
        has = cnt > 0
        out[has, j] = z[idx[has]]
    return out


def superpoint_features(p: np.ndarray, res: dict) -> dict:
    """可见点 p (M,3) + 超点结果 → 逐超点 18 维特征（维度布局见
    face_net.FEAT_SLICES）。全部向量化聚合（P-052 纪律）。"""
    sp_pt = res["sp_point"].astype(np.int64)
    sp_vox = res["sp_vox"].astype(np.int64)
    n_sp = int(res["sp_npts"].size)
    pos = sp_pt >= 0
    sps = sp_pt[pos]
    n_pts = res["sp_npts"].astype(np.int64)

    bbox_lo = p.min(0).astype(np.float64)
    bbox_hi = p.max(0).astype(np.float64)
    bbox_ext = np.maximum(bbox_hi - bbox_lo, 1.0)

    cent = np.zeros((n_sp, 3), np.float64)
    for a in range(3):
        cent[:, a] = np.bincount(sps, weights=p[pos, a], minlength=n_sp)
    cent /= np.maximum(n_pts, 1)[:, None]

    feat = np.zeros((n_sp, FEAT_DIM), np.float32)
    feat[:, SL_CENT] = ((cent - bbox_lo) / bbox_ext).astype(np.float32)
    feat[:, 3] = (np.log(n_pts + 1.0) / 8.0).astype(np.float32)
    ext = _grouped_axis_extent(p[pos], sps, n_sp)
    feat[:, SL_EXT] = (np.log(ext + 1.0) / 7.0).astype(np.float32)

    # 超点 PCA（点数加权协方差 → 特征值升序）。**最小特征向量 = LS 平面
    # 法向**，进 feat[SL_NRM]——比体素法向均值好：均值被面边缘拐角体素
    # 污染（大面实例角度近失 18.8° 的归因，2026-10-26），LS 对边缘点
    # 鲁棒；实例平面估计与网络输入共用此列
    mu = cent
    cov = np.zeros((n_sp, 3, 3), np.float64)
    for a in range(3):
        for b in range(3):
            cov[:, a, b] = np.bincount(
                sps, weights=p[pos, a] * p[pos, b], minlength=n_sp)
    cov = cov / np.maximum(n_pts, 1)[:, None, None]
    cov -= np.einsum("ki,kj->kij", mu, mu)
    w_eig, v_eig = np.linalg.eigh(cov)
    w_eig = np.maximum(w_eig, 1e-9)
    feat[:, 7] = (np.log(w_eig[:, 0] / w_eig[:, 2]) / 3.0).astype(np.float32)
    feat[:, 8] = (np.log(w_eig[:, 1] / w_eig[:, 2]) / 3.0).astype(np.float32)
    n_ls = v_eig[:, :, 0].copy()
    flip = (n_ls[:, 2] < 0) | ((np.abs(n_ls[:, 2]) < 0.05)
                               & ((n_ls[:, 0] < 0)
                                  | ((np.abs(n_ls[:, 0]) < 0.05)
                                     & (n_ls[:, 1] < 0))))
    n_ls = np.where(flip[:, None], -n_ls, n_ls)
    feat[:, SL_NRM] = n_ls.astype(np.float32)

    # 体素级法向 → 离散度（SL_NRM 已存 LS 法向，此处不再覆盖）
    vm = sp_vox >= 0
    sv = sp_vox[vm]
    nrm = np.zeros((n_sp, 3), np.float64)
    for a in range(3):
        nrm[:, a] = np.bincount(sv, weights=res["vox_nrm"][vm, a],
                                minlength=n_sp)
    n_norm = np.maximum(
        np.bincount(sv, minlength=n_sp), 1)[:, None]
    nrm /= n_norm
    nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm /= np.maximum(nrm_len, 1e-9)
    dots = np.abs(np.einsum("vi,vi->v", res["vox_nrm"][vm],
                            nrm[sv]))
    ndisp = 1.0 - np.bincount(sv, weights=dots, minlength=n_sp) / n_norm[:, 0]
    feat[:, 12] = ndisp.astype(np.float32)

    feat[:, 13] = np.log(res["sp_npts"] / np.maximum(res["sp_nvox"], 1)
                         + 1e-3).astype(np.float32)
    zq = _grouped_z_quantiles(p[pos], sps, n_sp)
    feat[:, SL_ZQ] = ((zq - bbox_lo[2]) / bbox_ext[2]).astype(np.float32)

    adj = res["adj"].astype(np.int64)
    deg = np.bincount(adj[:, 0], minlength=n_sp) \
        + np.bincount(adj[:, 1], minlength=n_sp)
    feat[:, 16] = (np.log(deg + 1.0) / 5.0).astype(np.float32)
    cons = np.abs((nrm[adj[:, 0]] * nrm[adj[:, 1]]).sum(1))
    cons_sum = np.bincount(adj[:, 0], weights=cons, minlength=n_sp) \
        + np.bincount(adj[:, 1], weights=cons, minlength=n_sp)
    feat[:, 17] = (cons_sum / np.maximum(deg, 1)).astype(np.float32)

    return {"feat": feat, "cent": cent.astype(np.float32),
            "nrm": n_ls.astype(np.float32), "bbox": np.concatenate(
                [bbox_lo, bbox_hi]).astype(np.float32)}


def build_cache(npz_path: str, out_dir: str, kdtree_workers: int = -1,
                tag: str = "") -> dict:
    """单个 scene npz → 精简训练缓存。返回统计（供汇总打印）。"""
    z = np.load(npz_path, allow_pickle=True)
    pts = z["points"].astype(np.float32)
    mid, fid, cls, vis = (z["member_id"], z["face_id"], z["class_id"],
                          z["visible"])
    ft = z["face_table"]
    meta = json.loads(str(z["scene_meta_json"]))

    keep = vis.astype(bool)
    p = pts[keep]
    res = sp.build(p, workers=kdtree_workers)

    # ── 标签：face_table 行号键 ─────────────────────────────────
    row_key = (ft[:, 0].astype(np.int64) * 1000
               + ft[:, 1].astype(np.int64))          # (F,)
    order = np.argsort(row_key)
    row_sorted = row_key[order]
    mi_k = mid[keep].astype(np.int64)
    fi_k = fid[keep].astype(np.int64)
    is_face = (cls[keep] == 1) & ((fid >= 0) & (mid >= 0))[keep]
    row_of_pt = np.full(len(p), -1, np.int64)
    hit = np.flatnonzero(is_face)
    if len(hit):
        pt_key = mi_k[hit] * 1000 + fi_k[hit]
        pos = np.clip(np.searchsorted(row_sorted, pt_key), 0,
                      len(row_sorted) - 1)
        if not (row_sorted[pos] == pt_key).all():
            bad = pt_key[row_sorted[pos] != pt_key][:3]
            raise AssertionError(f"{npz_path}: 面点键未在 face_table 命中 "
                                 f"(member*1000+fid={bad.tolist()})")
        row_of_pt[hit] = order[pos]

    n_sp = int(res["sp_npts"].size)
    sps = res["sp_point"].astype(np.int64)
    valid = sps >= 0

    # 面点多数票（覆盖权重） + 全点多数票（前景判据）
    fk, nf, _ = _group_majority(sps[valid & (row_of_pt >= 0)],
                                row_of_pt[valid & (row_of_pt >= 0)], n_sp)
    _gk, _gc, total = _group_majority(
        sps[valid], np.where(row_of_pt >= 0, row_of_pt, -1)[valid], n_sp)
    fg = (fk >= 0) & (nf > 0.5 * total)
    face_key = np.where(fg, fk, -1).astype(np.int32)

    # 逐（超点，面）点数对表——§7.1 覆盖的点级精确分摊（多数面口径会
    # 丢跨面超点的少数面点，使小面零承载、检出分母偏乐观，2026-10-26
    # 试点归因后改为对表）
    m_fp = valid & (row_of_pt >= 0)
    pair_sp = sps[m_fp]
    pair_face = row_of_pt[m_fp]
    pk = pair_sp * (2**20) + pair_face
    upk, pair_cnt = np.unique(pk, return_counts=True)
    pair_sp = (upk // (2**20)).astype(np.int32)
    pair_face = (upk % (2**20)).astype(np.int32)
    pair_cnt = pair_cnt.astype(np.int32)

    feats = superpoint_features(p, res)
    con_noface = int(((cls[keep] == 1) & (fid < 0)[keep]).sum())

    # 可见域面点纯度（M1 口径平移，域健康监控）
    face_pt_total = int(is_face[valid].sum())
    pur = float(nf[fg].sum() / max(face_pt_total, 1))

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(out_dir, f"sc_{stem}.npz")
    meta_out = {
        "cache_ver": CACHE_VER, "mode": meta["mode"],
        "config_id": meta.get("config_id"), "split": meta.get("split"),
        "domain": "d1" if meta["mode"] == "d1" else "d2",
        "n_points_vis": int(len(p)), "n_points_full": int(len(pts)),
        "n_sp": n_sp, "n_faces": int(len(ft)),
        "fg_share": float(fg.mean()), "purity_vis": round(pur, 4),
        "noise_pt_share": float((~valid).mean()),
        "concrete_noface_pts": con_noface,
        "sp_params": {"knn": sp.KNN, "ang": sp.ANG_THR_DEG,
                      "plan": sp.PLAN_THR, "min_vox": sp.MIN_VOX},
        "src": os.path.relpath(npz_path),
    }
    np.savez_compressed(
        out_path,
        feat=feats["feat"].astype(np.float32),
        cent=feats["cent"], nrm=feats["nrm"], bbox=feats["bbox"],
        adj=res["adj"].astype(np.int32),
        adj_w=res["adj_w"].astype(np.int32),
        fg=fg.astype(np.uint8), face_key=face_key,
        n_face=nf.astype(np.int32),
        n_pts=res["sp_npts"].astype(np.int32),
        pair_sp=pair_sp, pair_face=pair_face, pair_cnt=pair_cnt,
        face_table=ft, meta_json=json.dumps(meta_out, ensure_ascii=False))
    return meta_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-root", default=None,
                    help="scene npz 根目录（默认 mock/scene/v1）")
    ap.add_argument("--out-dir", default=None,
                    help="缓存目录（默认 out/exp1_train/cache/scene_cache）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--kdtree-workers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="调试：只建前 N 个")
    args = ap.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    root = args.scene_root or os.path.join(proj, "mock", "scene", "v1")
    out_dir = args.out_dir or os.path.join(
        proj, "out", "exp1_train", "cache", "scene_cache")
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for mode in sorted(os.listdir(root)):
        mdir = os.path.join(root, mode)
        if not os.path.isdir(mdir):
            continue
        for f in os.listdir(mdir):
            if f.endswith(".npz"):
                files.append(os.path.join(mdir, f))
    files.sort()
    files = files[args.shard::args.num_shards]
    if args.limit:
        files = files[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(files)} scenes "
          f"→ {out_dir}", flush=True)

    t0 = time.time()
    stats = []
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"sc_{stem}.npz")
        if os.path.exists(out_path):
            continue                     # 幂等续跑
        try:
            s = build_cache(f, out_dir, kdtree_workers=args.kdtree_workers)
            stats.append(s)
            print(f"[{i+1}/{len(files)}] {stem}: sp={s['n_sp']} "
                  f"fg={s['fg_share']:.2f} 纯度={s['purity_vis']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {stem}: FAIL {e!r}", flush=True)
            raise
    if stats:
        pur = [s["purity_vis"] for s in stats]
        print(f"shard 完成 {len(stats)} scenes，"
              f"纯度 med={np.median(pur):.3f} min={min(pur):.3f}，"
              f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
