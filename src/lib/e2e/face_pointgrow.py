# -*- coding: utf-8 -*-
"""点级生长连接面提取（T-037 §7 新架构的面片来源，2026-10-27 验证晋升）。

管线（docs/handover/t037_m3_progress.md §7.5 验证过的配方）：
oracle 材质分割的混凝土点 → 15mm 体素 → kNN-PCA 法向（k=30，点数加权）
→ kNN 法向定向传播 → 区域生长（带符号法向门 + 动态重拟合参考平面 +
种子锚 2δ，不用 RANSAC）→ 共面邻接合并 → 端面/顶面挑片。

四个结构性机制（实锤见交接文档 §7.5，改动前必读）：
1. 法向符号只能靠 kNN 定向传播——z+ 规范化的 tie-break 在 |n_z|≈0.05
   边界对 ±v 不对称，实测同一 GT 面内 12% 邻接对反号；
2. 法向门不能带 abs()——薄特征正反面会融成一片（实测 89° 垃圾参数片）；
3. 动态重拟合平面的 δ 门必须配种子平面锚（2δ）——否则沿法向单调蠕行
   （实测吞并 104mm 外另一构件的平行端面）；
4. 外向定向用"背离混凝土实体"（面片两侧 0-400mm 薄层、横向限片内的
   混凝土点数，实体侧必多）——扫描位遮挡测试在薄板上失效（sleeve 4%）。

构件级接口：extract_faces(质心系点, 材质标签) → 与 e2e_fullchain
oracle_faces 同构的 {"rebar"/"sleeve": {n3,d,c3}, "top": {"d"}, "axis"}；
另带 "stface"（箍筋面候选=钢点薄层计数最大的侧面/顶面，含 n3/d/mu/n_st/
kind——滚转构件的箍筋面是竖直侧面，调用方按它滚动构件系，T-040 ②）。
提取失败返回 None（调用方按 face_fail 计）。
"""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

VOXEL = 15.0
K_NORM = 30
K_NBR = 10
GROW_THETA = 20.0
GROW_DELTA = 30.0
THETA_CRAWL = 35.0
MIN_PATCH_VOX = 20
MIN_CONCRETE_PTS = 5000
SLAB_REBAR = (0.0, 300.0)     # 出筋判定薄层（外向，§7.3① 口径）
SLAB_REBAR_THR = 20
BAND_MM = 60.0                # c3 面带（|t|<BAND_MM，与 oracle 口径一致）
SLAB_STIRRUP = (-60.0, 300.0)  # 箍筋钢点薄层（面内 −60 起——保护层下的
                               # 贴面腿；到面外 300——叠合梁箍筋凸出
                               # 混凝土面之上，s6 m1 实锤只数面内会全灭）


# ── 体素化 + 法向 + 邻接 ────────────────────────────────────────────
def voxelize(p, voxel):
    key = np.floor(p / voxel).astype(np.int64)
    key1 = (((key[:, 0] + 4096) << 42) | ((key[:, 1] + 4096) << 21)
            | (key[:, 2] + 4096))
    _u, inv = np.unique(key1, return_inverse=True)
    nv = int(inv.max()) + 1
    cent = np.zeros((nv, 3))
    for a in range(3):
        cent[:, a] = np.bincount(inv, weights=p[:, a], minlength=nv)
    npts = np.bincount(inv, minlength=nv).astype(np.float64)
    cent /= npts[:, None]
    return {"cent": cent, "npts": npts, "inv": inv, "nv": nv}


def voxel_normals(cent, npts, k=K_NORM, chunk=200000):
    """kNN-PCA 法向（LS 平面法向），点数加权，返回原始符号。"""
    tree = cKDTree(cent)
    kk = min(k + 1, len(cent))
    n = np.empty((len(cent), 3))
    for s in range(0, len(cent), chunk):
        e = min(s + chunk, len(cent))
        _d, nn = tree.query(cent[s:e], k=kk, workers=-1)
        nb = cent[nn]
        w = npts[nn][:, :, None]
        sw = w.sum(1)
        mu = (nb * w).sum(1) / sw
        x = (nb - mu[:, None, :]) * w
        cov = np.einsum("nki,nkj->nij", x, x) / sw.reshape(-1, 1, 1)
        _ev, evec = np.linalg.eigh(cov)
        n[s:e] = evec[:, :, 0]
    return n


def orient_normals(nrm, indptr, dst):
    """法向定向传播：BFS 沿 kNN 图，每点翻转到与其首次到达的父节点一致。"""
    nv = len(nrm)
    out = nrm.copy()
    oriented = np.zeros(nv, bool)
    for seed in range(nv):
        if oriented[seed]:
            continue
        oriented[seed] = True
        frontier = np.array([seed])
        while len(frontier):
            counts = indptr[frontier + 1] - indptr[frontier]
            total = int(counts.sum())
            if total == 0:
                break
            offs = np.cumsum(counts) - counts
            idx = (np.repeat(indptr[frontier], counts)
                   + np.arange(total) - np.repeat(offs, counts))
            parent = np.repeat(frontier, counts)
            cand = dst[idx]
            fresh = ~oriented[cand]
            parent, cand = parent[fresh], cand[fresh]
            if len(cand) == 0:
                break
            o = np.argsort(cand, kind="stable")
            cand_s = cand[o]
            first = np.ones(len(cand_s), bool)
            first[1:] = cand_s[1:] != cand_s[:-1]
            parent, cand = parent[o][first], cand_s[first]
            flip = np.einsum("ij,ij->i", out[parent], out[cand]) < 0
            out[cand[flip]] = -out[cand[flip]]
            oriented[cand] = True
            frontier = cand
    return out


def build_neighbors(cent, k=K_NBR):
    tree = cKDTree(cent)
    kk = min(k + 1, len(cent))
    _d, nn = tree.query(cent, k=kk, workers=-1)
    src = np.repeat(np.arange(len(cent)), kk - 1)
    dst = nn[:, 1:].reshape(-1)
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    indptr = np.zeros(len(cent) + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=len(cent)), out=indptr[1:])
    return indptr, dst


def fit_plane_w(cent, w):
    """加权 PCA。返回 (法向, 质心, 特征值升序)。"""
    sw = w.sum()
    if sw <= 0:
        return None, None, None
    mu = (cent * w[:, None]).sum(0) / sw
    x = cent - mu
    cov = (x * w[:, None]).T @ x / sw
    ev, vec = np.linalg.eigh(cov)
    return vec[:, 0], mu, ev


# ── 区域生长（带符号门 + 动态重拟合 + 种子锚）────────────────────────
def grow(cent, npts, nrm, indptr, dst, theta_deg=GROW_THETA,
         delta_mm=GROW_DELTA, max_iter=200):
    nv = len(cent)
    lab = np.full(nv, -1, np.int64)
    used = np.zeros(nv, bool)
    cos_in = np.cos(np.radians(theta_deg))
    cos_cr = np.cos(np.radians(THETA_CRAWL))
    order = np.argsort(-npts)
    pid = 0
    for seed in order:
        if lab[seed] >= 0:
            continue
        n_seed = nrm[seed].copy()
        d_seed = float(n_seed @ cent[seed])
        anchor = 2.0 * delta_mm
        lab[seed] = pid
        used[seed] = True
        mm = np.array([seed])
        frontier = mm
        n_ref, d_ref = n_seed, d_seed
        for _ in range(max_iter):
            counts = indptr[frontier + 1] - indptr[frontier]
            total = int(counts.sum())
            if total == 0:
                break
            offs = np.cumsum(counts) - counts
            idx = (np.repeat(indptr[frontier], counts)
                   + np.arange(total) - np.repeat(offs, counts))
            cand = dst[idx]
            cand = cand[~used[cand]]
            if len(cand) == 0:
                break
            cand = np.unique(cand)
            ok = ((nrm[cand] @ n_ref) > cos_in) \
                & ((nrm[cand] @ n_seed) > cos_cr) \
                & (np.abs(cent[cand] @ n_ref - d_ref) <= delta_mm) \
                & (np.abs(cent[cand] @ n_seed - d_seed) <= anchor)
            take = cand[ok]
            if len(take) == 0:
                break
            lab[take] = pid
            used[take] = True
            mm = np.concatenate([mm, take])
            w = npts[mm]
            sw = w.sum()
            mu = (cent[mm] * w[:, None]).sum(0) / sw
            x = cent[mm] - mu
            covm = (x * w[:, None]).T @ x / sw
            _ev, vec = np.linalg.eigh(covm)
            n2 = vec[:, 0]
            if n2 @ n_seed < 0:
                n2 = -n2
            n_ref, d_ref = n2, float(n2 @ mu)
            frontier = take
        pid += 1
    return lab


def merge_coplanar(lab, patch_planes, cent, indptr, dst,
                   ang_thr=3.0, dist_mm=20.0):
    """共面邻接面片合并（修同面拆片）。patch_planes: {pid: (n, d)}。"""
    if len(patch_planes) < 2:
        return None
    nv = len(cent)
    clean = np.zeros(int(lab.max()) + 1, bool)
    clean[np.fromiter(patch_planes.keys(), np.int64,
                      len(patch_planes))] = True
    src = np.repeat(np.arange(nv), indptr[1:] - indptr[:-1])
    m = clean[lab[src]] & clean[lab[dst]] & (lab[src] != lab[dst])
    if not m.any():
        return None
    a, b = lab[src[m]], lab[dst[m]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(lo * (2 ** 24) + hi)
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    bnd = np.searchsorted(ls, np.arange(int(lab.max()) + 2))
    mems = {pid: order[bnd[pid]:bnd[pid + 1]] for pid in patch_planes}
    planes = {pid: (patch_planes[pid][0], patch_planes[pid][1],
                    len(mems[pid])) for pid in patch_planes}
    parent = {pid: pid for pid in patch_planes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cos_t = np.cos(np.radians(ang_thr))
    n_merged = 0
    for k in pairs:
        pa, pb = int(k // (2 ** 24)), int(k % (2 ** 24))
        ra, rb = find(pa), find(pb)
        if ra == rb:
            continue
        na, da, wa = planes[ra]
        nb, db, wb = planes[rb]
        if na @ nb < 0:
            nb, db = -nb, -db
        if float(na @ nb) <= cos_t:
            continue
        cb = cent[mems[rb]]
        ca = cent[mems[ra]]
        if (np.percentile(np.abs(cb @ na - da), 50) > dist_mm
                or np.percentile(np.abs(ca @ nb - db), 50) > dist_mm):
            continue
        if wa >= wb:
            ra_, rb_ = ra, rb
        else:
            ra_, rb_ = rb, ra
        mu_all = np.concatenate([mems[ra_], mems[rb_]])
        n2, mu2, _ = fit_plane_w(cent[mu_all], np.ones(len(mu_all)))
        parent[rb_] = ra_
        planes[ra_] = (n2, float(n2 @ mu2), wa + wb)
        n_merged += 1
    if n_merged == 0:
        return None
    remap = np.full(int(lab.max()) + 1, -1, np.int64)
    for new, r in enumerate(sorted({find(x) for x in parent})):
        remap[r] = new
    for pid in parent:
        remap[pid] = remap[find(pid)]
    lab2 = remap[np.clip(lab, 0, None)]
    lab2[lab < 0] = -1
    return lab2


# ── 构件级面提取 ────────────────────────────────────────────────────
def _outward(n, mu, con, r_eff):
    """背离混凝土实体：两侧 0-400mm 薄层（横向限 r_eff 内）混凝土点数。"""
    rel = con - mu
    t = rel @ n
    lat = np.linalg.norm(rel - np.outer(t, n), axis=1)
    near = (lat < r_eff) & (np.abs(t) > 5.0)
    c_pos = int((near & (t > 0) & (t < 400.0)).sum())
    c_neg = int((near & (t < 0) & (t > -400.0)).sum())
    return -n if c_pos > c_neg else n


def _slab_count(q, n_out, mu, r_eff, t_lo, t_hi):
    """点集 q 在外向薄层 (t_lo,t_hi) 内、横向 r_eff 内的点数。"""
    if len(q) == 0:
        return 0
    rel = q - mu
    t = rel @ n_out
    lat = np.linalg.norm(rel - np.outer(t, n_out), axis=1)
    return int(((t > t_lo) & (t < t_hi) & (lat < r_eff)).sum())


def extract_faces(p, lbls, axis_hint=(1.0, 0.0, 0.0)):
    """质心系构件点 + 材质标签 → 连接面 dict（失败返回 None）。

    材质约定（mock/inuse）：1=混凝土 2=钢筋 3=箍筋。"""
    con_m = lbls == 1
    if int(con_m.sum()) < MIN_CONCRETE_PTS:
        return None
    con = p[con_m]
    reb = p[lbls == 2]
    v = voxelize(con, VOXEL)
    indptr, dst = build_neighbors(v["cent"], k=K_NBR)
    nrm = orient_normals(voxel_normals(v["cent"], v["npts"], k=K_NORM),
                         indptr, dst)
    lab = grow(v["cent"], v["npts"], nrm, indptr, dst)
    # 逐片平面（先给合并用）
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    n_patches = int(lab.max()) + 1
    bnd = np.searchsorted(ls, np.arange(n_patches + 1))
    planes = {}
    for pid in range(n_patches):
        mem = order[bnd[pid]:bnd[pid + 1]]
        if len(mem) < MIN_PATCH_VOX:
            continue
        n, mu, ev = fit_plane_w(v["cent"][mem], v["npts"][mem])
        if n is None or ev is None:
            continue
        if ev[1] < 3.0 * max(ev[0], 1e-9) or np.sqrt(ev[1]) < 20.0:
            continue
        planes[pid] = (n, float(n @ mu))
    lab2 = merge_coplanar(lab, planes, v["cent"], indptr, dst)
    if lab2 is not None:
        lab = lab2
        order = np.argsort(lab, kind="stable")
        ls = lab[order]
        n_patches = int(lab.max()) + 1
        bnd = np.searchsorted(ls, np.arange(n_patches + 1))

    ax = np.asarray(axis_hint, np.float64)
    ends = []
    top = None
    st = p[lbls == 3]
    st_best = None            # 箍筋面候选（钢点薄层计数最大者）
    top_n_st = None           # 重力顶面候选的钢点计数（滚转保守门用）
    for pid in range(n_patches):
        mem = order[bnd[pid]:bnd[pid + 1]]
        if len(mem) < MIN_PATCH_VOX:
            continue
        n, mu, ev = fit_plane_w(v["cent"][mem], v["npts"][mem])
        if n is None or ev is None:
            continue
        if ev[1] < 3.0 * max(ev[0], 1e-9) or np.sqrt(ev[1]) < 20.0:
            continue
        r_eff = float(np.percentile(
            np.linalg.norm(v["cent"][mem] - mu, axis=1), 95)) + 40.0
        n_out = _outward(n, mu, con, r_eff)
        if abs(float(n_out @ ax)) > 0.6:          # 端面（⊥轴）
            # 检测头契约要求法向=构件轴（训练视图全部轴对齐）。拟合
            # 法向带 2-3° 翘曲倾斜时投影基混入轴向泄漏，每根筋按外露
            # 长度不同偏 20-60mm（2026-10-27 全链诊断 C 组 112.9mm 的
            # 根因）——法向 snap 到轴，倾斜信息保留在 d 的重投影里。
            n_out = ax * (1.0 if float(n_out @ ax) > 0 else -1.0)
            cands = v["cent"][mem]
            t = (cands - mu) @ n_out
            band = cands[np.abs(t) < BAND_MM]
            c3 = (band.mean(0) if len(band) >= 10 else mu)
            d = float(n_out @ c3)
            n_rb = _slab_count(reb, n_out, c3, r_eff, *SLAB_REBAR)
            ends.append({"n3": n_out, "d": d, "c3": c3, "n_rb": n_rb})
        else:                                     # 侧面/顶面（箍筋面候选）
            n_st = _slab_count(st, n_out, mu, r_eff, *SLAB_STIRRUP)
            cand = {"n3": n_out, "d": float(n_out @ mu), "mu": mu.copy(),
                    "n_st": n_st,
                    "kind": "top" if n_out[2] > 0.9 else "side"}
            if cand["kind"] == "top" and (top is None or cand["d"] > top["d"]):
                top = {"d": cand["d"]}
                top_n_st = n_st
            # 择面：钢点计数最大；平票偏向重力顶面（正立构件零改动）
            if (st_best is None
                    or (n_st, cand["kind"] == "top")
                    > (st_best["n_st"], st_best["kind"] == "top")):
                st_best = cand
    if st_best is not None:
        st_best["top_n_st"] = top_n_st
    if len(ends) < 2:
        return None
    ends.sort(key=lambda e: -e["n_rb"])
    rebar = ends[0]
    opp = [e for e in ends[1:] if float(e["n3"] @ rebar["n3"]) < 0]
    sleeve = opp[0] if opp else ends[1]      # 出筋端取薄层钢筋最多者，
    return {"rebar": {k: rebar[k] for k in ("n3", "d", "c3")},
            "sleeve": {k: sleeve[k] for k in ("n3", "d", "c3")},
            "top": top,
            "stface": st_best,
            "axis": rebar["n3"]}
