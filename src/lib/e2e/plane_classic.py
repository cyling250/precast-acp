# -*- coding: utf-8 -*-
"""经典法向聚类平面提取（用户 2026-10-24 定案的 A 阶段，无 RANSAC）。

单趟确定性流程（无随机采样、无迭代阈值、无"识别先后"）：
  体素降采样 → 每个体素心 kNN 邻域 PCA 法向 → 平面度过滤（空洞/蜂窝/
  粗糙点曲率大，天然被丢弃）→ 法向+偏移粗桶聚类 → 逐簇重拟合（中位数
  抗离群）→ 相邻簇合并 → 尺寸过滤。
大面小面（含墙厚棱边面）同趟全部产出；混凝土粗糙度/脱模斜度不敏感
（斜面仍是平面，按真实法向提取）；低频翘曲由粗桶+合并吸收。

需要人定的只有物理常数：vox≈点距×10、kNN 邻域数、最小面尺寸
（取箍筋间距量级 250mm——比它小的结构不是"面"）。
"""

import numpy as np


def extract_planes(pts, vox=40.0, knn=24, planar_thr=0.12,
                   n_bucket=0.25, d_bucket=120.0, min_n=120,
                   min_extent=250.0, max_cents=40000):
    """返回 [{n, c, d, n_pts, extent}]——n 单位法向（符号规范化）、
    c 面上一点（成员质心）、d=p·n、extent 成员 bbox 对角。"""
    from scipy.spatial import cKDTree
    p = pts[::max(len(pts) // 300000, 1)].astype(np.float64)
    key = np.floor(p / vox).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    cents = np.zeros((len(uk), 3))
    np.add.at(cents, inv, p)
    cents /= cnt[:, None]
    if len(cents) > max_cents:
        cents = cents[:: len(cents) // max_cents + 1]

    tree = cKDTree(cents)
    _, ix = tree.query(cents, k=knn)
    nb = cents[ix]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    pl = w[:, 0] / (w[:, 2] + 1e-9)
    m = pl < planar_thr
    n, c = nrm[m], cents[m]
    if len(n) == 0:
        return []

    flip = (n[:, 2] < 0) | ((np.abs(n[:, 2]) < 0.05)
                            & ((n[:, 0] < 0) | ((np.abs(n[:, 0]) < 0.05)
                                                & (n[:, 1] < 0))))
    n = np.where(flip[:, None], -n, n)
    d = (c * n).sum(1)

    key2 = np.column_stack([np.floor(n / n_bucket), np.floor(d / d_bucket)])
    uk2, inv2 = np.unique(key2, axis=0, return_inverse=True)
    planes = []
    for g in range(len(uk2)):
        idx = np.where(inv2 == g)[0]
        if len(idx) < min_n // 2:
            continue
        ng = n[idx].mean(0)
        ng /= np.linalg.norm(ng)
        planes.append({"n": ng, "c": c[idx].mean(0),
                       "d": float(np.median((c[idx] * ng).sum(1))),
                       "n_pts": len(idx), "c_mem": c[idx]})
    # 合并：法向 <10° 且 |Δd| < 1.5 桶宽
    merged = True
    while merged and len(planes) > 1:
        merged = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                ang = np.degrees(np.arccos(np.clip(a["n"] @ b["n"], -1, 1)))
                if ang < 10.0 and abs(a["d"] - b["d"]) < d_bucket * 1.5:
                    cm = np.vstack([a["c_mem"], b["c_mem"]])
                    ng = (a["n"] * a["n_pts"] + b["n"] * b["n_pts"])
                    ng /= np.linalg.norm(ng)
                    planes[i] = {"n": ng, "c": cm.mean(0),
                                 "d": float(np.median((cm * ng).sum(1))),
                                 "n_pts": a["n_pts"] + b["n_pts"],
                                 "c_mem": cm}
                    planes.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for pl_ in planes:
        if pl_["n_pts"] < min_n:
            continue
        ext = float(np.linalg.norm(pl_["c_mem"].max(0) - pl_["c_mem"].min(0)))
        if ext < min_extent:
            continue
        pl_.pop("c_mem")
        pl_["extent"] = ext
        out.append(pl_)
    return out


def connection_faces(planes, centroid, nz_max=0.35):
    """连接面候选：近竖直法向的面（端面/棱边面），按离质心远近排序。

    外向 = 面心相对构件质心的水平方向（与 world v5 的导出约定一致）。
    """
    cands = []
    for p in planes:
        if abs(p["n"][2]) > nz_max:
            continue
        c = p["c"] - centroid
        r = np.hypot(c[0], c[1])
        if r < 1e-3:
            continue
        n2 = np.array([c[0], c[1]]) / r  # 外法向（导出，非 p.n——符号自由）
        d = float(n2 @ np.array([p["c"][0] - centroid[0],
                                 p["c"][1] - centroid[1]]))
        cands.append({"n2": n2, "d": d, "cz": float(p["c"][2] - centroid[2]),
                      "c3": p["c"] - centroid,
                      "u_ax": np.array([-n2[1], n2[0], 0.0]),
                      "extent": p["extent"], "n_pts": p["n_pts"]})
    cands.sort(key=lambda f: -f["extent"])
    return cands


def find_teeth_and_faces(pts, vox=50.0, erode_n=6, min_len=300.0,
                         max_len=1700.0, max_vox=60):
    """形态学找齿 → 出筋面（方向无关：楼梯斜面/墙棱边/柱端面同构）。

    单趟确定性：体素占据 → 26 邻域腐蚀（薄结构消失、主体存活）→
    齿体素=占据−腐蚀 且与主体相邻 → 齿簇（26 连通）→ 形状过滤
    （细长、长 300–1700mm、体素数少——地面薄壳/背景杂物由此排除）→
    每齿方向=PCA 主轴（尖端=离主体远端）、根部=贴主体端 →
    出筋面法向=齿向量的平均（齿 ⊥ 面），面偏移=根部点·法向的中位数。
    返回 (teeth, faces)：teeth=[{root, tip, dir, len}]，faces=[{n3, d,
    c3}]（c3=根部质心，世界坐标）。
    """
    p = pts.astype(np.float64)
    key = np.floor(p / vox).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    occ = {tuple(k): i for i, k in enumerate(uk)}  # 体素键 → 唯一索引
    idx_of = {tuple(k): i for i, k in enumerate(uk)}
    occ_set = set(idx_of)
    # 邻居数（26 邻域）——以整数键平移实现
    offs = np.array([[i, j, l] for i in (-1, 0, 1) for j in (-1, 0, 1)
                     for l in (-1, 0, 1) if (i, j, l) != (0, 0, 0)])
    keys_arr = uk
    nb_cnt = np.zeros(len(uk), np.int32)
    for o in offs:
        shifted = keys_arr + o
        # 向量化查存在性：转 structured 视图
        sview = np.ascontiguousarray(shifted).view(
            [("a", np.int64), ("b", np.int64), ("c", np.int64)]).ravel()
        kview = np.ascontiguousarray(uk).view(
            [("a", np.int64), ("b", np.int64), ("c", np.int64)]).ravel()
        lookup = np.isin(sview, kview)
        nb_cnt += lookup
    eroded = nb_cnt >= erode_n
    # 齿体素：占据且未腐蚀
    tooth_mask = ~eroded
    tooth_idx = np.where(tooth_mask)[0]
    if len(tooth_idx) == 0:
        return [], []
    # 注：不做"一跳邻接主体"过滤——齿链中段体素只邻接其他齿体素，会被
    # 误杀（实测 0 齿根因）；杂物排除改由形状过滤承担（细长 + 长度 +
    # 体素数上限：地面薄壳 σ1/σ2≈1.6、杂物簇尺寸均被拒）。
    # 26 连通聚类（齿体素间）
    tset = set(tooth_idx.tolist())
    seen = set()
    clusters = []
    for i in tooth_idx:
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for o in offs:
                v = tuple(uk[u] + o)
                j = idx_of.get(v)
                if j is not None and j in tset and j not in seen:
                    seen.add(j)
                    stack.append(j)
        clusters.append(np.array(comp))
    # 形状过滤 + 齿参数
    body_center = uk[eroded].mean(0) * vox if eroded.any() else uk.mean(0) * vox
    teeth = []
    for comp in clusters:
        if not (2 <= len(comp) <= max_vox):
            continue
        pts_c = (uk[comp] + 0.5) * vox
        c = pts_c.mean(0)
        _, sv, V = np.linalg.svd(pts_c - c, full_matrices=False)
        if sv[1] < 1e-6 or sv[0] / sv[1] < 3.0:  # 不够细长
            continue
        ax = V[0]
        t = (pts_c - c) @ ax
        ln = float(t.max() - t.min())
        if ln < min_len or ln > max_len:
            continue
        # 尖端=离主体质心远的一端；根=近端
        if (c + ax * t.max()) @ (c - body_center) > \
                (c + ax * t.min()) @ (c - body_center):
            tip_t, root_t = t.max(), t.min()
        else:
            tip_t, root_t = t.min(), t.max()
        sgn = 1.0 if tip_t > root_t else -1.0
        teeth.append({"root": c + ax * root_t, "tip": c + ax * tip_t,
                      "dir": ax * sgn, "len": ln})
    if not teeth:
        return [], []
    # 精拟合（粗选择+精拟合，用户 2026-10-24 提案的经典实现）：法向保留
    # 齿向平均（实测 1.2°，SVD 重拟会被面带内地面条带带歪）；偏移 d 用
    # 面带点（滤地面）沿齿向投影的中位数重估——根部体素在面外 1-2 体素
    # 的系统偏移由此消除。
    n3 = np.sum([t["dir"] for t in teeth], 0)
    n3 /= np.linalg.norm(n3)
    roots = np.array([t["root"] for t in teeth])
    d0 = float(np.median(roots @ n3))
    near = pts.astype(np.float64)
    ground = np.percentile(near[:, 2], 5.0)
    near = near[near[:, 2] > ground + 60.0]
    near = near[np.linalg.norm(near[:, :2] - roots.mean(0)[:2], axis=1)
                < 800.0]
    m = np.abs(near @ n3 - d0) < 150.0
    if m.sum() >= 50:
        d0 = float(np.median(near[m] @ n3))
    faces = [{"n3": n3, "d": d0, "c3": roots.mean(0),
              "n_teeth": len(teeth)}]
    return teeth, faces


def find_end_faces(pts, cell=40.0, ground_gap=80.0, band=80.0):
    """足迹边界法定位两端面（水平端面构件的快速路径；楼梯等斜出筋面
    请用 find_teeth_and_faces——方向无关）。

    流程（确定性）：去地面 → 粗格足迹占据 → 足迹 PCA=构件轴（任意摆放
    通吃）→ 沿轴两极端带的面心=端面心 → 外法向=±轴。出筋端由足迹单侧
    延伸出稀疏梳齿判定（无梳齿时两端等价返回，端别交给检测头）。
    返回 (faces, axis2)：faces 各含 c3/n2/d/cz/u_ax（世界质心系）。
    """
    p = pts.astype(np.float64)
    ground = np.percentile(p[:, 2], 5.0)
    m = p[:, 2] > ground + ground_gap  # 去地面（含低矮背景）
    q = p[m]
    gx = np.floor(q[:, :2] / cell).astype(np.int64)
    uk = np.unique(gx, axis=0)
    cc = (uk + 0.5) * cell  # 占据格中心
    ctr = cc.mean(0)
    _, _, V = np.linalg.svd(cc - ctr, full_matrices=False)
    a2 = V[0]  # 构件轴（水平分量主导——构件躺放）
    a2 = np.array([a2[0], a2[1]])
    a2 /= np.linalg.norm(a2)
    t = (cc - ctr) @ a2
    lo, hi = np.percentile(t, [1.0, 99.0])  # 主体范围（抗离群格）
    faces = []
    for sign in (-1.0, 1.0):
        edge = t if sign > 0 else -t
        mband = edge > (hi if sign > 0 else -lo) - band
        cells = cc[mband]
        c2 = cells.mean(0)
        r = float(np.hypot(*(c2 - ctr)))
        n2 = (c2 - ctr) / r * sign  # 外向
        # 面带内真实点的 z 中位 → cz
        rel = q - np.array([ctr[0], ctr[1], 0.0])
        tt = rel[:, :2] @ a2
        mm = (tt > (hi - 150.0)) if sign > 0 else (tt < (lo + 150.0))
        cz = float(np.median(q[mm, 2])) if mm.sum() > 20 else float(
            np.median(q[:, 2]))
        d = float(n2 @ c2)
        faces.append({"n2": n2, "d": d, "c3": np.array([c2[0], c2[1], cz]),
                      "cz": cz - float(np.median(q[:, 2])),
                      "u_ax": np.array([-n2[1], n2[0], 0.0])})
    return faces, a2
