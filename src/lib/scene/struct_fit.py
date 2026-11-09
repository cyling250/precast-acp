"""
逐构件结构拟合（T-037 M3 段③，用户定稿的"结构先验"路线）。

输入：单构件的前景超点（质心 mm / LS 法向 / 点数权重）——来自段②的
构件嵌入聚类。输出：该构件的棱柱面假设（2 端面 + N 侧面，各带外向
法向/偏移/支撑），供判型头与检测头按面工作。

先验边界（用户 2026-10-27 拍板）：
- 当先验：棱柱拓扑（多边形截面×拉伸）、轴向近水平、端面⊥轴；
- 不当先验：一切尺寸（截面大小、边数、长度、位置）——边数只作
  RANSAC 的**上限**（来自类型档头 C），下限边长（80mm）是有效性
  约束（排除噪声碎片），均不预设构件形状。

几何配方（每步的鲁棒性依据）：
- 轴 = 质心加权 PCA 主方向 + 端面法向簇精化（翘曲 σ12-40mm 在 2-9m
  长基线上平均掉；重力先验作质量门不作约束）；
- 截面 = 侧面超点投影到 ⊥轴 平面，迭代加权线段 RANSAC（band 12mm，
  边长 ≥80mm 有效性约束，边数 ≤ 类型档上限）；
- 线定向 = "全部其余边界点在单侧"（加权中位数投票），凹多边形安全；
- 多边形 = 定向线按角度排序后相邻求交，面积/自交校验；
- 归属 = 超点质心取【侧面 2D 线段距 vs 端面轴向距】的 argmin（角点
  天然二分——寄宿问题的几何解）；
- 质量门不过 → ok=False，调用方回退 M2 实例路径。
"""

import numpy as np

BAND_MM = 12.0        # 线 RANSAC 内点带（8mm 体素+噪声+截面内翘曲）
MIN_EDGE_MM = 80.0    # 有效边最短长度（有效性约束，非尺寸先验）
MIN_LINE_W_FRAC = 0.005   # 装配前线支撑门（占构件权重比）——真实场景
                        # 杂散点上 RANSAC 会抠出 W~0.2% 的碎片线搅烂
                        # 边图（2026-10-27 真实缓存 99% fail 的解剖实锤）
MIN_SIDE_W_FRAC = 0.005   # 侧面最小支撑（占构件权重比）
COV_MIN = 0.85        # 归属覆盖门
AXIS_Z_MAX = 0.35     # 轴向重力门（|axis·z|）


def _wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x * w[:, None]).sum(0) / max(w.sum(), 1e-9)


def estimate_axis(cent: np.ndarray, nrm: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
    """轴向：粗轴（质心 PCA 主方向，仅用于分类）→ **侧面法向协方差
    最小特征向量**。为什么不用 PCA/端法向均值定轴：轴倾斜 θ 会让长
    L 构件投影到截面平面时端部拖尾 sin θ·L（3000mm×4.5°≈236mm），
    截面边被涂成 smear、RANSAC 找出一堆平行弦（金样例实锤）；侧面
    法向恒 ⊥ 真轴，其对轴的约束能力与构件长度无关。法向 z+ 规范化
    的符号无关性由 nnᵀ 协方差天然保证。"""
    mu = _wmean(cent, w)
    x = cent - mu
    cov = (x * w[:, None]).T @ x / w.sum()
    _, vec = np.linalg.eigh(cov)
    axis = vec[:, -1]
    for _ in range(2):                    # 两轮：分类→法向协方差→再分类
        dots = nrm @ axis
        side = np.abs(dots) <= 0.7
        if side.sum() >= 8:
            ns = nrm[side] * w[side, None]
            M = ns.T @ ns
            _, v2 = np.linalg.eigh(M)
            cand = v2[:, 0]
            if float(cand @ axis) < 0:
                cand = -cand
            axis = cand
    if axis[2] < 0:                       # 重力先验：轴取近水平定向
        axis = -axis
    return axis


def _line_ransac(p2: np.ndarray, w: np.ndarray,
                 n2d: np.ndarray = None, band: float = BAND_MM,
                 iters: int = 400, rng: np.random.Generator = None):
    """加权 2D 线 RANSAC：返回 (n2, d2, inlier_mask, spread)。n2 单位；
    d2 = n2·p。spread=内点沿线的展开长度。
    n2d：逐点 2D 法向（符号无关）——**法向内点门**：真边内点的法向与
    线法向一致；角部弦（拐角两侧各抓一段点连成斜线）两侧法向互斥被
    一票否决（金样例 L 形实锤：角弦支撑 22%，吞真边的点级联丢边）。"""
    n = len(p2)
    if n < 2 or w.sum() <= 0:
        return None
    p = w / w.sum()
    idx_all = np.arange(n)
    cos_n = np.cos(np.radians(20.0)) if n2d is not None else 0.0
    best = None
    for _ in range(iters):
        i, j = rng.choice(idx_all, size=2, replace=False, p=p)
        d = p2[j] - p2[i]
        L = np.linalg.norm(d)
        if L < MIN_EDGE_MM:
            continue
        n2 = np.array([d[1], -d[0]]) / L
        d2 = float(n2 @ p2[i])
        res = np.abs(p2 @ n2 - d2)
        inl = res < band
        if n2d is not None:
            inl = inl & (np.abs(n2d @ n2) > cos_n)
        ws = float(w[inl].sum())
        if best is None or ws > best[0]:
            best = (ws, inl)
    if best is None:
        return None
    _ws, inl = best
    # LS 精化（内点 PCA 最小方向；内点集保持法向门）
    q = p2[inl]
    wq = w[inl]
    sw = float(wq.sum())
    if sw <= 0:                    # 内点权重全零（空超点）：退回粗线，
        return None                # 不能拿 0 除算出 nan 协方差
    mu = _wmean(q, wq)
    cov = ((q - mu) * wq[:, None]).T @ (q - mu) / sw
    _, vec = np.linalg.eigh(cov)
    n2 = vec[:, 0]
    d2 = float(n2 @ mu)
    res = np.abs(p2 @ n2 - d2)
    inl = res < band
    if n2d is not None:
        inl = inl & (np.abs(n2d @ n2) > cos_n)
    t = (p2[inl] - mu) @ (vec[:, 1])      # 沿线坐标
    spread = float(t.max() - t.min()) if inl.sum() >= 2 else 0.0
    return n2, d2, inl, spread


def _orient_line(n2: np.ndarray, d2: float, others: np.ndarray,
                 wo: np.ndarray) -> tuple[np.ndarray, float]:
    """线定向：其余边界点应在 n·p−d ≤ 0 一侧（加权中位数投票）——
    凹多边形安全（不依赖形心在形内）。"""
    s = others @ n2 - d2
    med = float(np.median(s)) if len(s) else 0.0
    if med > 0:
        return -n2, -d2
    return n2, d2


def _seg_dist(p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    l2 = float(ab @ ab)
    if l2 < 1e-12:
        return np.linalg.norm(p2 - a, axis=1)
    t = np.clip(((p2 - a) @ ab) / l2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p2 - proj, axis=1)


def fit_member(cent: np.ndarray, nrm: np.ndarray, n_pts: np.ndarray,
               max_lines: int = 14, rng: np.random.Generator = None,
               min_band: float = BAND_MM) -> dict:
    """单构件棱柱拟合。返回 dict：
    ok / quality / axis(3,) / faces=[{kind:'side'/'end', n(3), d, W,
    res_p90}] / assign (M,) 逐超点面序号（-1 未归属，侧面 0..L-1 按角度
    序、端面 L/L+1）。"""
    if rng is None:
        rng = np.random.default_rng(0)
    M = len(cent)
    out = {"ok": False, "faces": [], "assign": np.full(M, -1, np.int64),
           "axis": np.zeros(3), "quality": {}, "poly_area": 0.0}
    w = n_pts.astype(np.float64)
    if M < 12 or w.sum() <= 0:
        out["quality"]["reason"] = "too_few_sp"
        return out

    axis = estimate_axis(cent, nrm, w)
    out["axis"] = axis
    # 正交基
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    dots = nrm @ axis
    end_m = np.abs(dots) > 0.7
    side_m = ~end_m
    if side_m.sum() < 8 or end_m.sum() < 2:
        out["quality"]["reason"] = "side/end 分裂失败"
        return out

    # ── 截面 2D 提取（迭代线 RANSAC，法向内点门）──────────────────
    p2 = np.column_stack([cent[side_m] @ u, cent[side_m] @ v])
    n2d_all = np.column_stack([nrm[side_m] @ u, nrm[side_m] @ v])
    n2d_all /= np.maximum(np.linalg.norm(n2d_all, axis=1, keepdims=True),
                          1e-9)
    w2 = w[side_m]
    lines = []                            # (n2, d2, inl_idx_local, spread)
    alive = np.ones(len(p2), bool)
    wa = w2.copy()
    total_w = float(w2.sum())
    for _ in range(max_lines):
        if alive.sum() < 6 or wa[alive].sum() < 0.02 * total_w:
            break
        res = _line_ransac(p2[alive], wa[alive], n2d=n2d_all[alive],
                           band=min_band, rng=rng)
        if res is None:
            break
        n2, d2, inl_local, spread = res
        if spread < MIN_EDGE_MM:
            alive_glob = np.flatnonzero(alive)
            alive[alive_glob[inl_local][:max(1, len(inl_local) // 2)]] = False
            continue
        gidx = np.flatnonzero(alive)[inl_local]
        lines.append([n2, d2, gidx, spread])
        alive[gidx] = False
    # ── 装配前支撑门 ──────────────────────────────────────────────
    # 杂散点碎片线（W~0.2%）在完整路径里搅烂边图（交点表混入假顶点），
    # 在降级路径里输出垃圾面——一律在装配前丢掉。门后 ≥2 线即可继续
    # （完整装配需 ≥3，部分可见构件走降级输出线级平面）。
    lines = [ln for ln in lines
             if float(w2[ln[2]].sum()) >= MIN_LINE_W_FRAC * total_w]
    if len(lines) < 2:
        out["quality"]["reason"] = f"lines<2 after gate ({len(lines)})"
        return out

    # ── 平行弦合并 + 宽带 LS 重拟合 ────────────────────────────────
    # 翘曲/轴残差把一条真边涂成 ±20mm 的 smear，band 12mm 的 RANSAC
    # 会在同一条边上先后抽出多条平行弦（金样例实锤：4 边抽出 6 线）。
    # 合并：法向夹角 <10° 且质心处偏移 <40mm 的弦并为一簇；每簇取全体
    # 25mm 带内点做 LS 重拟合（中位线=真边）。
    merged = True
    while merged:
        merged = False
        for i in range(len(lines)):
            if merged:
                break
            for j in range(i + 1, len(lines)):
                ni, di, gi, si = lines[i]
                nj, dj, gj, sj = lines[j]
                if abs(float(ni @ nj)) < np.cos(np.radians(10.0)):
                    continue
                ci = _wmean(p2[gi], w2[gi])
                cj = _wmean(p2[gj], w2[gj])
                if abs(float(ni @ cj - di)) > 40.0:
                    continue
                g = np.unique(np.concatenate([gi, gj]))
                q = p2[g]
                wq = w2[g]
                mu = _wmean(q, wq)
                cov = ((q - mu) * wq[:, None]).T @ (q - mu) / wq.sum()
                _, vec = np.linalg.eigh(cov)
                n2 = vec[:, 0]
                if float(n2 @ ni) < 0:
                    n2 = -n2
                d2 = float(n2 @ mu)
                inl = np.abs(p2 @ n2 - d2) < 25.0
                t = (p2[inl] - mu) @ vec[:, 1]
                spread = float(t.max() - t.min()) if inl.sum() >= 2 else 0.0
                lines[i] = [n2, d2, np.flatnonzero(inl), spread]
                lines.pop(j)
                merged = True
                break

    # ── 线定向（全点单侧投票）→ 边图行走装配 ─────────────────────
    # 角度排序装配对凹形+平行边不成立（L 有 3 对平行边，角度排序交点
    # 飞出，面积爆炸 19.8M mm² 实锤）。改边图：交点落在两边范围
    # （±80mm 容差）内才算合法邻居；每条边的多边形顶点 = 合法交点的
    # 两个极端（沿边坐标 min/max）；从任一边沿极值顶点走链，闭环即多边形。
    # 部分可见构件缺边/交点缺失时闭合原理上不可行——交付物是平面参数
    # （检测头消费面，不消费多边形），装配失败 → degraded：过滤后各线
    # 直接作为 side 面假设（归属用无限线距），不 return。
    others = p2[w2 > 0]
    wo = w2[w2 > 0]
    L = len(lines)
    dirs, extents = [], []
    for k, (n2, d2, gidx, spread) in enumerate(lines):
        n2o, d2o = _orient_line(n2, d2, others, wo)
        lines[k][0], lines[k][1] = n2o, d2o
        di = np.array([-n2o[1], n2o[0]])           # 沿线方向
        ti = p2[gidx] @ di
        dirs.append(di)
        extents.append((float(ti.min()), float(ti.max())))

    class _WalkFail(Exception):
        pass

    degraded = False
    seg_of = None
    order = None
    try:
        # 合法交点表
        vert_of = {}                                # (i,j) i<j → 交点
        t_at = {}                                   # 交点在 i 沿线坐标
        for i in range(L):
            for j in range(i + 1, L):
                n1, d1, _g, _s = lines[i]
                n2_, d2_, _g2, _s2 = lines[j]
                A = np.array([n1, n2_])
                det = abs(np.linalg.det(A))
                if det < 1e-9:
                    continue
                vij = np.linalg.solve(A, np.array([d1, d2_]))
                ti = vij @ dirs[i]
                tj = vij @ dirs[j]
                lo_i, hi_i = extents[i]
                lo_j, hi_j = extents[j]
                if (lo_i - 80 <= ti <= hi_i + 80) and \
                        (lo_j - 80 <= tj <= hi_j + 80):
                    vert_of[(i, j)] = vij
                    t_at[(i, j)] = ti
                    t_at[(j, i)] = tj
        if len(vert_of) < L:
            raise _WalkFail(
                f"edge graph incomplete ({len(vert_of)}/{L})")
        # 每边两个极端邻居
        nbr = {}
        for i in range(L):
            cand = [(t_at[(i, j)], j) for j in range(L)
                    if j != i and (min(i, j), max(i, j)) in vert_of]
            if len(cand) < 2:
                raise _WalkFail(f"edge {i} has <2 neighbors")
            cand.sort()
            nbr[i] = (cand[0][1], cand[-1][1])
        # 闭环行走：E = 边的环序；V_i = E[i] 与 E[i+1] 的交点
        E = [0]
        prev_edge, cur_edge = 0, nbr[0][1]
        E.append(cur_edge)
        for _ in range(4 * L):
            a, b = nbr[cur_edge]
            nxt = b if a == prev_edge else a
            if nxt == 0:
                break
            E.append(nxt)
            prev_edge, cur_edge = cur_edge, nxt
        else:
            raise _WalkFail("walk did not close")
        if len(E) != L:
            raise _WalkFail(f"walk wrong length ({len(E)}/{L})")
        # V_i = E[i]×E[i+1] 交点（V_last = E[-1]×E[0] 闭环）
        V = []
        for i in range(len(E)):
            j = (i + 1) % len(E)
            key = (min(E[i], E[j]), max(E[i], E[j]))
            if key not in vert_of:
                raise _WalkFail("closure vertex missing")
            V.append(vert_of[key])
        seg_of = {E[i]: (V[i - 1], V[i]) for i in range(len(E))}
        # 面积（shoelace）
        area = 0.0
        for k in range(len(V)):
            j = (k + 1) % len(V)
            area += V[k][0] * V[j][1] - V[j][0] * V[k][1]
        area *= 0.5
        if area < 0:
            # 环序反向——统一把每边线段端点也反转（外向法向不受影响：
            # 线定向由"全点单侧投票"独立决定）
            V = V[::-1]
            seg_of = {E[i]: (V[i - 1], V[i]) for i in range(len(E))}
            area = -area
        if area < 5e4 or not np.isfinite(area):     # 0.05 m² 退化门
            raise _WalkFail(f"area {area:.0f}mm²")
        out["poly_area"] = float(area)
        order = E
    except _WalkFail as e:
        degraded = True
        seg_of = None               # 可能已在 try 内赋值后，面积门才抛
        out["quality"]["walk_fail"] = str(e)
        order = list(range(L))
    if degraded:
        out["poly_area"] = 0.0

    # ── 端面 ──────────────────────────────────────────────────────
    x = cent @ axis
    # 按【轴向位置】分 lo/hi 簇——法向是 z+ 规范化朝向，两端面共享同一
    # 规范符号，按符号分会永远空一簇（金样例 box-clean 实锤）
    x_mid = float((x[end_m] * w[end_m]).sum() / max(w[end_m].sum(), 1e-9))
    lo_m = end_m & (x < x_mid)
    hi_m = end_m & (x >= x_mid)
    if lo_m.sum() < 1 or hi_m.sum() < 1:
        out["quality"]["reason"] = "end cluster empty"
        return out
    x_lo = float(_wmean(cent[lo_m], w[lo_m]) @ axis)
    x_hi = float(_wmean(cent[hi_m], w[hi_m]) @ axis)
    if x_hi - x_lo < 500:
        out["quality"]["reason"] = "member too short"
        return out

    # ── 归属（侧面 2D 线段距 vs 端面轴向距，argmin）────────────────
    p_all = np.column_stack([cent @ u, cent @ v])
    x_all = cent @ axis
    dmat = np.full((M, L), np.inf)
    for k in range(L):
        li = order[k]
        if seg_of is not None:
            a, b = seg_of[li]
            dmat[:, k] = _seg_dist(p_all, a, b)
        else:                       # 降级：无多边形，无限线距
            dmat[:, k] = np.abs(p_all @ lines[li][0] - lines[li][1])
    d_lo = np.abs(x_all - x_lo)
    d_hi = np.abs(x_hi - x_all)
    stack = np.column_stack([dmat, d_lo, d_hi])
    assign = stack.argmin(1)
    assign[(stack.min(1) > 60.0)] = -1          # 离群超点
    out["assign"] = assign

    # ── 面假设（世界系：n_world = nu·u + nv·v；d 与原点共架）────────
    faces = []
    W_total = float(w.sum())
    for k in range(L):
        li = order[k]
        n2o, d2o, gidx, spread = lines[li]
        sel = assign == k
        W_f = float(w[sel].sum())
        n3 = n2o[0] * u + n2o[1] * v
        res = np.abs(p_all[sel] @ n2o - d2o) if sel.any() else np.array([9e9])
        faces.append({"kind": "side", "n": n3, "d": d2o,
                      "W": W_f, "res_p90": float(np.percentile(res, 90))
                      if len(res) else 9e9})
    for tag, x_e, n_e in (("end_lo", x_lo, -axis), ("end_hi", x_hi, axis)):
        sel = assign == (L if tag == "end_lo" else L + 1)
        W_f = float(w[sel].sum())
        res = np.abs(x_all[sel] - x_e) if sel.any() else np.array([9e9])
        # n·p=d 规范化：n=−axis → d=−x_lo；n=+axis → d=+x_hi
        faces.append({"kind": "end", "n": n_e,
                      "d": float(-x_lo if tag == "end_lo" else x_hi),
                      "W": W_f, "res_p90": float(np.percentile(res, 90))
                      if len(res) else 9e9})
    out["faces"] = faces

    # ── 质量门 ────────────────────────────────────────────────────
    # res_p90 只统计有支撑面：降级路径允许缺端面/缺边（部分可见构件），
    # 无支撑面没有残差证据，不应连坐整个拟合（完整路径全部面有支撑，
    # 行为不变）
    cov = float(w[assign >= 0].sum() / W_total)
    sup_side = [f for f in faces[:L] if f["W"] > 0]
    min_side = min((f["W"] for f in sup_side), default=0.0)
    res_p90 = max(f["res_p90"] for f in faces if f["W"] > 0)
    q = {"coverage": round(cov, 3), "n_lines": L,
         "min_side_frac": round(min_side / W_total, 4),
         "res_p90_max": round(res_p90, 1),
         "axis_z": round(float(abs(axis[2])), 3),
         "degraded": degraded}
    if "walk_fail" in out["quality"]:
        q["walk_fail"] = out["quality"]["walk_fail"]
    out["quality"] = q
    ok = (cov >= COV_MIN and res_p90 <= 40.0
          and min_side / W_total >= MIN_SIDE_W_FRAC
          and abs(axis[2]) <= AXIS_Z_MAX)
    out["ok"] = bool(ok)
    if not ok:
        out["quality"]["reason"] = (
            "coverage" if cov < COV_MIN else
            "residual" if res_p90 > 40.0 else
            "min_side" if min_side / W_total < MIN_SIDE_W_FRAC else
            "axis_gravity")
    return out
