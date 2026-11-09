# -*- coding: utf-8 -*-
"""几何测量层（检测-量测双轨的"量测"段，2026-10-25 固化）。

分工红线：深度学习只做语义分割（哪些点是什么），本模块把分割后的
类别点云变成五组装配参数——全部几何、零学习、域无关。

方法（2026-10-25 天花板验证，v2.4 test 24 件 oracle 标签：
根数 24/24、位置 p50 1.44mm、规格件级 92%、套筒 ID p50 0.5mm）：

- 钢筋根数：端面带（FACE_BAND）2D DBSCAN 播种 + 宽簇二次分裂 +
  Voronoi 归簇（cap=min(VORONOI_CAP, 0.95×次近种子距)）——与沿筋
  密度断裂无关（T-035：阈值 25 丢遮挡筋、eps 40 并簇的教训）。
- 插入位置：逐筋 x 分箱质心轨迹 robust 二次拟合 q(x)（可见弧的质心
  偏置 e_arc 是沿筋常量，被 q 常数项吸收）+ 端头同心圆组（筋弧/端面
  盘/垫片墙同圆心，圆拟合不受 e_arc 影响）取最大圆（垫片墙）圆心，
  root = 圆心 + q(x_face) − q(x_wall)（e_arc 差分严格抵消）。
  v1 教训：resid 阈值（带 e_arc）会切掉垫片环内侧点→浅弧拟合漂移
  7–31mm——故本版用同心圆组，不做阈值过滤。
- 直径规格：侧影半径含 0.45–1.0 随机收缩因子（两域同证）不可测；
  名义径载体=垫片墙半径（f×r_true，f~U(2.0,2.4) 件内池化均值 2.2）
  与端面盘半径（r_true+1，精确）；件内同径池化后 snap 规格表。
- 套筒：端面凸出带环候选 → 环校验（环形支撑=孔心空 + 半径范围 +
  角度覆盖）→ 中心=环圆心、内径=径向剖面首支撑、根数=过校验环数。

常数全部物理导出、一次设定（见各常数注释的安全区推导），无逐件
调参；跨域只允许"分治规则"（按可自动判定的构件类型选分支）。
"""

import numpy as np

# ── 常数（物理推导见行尾；改前先读推导） ─────────────────────────────
FACE_BAND = 150.0    # 端带宽度 mm：只要求每筋有截面点；外露≥400 → 安全区 50–300
SEED_EPS = 20.0      # 播种 DBSCAN：∈(筋上点距~3, 最小筋距/2=22.5)，取中偏上
SEED_MIN = 12        # 种子最少点数：低密度+遮挡盘的下限（25 曾丢遮挡筋）
WIDE_EXT = 30.0      # face 簇对角上限：单盘 ≤2r_v+噪声≈26；超限=并簇嫌疑
SPLIT_EPS = 11.0     # 并簇二次分裂 eps（< 盘径，≥盘内点距）
SPLIT_MIN = 12
VORONOI_CAP = 40.0   # 归簇距离上限（防远端离群；实际由 0.95×次近距收紧）
MIN_BAR_PTS = 120    # 成筋最少点数（外露 400×低密度弧 的下限量级）
SLICE_W = 15.0       # 轨迹分箱宽：≥点距，≪弓曲波长
TRAJ_MIN_BINS = 6
TRAJ_MIN_PTS = 20    # 每箱最少点数
TIP_BAND = 85.0      # 端头同心圆组搜索带：≥垫片长上限 60 + 余量
TIP_EDGE = 70.0      # 轨迹拟合在 x_tip−TIP_EDGE 处截断（垫片不进轨迹）
RING_MIN_PTS = 40    # 垫片墙最少点数
WASHER_F = 2.2       # 垫片半径系数均值（生成器/工程惯例 U(2.0,2.4)）
DISK_DELTA = 1.0     # 端面盘半径 = r_true + DISK_DELTA（生成器定义）
EXPO_TOPK = 15       # 外露=顶端 k 点均值（抗单点噪声）
MIN_EXPOSED = 200.0  # 低于此长度的簇不算筋（杯垫/碎块）


def kasa(xy):
    """Kåsa 圆拟合（带一轮残差剔除）。xy (N,2) → (cx, cy, r) | None。"""
    xy = np.asarray(xy, np.float64)
    if len(xy) < 8:
        return None
    A = np.column_stack([xy, np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
    res = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)
    keep = res < max(2.5, np.quantile(res, 0.75))
    if (~keep).any() and keep.sum() >= 8:
        sol, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
    return cx, cy, r


def robust_quad(x, v):
    """二次拟合 + 3×MAD 离群剔除（3 轮）。返回系数 (a,b,c)：v≈a x²+bx+c。"""
    x, v = np.asarray(x, float), np.asarray(v, float)
    keep = np.ones(len(x), bool)
    sol = None
    for _ in range(3):
        Pk = np.column_stack([x[keep] ** 2, x[keep], np.ones(keep.sum())])
        sol, *_ = np.linalg.lstsq(Pk, v[keep], rcond=None)
        P = np.column_stack([x ** 2, x, np.ones(len(x))])
        res = np.abs(v - P @ sol)
        mad = np.median(res[keep]) * 1.4826 + 1e-6
        new = res < max(3 * mad, 1.0)
        if new.sum() < 5 or (new == keep).all():
            break
        keep = new
    return sol


def concentric_ring_center(tip_pts, tip_x):
    """端头点集 → 最大同心圆（垫片墙）的 (圆心 yz, 半径, 墙均 x)。

    tip 区三类点同心：筋弧(r_v)/端面盘(r_true+1)/垫片墙(w_r)。步骤：
    木质心（含 e_arc 小偏）→ 距离 → [0.45,0.92]×q98 内最大间隙分出
    外环 → Kåsa → 以拟合圆心重选 |d−r|≤3 环带 → 精拟合（消除木质心
    偏置引起的环带切口）。圆拟合本身不受 e_arc 影响（弧点仍在同心
    圆上），这是相对 v1（resid 阈值过滤）的根治。
    """
    xy = tip_pts[:, 1:3].astype(np.float64)
    if len(xy) < RING_MIN_PTS:
        return None
    d0 = np.linalg.norm(xy - xy.mean(0), axis=1)
    dq = np.quantile(d0, 0.98)
    lo, hi = 0.45 * dq, 0.92 * dq
    band = d0[(d0 > lo) & (d0 < hi)]
    if len(band) < 10:
        return None
    hist, edges = np.histogram(band, bins=np.arange(lo, hi + 1.0, 1.0))
    if len(hist) < 3:
        return None
    gi = int(np.argmax(hist))
    r_cut = edges[gi] + 0.5
    sel = d0 > r_cut
    if sel.sum() < RING_MIN_PTS:
        return None
    kk = kasa(xy[sel])
    if kk is None:
        return None
    for _ in range(2):  # 以拟合圆心重选环带，消除初心的环带切口
        d1 = np.linalg.norm(xy - kk[:2], axis=1)
        sel = np.abs(d1 - kk[2]) < 3.0
        if sel.sum() < RING_MIN_PTS:
            break
        kk2 = kasa(xy[sel])
        if kk2 is None:
            break
        kk = kk2
    return kk[0], kk[1], kk[2], float(tip_x)


def seed_bars(rp, x_face):
    """端面带播种 → (seeds (K,2), assign 函数)。返回种子与逐点归属 id
    （-1=未归属）。"""
    from sklearn.cluster import DBSCAN
    fb = rp[(rp[:, 0] > x_face - 10) & (rp[:, 0] < x_face + FACE_BAND)]
    if len(fb) < SEED_MIN:
        return np.zeros((0, 2)), np.full(len(rp), -1)
    cl = DBSCAN(eps=SEED_EPS, min_samples=SEED_MIN).fit(fb[:, 1:3])
    seeds = []
    for k in np.unique(cl.labels_):
        if k < 0:
            continue
        m = cl.labels_ == k
        if m.sum() < SEED_MIN:
            continue
        sub = fb[m, 1:3]
        if np.linalg.norm(sub.max(0) - sub.min(0)) > WIDE_EXT:
            # 并簇嫌疑（错位使相邻盘隙 < eps）：小 eps 二次分裂
            cl2 = DBSCAN(eps=SPLIT_EPS, min_samples=SPLIT_MIN).fit(sub)
            for k2 in np.unique(cl2.labels_):
                if k2 < 0:
                    continue
                m2 = cl2.labels_ == k2
                if m2.sum() >= SPLIT_MIN:
                    seeds.append(sub[m2].mean(0))
        else:
            seeds.append(sub.mean(0))
    if not seeds:
        return np.zeros((0, 2)), np.full(len(rp), -1)
    seeds = np.array(seeds)
    d2 = ((rp[:, None, 1:3] - seeds[None]) ** 2).sum(-1)
    srt = np.sort(d2, axis=1)
    cap = (np.minimum(VORONOI_CAP, 0.95 * np.sqrt(srt[:, 1]))
           if len(seeds) > 1 else np.full(len(rp), VORONOI_CAP))
    nearest = d2.argmin(1)
    assign = np.where(np.sqrt(srt[:, 0]) < cap, nearest, -1)
    return seeds, assign


def measure_bar(bar, x_face):
    """单筋测量。bar (N,3) 世界系 → dict 或 None。

    root=(y,z) 端面插入位置；spec 证据 wall_r/disk_r；expo 外露长。
    """
    if len(bar) < MIN_BAR_PTS:
        return None
    xs = bar[:, 0]
    x_tip = float(np.quantile(xs, 0.997))
    x_lo = float(np.quantile(xs, 0.003))
    if x_tip - x_lo < MIN_EXPOSED:
        return None
    bx, by, bz = [], [], []
    for s in np.arange(x_lo + 8, x_tip - TIP_EDGE, SLICE_W):
        m = (xs >= s) & (xs < s + SLICE_W)
        if m.sum() < TRAJ_MIN_PTS:
            continue
        bx.append(s + SLICE_W / 2)
        by.append(bar[m, 1].mean())
        bz.append(bar[m, 2].mean())
    if len(bx) < TRAJ_MIN_BINS:
        return None
    qy = robust_quad(bx, by)
    qz = robust_quad(bx, bz)
    if qy is None or qz is None:
        return None
    resid = np.hypot(bar[:, 1] - (qy[0] * xs ** 2 + qy[1] * xs + qy[2]),
                     bar[:, 2] - (qz[0] * xs ** 2 + qz[1] * xs + qz[2]))
    mid = (xs > x_lo + 100) & (xs < x_tip - 90)
    r_v = float(np.median(resid[mid])) if mid.sum() > 50 else 6.0
    xf = min(max(x_face, x_lo), x_tip - TIP_EDGE)
    tip = bar[xs > x_tip - TIP_BAND]
    ring = concentric_ring_center(tip, x_tip)
    if ring is not None and 1.3 * r_v < ring[2] < 5.5 * r_v:
        cy, cz, w_r, _ = ring
        ry = (qy[0] * (xf ** 2 - x_tip ** 2) + qy[1] * (xf - x_tip)) + cy
        rz = (qz[0] * (xf ** 2 - x_tip ** 2) + qz[1] * (xf - x_tip)) + cz
        wall_r, mode = w_r, "ring"
    else:  # 无垫片兜底：轨迹直接外插（带 e_arc 偏置，标记降级）
        ry = qy[0] * xf ** 2 + qy[1] * xf + qy[2]
        rz = qz[0] * xf ** 2 + qz[1] * xf + qz[2]
        wall_r, mode = None, "traj"
    tm = xs > x_tip - 12
    dt = resid[tm]
    disk = np.sort(dt[dt < r_v + 6])
    disk_r = None
    if len(disk) > 30:
        R0 = float(np.quantile(disk, 0.88))
        ring2 = disk[(disk > 0.8 * R0) & (disk < 1.12 * R0)]
        if len(ring2) > 12:
            disk_r = float(np.quantile(ring2, 0.55))
    return {"root": (float(ry), float(rz)), "mode": mode,
            "wall_r": wall_r, "disk_r": disk_r, "vis_r": r_v,
            "expo": float(np.sort(xs)[-EXPO_TOPK:].mean() - x_face),
            "n": int(len(bar))}


def pool_spec(meas, table):
    """件内规格池化：垫片墙均值÷WASHER_F 为主、端面盘(−DISK_DELTA)为辅
    （盘为精确载体，按数量加权）。→ (d_hat, snap_id, snap_err)。"""
    wall = [m["wall_r"] for m in meas if m.get("wall_r")]
    disk = [m["disk_r"] for m in meas if m.get("disk_r")]
    d_hat = None
    if len(wall) >= 2 and disk:
        dw = 2 * float(np.mean(wall)) / WASHER_F
        dd = 2 * (float(np.median(disk)) - DISK_DELTA)
        w = len(disk) / (len(disk) + len(wall))
        d_hat = w * dd + (1 - w) * dw
    elif len(wall) >= 2:
        d_hat = 2 * float(np.mean(wall)) / WASHER_F
    elif disk:
        d_hat = 2 * (float(np.median(disk)) - DISK_DELTA)
    if d_hat is None:
        return None, None, None
    tab = np.asarray(table, np.float64)
    i = int(np.argmin(np.abs(tab - d_hat)))
    return float(d_hat), i, float(abs(tab[i] - d_hat))


def snap_id(table, d):
    tab = np.asarray(table, np.float64)
    i = int(np.argmin(np.abs(tab - d)))
    return i, float(abs(tab[i] - d))


def measure_rings(pts, face_x, sign=+1.0, band=60.0, pitch=2.0,
                  r_lo=15.0, r_hi=28.0, rim_min=1.0, close_rad_max=8,
                  pk=None, occ_dens_thr=0.0, h_gate=25.0):
    """端面区 → 套筒环列表 [{c, r_in, n}, ...]（无需 GT 先验）。

    密度孔法（真实 9/9 召回）：套筒占据空间
    → 端面点密度图上形成**封闭孔洞**，与凸起无关（凹陷套筒同样可检）。
    步骤：端面强峰+倾斜校正 → |h|≤6mm 面附近点 → 2mm 栅格占据图 →
    密度自适应闭运算补实面板（低密度 Swiss-cheese 会导致孔泄漏）→
    fill_holes 提取封闭孔 → 面积/圆度筛选 → **孔缘顶帽得分**（2mm 格
    均值−51 格中位数背景，local_face_reference）剔除无数据斑孔
    （斑缘平≈0，套筒孔缘被垫片环抬高≥1mm；真实凹陷套筒传 rim_min≤0）。
    缺弧环孔不封闭会漏检（数据固有；留 Hough 融合接口）。

    close_rad_max：闭运算结构元半径上限（默认 8，"孔 42mm≫8mm 不桥接"
    只对合成规格孔 r≥18 成立；小孔域必须同步收紧——物理规则
    ≈0.35×r_lo，真实 Ø27 孔传 3，2026-10-25 真实诊断实锤 rad≈6 桥死
    全部小孔）。pk：外部钉住端面轴向位置（真实两级端面/凸台时粗峰可
    能锁错层，套筒环随层被 |h| 门控筛掉；传视图构造阶段的同一峰值，
    2026-10-25 真实验证教训）。occ_dens_thr：占据密度阈值（点/2mm 格），
    默认 0=任一点即占据（合成孔真空心成立）；真实孔内有稀疏点
    （实测 0~154 点/孔），cnt>0 会把孔碎化——传 0.2（相对密度阈值）恢复孔洞。
    """
    from scipy import ndimage
    rel = (pts[:, 0] - face_x) * sign
    if pk is not None:
        # 外部钉住端面（两级端面/凸台防锁错层，见 docstring）
        m = np.abs(rel - pk) < band
        P = pts[m]
        if len(P) < 200:
            return []
        rel_p = rel[m]
    else:
        # 粗峰定位放宽到端部 2m：翘曲可把端面推出名义位置数百毫米
        # （v2.4 实测 +45 到 >500），端面=局部最密 1mm 平面
        m_coarse = (rel > rel.max() - 2000.0) & (rel < rel.max() + 20.0)
        if m_coarse.sum() < 200:
            return []
        hist_c, edges_c = np.histogram(rel[m_coarse],
                                       bins=np.arange(rel.max() - 2000,
                                                      rel.max() + 21, 1.0))
        pk_c = edges_c[int(np.argmax(hist_c))] + 0.5
        m = np.abs(rel - pk_c) < band
        P = pts[m]
        if len(P) < 200:
            return []
        rel_p = rel[m]
        hist, edges = np.histogram(rel_p, bins=np.arange(pk_c - band,
                                                         pk_c + band + 0.5,
                                                         1.0))
        pk = edges[int(np.argmax(hist))] + 0.5
    fm = np.abs(rel_p - pk) < 4.0
    if fm.sum() > 100:
        Y, Z, X = P[fm, 1], P[fm, 2], rel_p[fm]
        A = np.column_stack([Y, Z, np.ones(fm.sum())])
        coef, *_ = np.linalg.lstsq(A, X, rcond=None)
        base = coef[0] * P[:, 1] + coef[1] * P[:, 2] + coef[2]
    else:
        base = pk
    # 门控（固化实验 2026-10-25 迭代定案）：全局倾斜面 ±25mm——完整环
    # 召回 79.9%（|h|<6: 70%/局部中位数基准: 56%/取消过滤: 73%）——
    # 局部基准的正确实现（空格 NA 处理）是待解项
    h = rel_p - base
    P = P[np.abs(h) < h_gate]
    h = h[np.abs(h) < h_gate]
    # 顶帽得分格网（孔缘检验用）：格均值高程 − 大窗中位数背景
    tu = ((P[:, 1] - P[:, 1].min()) / pitch).astype(int)
    tv = ((P[:, 2] - P[:, 2].min()) / pitch).astype(int)
    tu_n, tv_n = tu.max() + 1, tv.max() + 1
    tflat = tu * tv_n + tv
    cnt = np.bincount(tflat, minlength=tu_n * tv_n).reshape(tu_n, tv_n)
    sm = np.bincount(tflat, weights=h,
                     minlength=tu_n * tv_n).reshape(tu_n, tv_n)
    hm = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)
    bg = ndimage.median_filter(hm, size=51)
    score_grid = hm - bg

    y0, z0 = P[:, 1].min() - pitch, P[:, 2].min() - pitch
    iu = ((P[:, 1] - y0) / pitch).astype(int)
    iv = ((P[:, 2] - z0) / pitch).astype(int)
    nu, nv = iu.max() + 3, iv.max() + 3
    cnt = np.zeros((nu, nv), np.int32)
    np.add.at(cnt, (iu + 1, iv + 1), 1)
    occ = cnt > occ_dens_thr
    # 密度自适应闭运算：空格率越高结构元越大（低密度端面 Swiss-cheese
    # 会让孔泄漏到外连通域而不封闭）
    f0 = 1.0 - occ.mean()
    rad = int(np.clip(2 + round(f0 * 8), 2, max(2, close_rad_max)))
    st = np.zeros((2 * rad + 1, 2 * rad + 1), bool)
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    st[yy ** 2 + xx ** 2 <= rad ** 2] = True
    occ = ndimage.binary_closing(occ, structure=st)
    holes = ndimage.binary_fill_holes(occ) & ~occ
    labs, n_lab = ndimage.label(holes)
    a_lo = np.pi * r_lo ** 2 / (pitch * pitch) * 0.55   # 面积(格)=πr²/pitch²
    a_hi = np.pi * r_hi ** 2 / (pitch * pitch) * 1.15
    out = []
    for k in range(1, n_lab + 1):
        cells = np.argwhere(labs == k)
        area = len(cells)
        if not (a_lo <= area <= a_hi):
            continue
        r_eq = float(np.sqrt(area / np.pi) * pitch)
        cy, cx = cells.mean(0)
        # 圆度：孔元到质心距离的变异（填充圆盘理论 CV=0.354，取 0.45）
        d = np.linalg.norm(cells - [cy, cx], axis=1)
        if d.std() / max(d.mean(), 1e-6) > 0.45:
            continue
        cov = np.cov((cells - [cy, cx]).T)
        ev = np.linalg.eigvalsh(cov)
        if ev[1] < 1e-6 or np.sqrt(ev[0] / ev[1]) < 0.55:
            continue
        hy = y0 + (cy - 1) * pitch + pitch / 2
        hz = z0 + (cx - 1) * pitch + pitch / 2
        # 孔缘顶帽得分：剔除无数据斑孔（斑缘平，套筒孔缘被垫片环抬高）
        dd = np.hypot(P[:, 1] - hy, P[:, 2] - hz)
        sel = np.where((dd > r_eq - 2) & (dd < r_eq + 10))[0]
        if len(sel) >= 30:
            rim_score = float(np.quantile(score_grid[tu[sel], tv[sel]], 0.8))
            if rim_score < rim_min:
                continue
        # 内径精测：孔邻域环壁点的内边缘（18 扇区各取 d 的 5% 分位→中数；
        # 面积等效半径被格点侵占+挖孔余量侵蚀约 2-3mm，此法回归物理边）
        am = (dd > r_eq - 4) & (dd < r_eq + 14)
        r_in = r_eq
        if am.sum() > 60:
            ang = np.arctan2(P[am, 2] - hz, P[am, 1] - hy)
            sec = ((ang + np.pi) / (np.pi / 9)).astype(int) % 18
            edge = []
            for s in range(18):
                ds = dd[am][sec == s]
                if len(ds) >= 4:
                    edge.append(np.quantile(ds, 0.05))
            if len(edge) >= 10:
                r_in = float(np.median(edge) - 1.0)  # 1mm=挖孔余量
        out.append({"c": (float(hy), float(hz)), "r_in": r_in,
                    "n": int(area)})
    # ── 第二来源：低密度区轮廓（圆度法机制）——缺弧环的低密度
    #    区域与外部连通、fill_holes 不封闭，但轮廓+圆度+真空心仍可检 ──
    try:
        import cv2
        from scipy.ndimage import gaussian_filter
        dens = gaussian_filter(cnt.astype(float), 1.0)
        hole_img = ((dens < 0.2 / (pitch * pitch))).astype(np.uint8) * 255
        hole_img = cv2.morphologyEx(hole_img, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(hole_img, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cc in contours:
            area_mm = cv2.contourArea(cc) * pitch * pitch
            perim = cv2.arcLength(cc, True) * pitch
            if perim <= 1e-6 or area_mm <= 1.0:
                continue
            circ = 4.0 * np.pi * area_mm / (perim * perim)
            r_eq = float(np.sqrt(area_mm / np.pi))
            if not (r_lo <= r_eq <= r_hi) or circ < 0.5:
                continue
            msk = np.zeros_like(hole_img)
            cv2.drawContours(msk, [cc], -1, 1, -1)
            if dens[msk == 1].size and dens[msk == 1].mean() >=                     0.15 / (pitch * pitch):
                continue
            M = cv2.moments(cc)
            if M["m00"] == 0:
                continue
            hy = y0 + M["m10"] / M["m00"] * pitch
            hz = z0 + M["m01"] / M["m00"] * pitch
            if all(np.hypot(o["c"][0] - hy, o["c"][1] - hz) > 25
                   for o in out):
                dd = np.hypot(P[:, 1] - hy, P[:, 2] - hz)
                am = (dd > r_eq - 4) & (dd < r_eq + 14)
                r_in = r_eq
                if am.sum() > 60:
                    ang = np.arctan2(P[am, 2] - hz, P[am, 1] - hy)
                    sec = ((ang + np.pi) / (np.pi / 9)).astype(int) % 18
                    edge = [np.quantile(dd[am][sec == t_], 0.05)
                            for t_ in range(18)
                            if (sec == t_).sum() >= 4]
                    if len(edge) >= 10:
                        r_in = float(np.median(edge) - 1.0)
                out.append({"c": (float(hy), float(hz)), "r_in": r_in,
                            "n": int(cv2.contourArea(cc)), "mode": "contour"})
    except ImportError:
        pass
    return out


def complete_grid(rings, grid_tol=0.35, min_neighbors=2):
    """套筒网格规整（离网剔除/按格补漏）。

    物理依据：套筒与纵筋同轴对位、同网格布置（v2.4 生成器与工程事实）。
    步骤：检测中心 PCA 得格方向 → 互距中数得格距 s → 保留格上邻居
    ≥min_neighbors 的检测（离网剔除）→ 对有空位的格点做密度谷验证
    （node 半径 12mm 内点数显著低于环带）后补为推断环。
    rings 元素需含 c (y,z)；返回同构列表（补项 mode="inferred"）。
    """
    # contour 候选（低密度轮廓，精度差）与 hole 检测分离：先按 hole
    # 规划网格，contour 只确认节点，游离丢弃（v2.4 实测 contour 自由
    # 加入会引入泡沫/噪声谷假环）
    base = [r for r in rings if r.get("mode") != "contour"]
    contour = [r for r in rings if r.get("mode") == "contour"]
    rings = base
    if len(rings) < 3:
        confirmed_nodes = {(round(r["c"][0], 1), round(r["c"][1], 1)): r
                           for r in contour}
        out = list(base)
        for key, r in confirmed_nodes.items():
            if all(np.hypot(r["c"][0] - o["c"][0],
                            r["c"][1] - o["c"][1]) > 25 for o in out):
                out.append(r)
        return out
    C = np.array([r["c"] for r in rings])
    ctr = C.mean(0)
    _, _, V = np.linalg.svd(C - ctr, full_matrices=False)
    ax = V[0]
    if ax[np.argmax(np.abs(ax))] < 0:
        ax = -ax
    perp = np.array([-ax[1], ax[0]])
    # 格距 = 最近邻互距中数（正交网格的 NN 距即格距），取 60–300mm 内
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    nn = nn[(nn > 60) & (nn < 300)]
    if len(nn) < 2:
        return rings
    spacing = float(np.median(nn))
    # 4 邻居检验（±spacing 两方向）
    keep, nodes = [], []
    for c in C:
        nb = sum(1 for off in (ax * spacing, -ax * spacing,
                               perp * spacing, -perp * spacing)
                 if np.min(np.linalg.norm(C - (c + off), axis=1)) <
                 spacing * grid_tol)
        if nb >= min_neighbors:
            keep.append(c)
    if len(keep) < 3:
        return rings
    K = np.array(keep)
    # 补漏（共线填补）：检测对间距 ≈ k×格距（k=2,3）时补中间空位——
    # 只沿既有行补，不凭空生成垂直行（单排套筒布局的 L/T 柱会多排）
    out = list(rings)
    cand = set()
    nK = len(K)
    for i in range(nK):
        for j in range(i + 1, nK):
            dvec = K[j] - K[i]
            dist = float(np.linalg.norm(dvec))
            k = round(dist / spacing)
            if k < 2 or k > 3 or abs(dist - k * spacing) > spacing * grid_tol:
                continue
            for m in range(1, k):
                node = K[i] + dvec * (m / k)
                if np.min(np.linalg.norm(C - node, axis=1)) >=                         spacing * grid_tol:
                    cand.add((round(float(node[0]), 1),
                              round(float(node[1]), 1)))
    for (ny_, nz_) in cand:
        # 节点确认：contour 候选落在节点上则采信（带 r_in），否则记推断环
        hit = next((r for r in contour
                    if np.hypot(r["c"][0] - ny_, r["c"][1] - nz_) < 25), None)
        if hit is not None:
            out.append({"c": (ny_, nz_), "r_in": hit["r_in"],
                        "n": hit["n"], "mode": "contour_confirmed"})
        else:
            out.append({"c": (ny_, nz_), "r_in": float("nan"),
                        "n": 0, "mode": "inferred"})
    return out
