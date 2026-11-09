# -*- coding: utf-8 -*-
"""套筒"提议—验证—量测"级联融合（2026-10-25 快验成立后固化，T-035）。

分工：任意来源的环候选（几何密度孔 / BEV 热图峰 / 未来其他检测器）→
统一物理验证器 → 通过者毫米级精测。架构定案：**多源提议、一律验证**
（快验证伪了"几何源免验"变体：几何自身误检 312/838 必须过本验证器）。

快验数字（v2.4 test 64 件 638 环，script_20261025_181300_sleeve_fusion）：
FUSED@0.15 召回 96.9%/误检 7.1%/位置 0.28mm/内径 0.43mm/规格 95.6%，
对 BEV(92.9/4.2/1.01mm/70.5%) 与纯几何(82.4/37.2/0.3mm/87.1%) 两源
全面占优；BEV 阈值 0.1–0.4 融合召回 97.0→96.1% 平坦——**阈值只管
召回，精度由本模块物理判据把守**（真实域检测分数漂移不构成精度风险）。

验证三判据（用户定案：套筒唯一可用先验=**圆环+环心无点**；格点/布局
先验弃用。常数全部物理导出、一次设定）：
① 环壁径向支撑：径向 2mm 扫描（默认 [15,28]mm=规格表内径半径域，
   域外场景可传 r_scan），36×10° 扇区支持率 ≥ COV_MIN（缺弧环松验收）；
② 孔心空：孔内(0.6r 盘)点数 ≤ max(绝对下限 8, 0.35×背景环带面密度×
   盘面积)——密度相对化使判据跨域（合成孔真空心≈0 点；真实孔内
   0–154 点仍远低于端面密度×面积；泡沫/实体中心密度≈端面 → 被杀。
   快验插桩：316 个假提案死于此判据=第一杀器）；
③ 孔缘顶帽：p80(环壁 h) − p25(背景环带 h) ≥ rim_min（垫片环物理凸起；
   无数据斑孔缘平 → 被杀。背景带 r+8..28 且取 p25——p50/宽窗被邻居
   垫片顶污染，快验 v1 实测把真环压死，缩窗+p25 修复）。

精测：环壁点 Kåsa 圆拟合圆心（不受弧角偏置）+ 18 扇区 5% 分位内边缘
（同 geom_measure.measure_rings 内径法，回归物理孔缘）→ ID=2·r_in →
snap 规格表由调用方做（表在 model_pillar.SLEEVE_IDIAM_TABLE）。

坐标约定：pts[:,0]=轴向坐标，pts[:,1:3]=端面平面坐标 (y,z)；face_x=
名义端面位置，sign=+1/-1 使"端面外"为正（环凸出方向）。conc 通常取
混凝土类点（合成 lbls==1）；真实无标签时传端区裁剪点。
"""

import numpy as np

from lib.measure.geom_measure import kasa, measure_rings

# ── 验证器常数（物理推导见模块文档；改前先读推导） ─────────────────────
R_SCAN = np.arange(15.0, 28.1, 2.0)   # 默认径向扫描域=规格表内径半径 18–27±余量
COV_MIN = 0.30                        # 环壁扇区支持率下限（缺弧环松验收）
HOLE_ABS_MIN = 8                      # 孔心空绝对下限（噪声/毛刺余量）
HOLE_DENS_FRAC = 0.35                 # 孔内点密度比端面背景环带显著低（<35%）
H_LO, H_HI = -5.0, 30.0               # 面带：环凸 p10=3.5、真实凹陷至 −2.7
RIM_MIN = 1.0                         # 顶帽门限=垫片环凸起下限（mm）
WALL_W = 4.0                          # 环壁半宽（r±4）
BG_IN, BG_OUT = 8.0, 28.0             # 背景环带（防邻居垫片污染：≤r+28）
DEDUP_RAD = 25.0                      # 提议去重半径（<最小环距一半）


def face_band(pts, face_x, sign, h_lo=H_LO, h_hi=H_HI, pk=None):
    """端面定位（粗峰+倾斜平面，同 measure_rings 头部机制）+ 面带点提取。

    返回 (by, bz, bh) 或 None（端面不可定位）。bh=相对局部端面平面的
    有符号高度（环凸方向为正）。pk=外部钉住的端面轴向位置（真实构件
    有两级端面/凸台时，粗峰可能锁到基层——由视图构造阶段的同一峰值
    传入，单一事实源；2026-10-25 真实验证教训）。
    """
    rel = (pts[:, 0] - face_x) * sign
    if pk is None:
        m = (rel > rel.max() - 2000.0) & (rel < rel.max() + 20.0)
        if m.sum() < 200:
            return None
        hist, edges = np.histogram(rel[m], bins=np.arange(
            rel.max() - 2000.0, rel.max() + 21.0, 1.0))
        pk = edges[int(np.argmax(hist))] + 0.5
    Pm, relm = pts, rel
    fm = np.abs(relm - pk) < 4.0
    if fm.sum() > 100:
        A = np.column_stack([Pm[fm, 1], Pm[fm, 2], np.ones(int(fm.sum()))])
        coef, *_ = np.linalg.lstsq(A, relm[fm], rcond=None)
    else:
        coef = np.array([0.0, 0.0, pk])
    h = rel - (coef[0] * pts[:, 1] + coef[1] * pts[:, 2] + coef[2])
    band = (h > h_lo) & (h < h_hi)
    if int(band.sum()) < 50:
        return None
    return pts[band, 1], pts[band, 2], h[band]


def _sector_cov(y, z, cy, cz, sel, n_sec=36, min_pts=2):
    ang = np.arctan2(z[sel] - cz, y[sel] - cy)
    sec = ((ang + np.pi) / (2 * np.pi / n_sec)).astype(int) % n_sec
    return float((np.bincount(sec, minlength=n_sec) >= min_pts).sum()) / n_sec


def verify_and_measure(by, bz, bh, cy, cz, *, r_scan=R_SCAN,
                       cov_min=COV_MIN, hole_abs_min=HOLE_ABS_MIN,
                       hole_dens_frac=HOLE_DENS_FRAC, rim_min=RIM_MIN,
                       wall_w=WALL_W, bg_in=BG_IN, bg_out=BG_OUT,
                       edge_lo_mm=12.0, edge_q=0.05, edge_agg_q=0.25,
                       edge_bias=0.8):
    """统一物理验证器 + 精测。返回 (dict|None, reason)。

    dict = {c:(y,z) 精测圆心, r_in 孔内缘半径, cov 环壁支持率,
            rim 顶帽分, r_scan_best 命中扫描半径}。
    内缘估计定稿参数（T-039 阶段 2，pred 材质台架扫描 0.786→0.906、
    oracle 0.955→0.964 同升，默认值=定稿）：edge_lo_mm=窗口内伸
    （pred 环壁**内缘点被误分类吃掉**致 r_in 偏大 70%——窗口内伸让
    存活内缘点回到估计，扇区聚合下分位吸收整扇缺失/散点）；
    edge_bias=挖孔余量（pred 点恢复后重标定 1.0→0.8）。
    """
    d = np.hypot(by - cy, bz - cz)
    best_cov, r_ring = 0.0, None
    for r in r_scan:
        sel = np.abs(d - r) < 2.5
        if int(sel.sum()) < 10:
            continue
        cov = _sector_cov(by, bz, cy, cz, sel)
        if cov > best_cov:
            best_cov, r_ring = cov, float(r)
    if r_ring is None or best_cov < cov_min:
        return None, f"cov={best_cov:.2f}"
    # 孔心空（密度相对化）：背景环带面密度 × 0.35 × 孔盘面积，下限绝对值。
    # 背景环带过疏（<10 点，稀疏真实端面常态）时退用全带平均密度——
    # 不退回绝对下限（会误杀真实稀疏孔），全带密度同样≪泡沫实体密度。
    bg = (d > r_ring + bg_in) & (d < r_ring + bg_out)
    hole_disk = d < 0.6 * r_ring
    if int(bg.sum()) >= 10:
        area_bg = np.pi * ((r_ring + bg_out) ** 2 - (r_ring + bg_in) ** 2)
        rho = float(bg.sum()) / max(area_bg, 1e-6)
    else:
        rho = float(len(by)) / max(np.pi * (r_ring + bg_out) ** 2, 1e-6)
    hole_max = max(float(hole_abs_min),
                   hole_dens_frac * rho * np.pi * (0.6 * r_ring) ** 2)
    if int(hole_disk.sum()) > hole_max:
        return None, f"hole n={int(hole_disk.sum())}>{hole_max:.0f}"
    wall = np.abs(d - r_ring) < wall_w
    if int(wall.sum()) < 8:
        return None, "wallpts"
    bg_level = (float(np.quantile(bh[bg], 0.25)) if int(bg.sum()) >= 10
                else 0.0)
    rim = float(np.quantile(bh[wall], 0.8)) - bg_level
    if rim < rim_min:
        return None, f"rim={rim:.1f}(bg={bg_level:.1f})"
    # 精测：环壁 Kåsa 圆心 + 18 扇区分位内边缘（同 measure_rings 法）
    fit = kasa(np.column_stack([by[wall], bz[wall]]))
    cy2, cz2 = (float(fit[0]), float(fit[1])) if fit is not None else (cy, cz)
    d2 = np.hypot(by - cy2, bz - cz2)
    am = (d2 > r_ring - edge_lo_mm) & (d2 < r_ring + 14)
    r_in = r_ring
    if int(am.sum()) > 60:
        ang = np.arctan2(bz[am] - cz2, by[am] - cy2)
        sec = ((ang + np.pi) / (np.pi / 9)).astype(int) % 18
        edge = [float(np.quantile(d2[am][sec == s], edge_q))
                for s in range(18) if int((sec == s).sum()) >= 4]
        if len(edge) >= 10:
            r_in = float(np.quantile(edge, edge_agg_q) - edge_bias)
    return {"c": (cy2, cz2), "r_in": r_in, "cov": best_cov, "rim": rim,
            "r_scan_best": r_ring}, "ok"


def fuse_rings(pts, face_x, sign, *, extra_cands=(), r_scan=R_SCAN,
               rim_min=RIM_MIN, band=60.0, pitch=2.0, dedup_rad=DEDUP_RAD,
               h_lo=H_LO, h_hi=H_HI, face_pk=None, occ_dens_thr=0.0,
               h_gate=25.0, **vw):
    """融合环检测：几何密度孔提议 + extra_cands（如 BEV 峰 (y,z) 列表）
    → 去重 → 统一验证 → 精测。

    返回 [{c, r_in, cov, rim, r_scan_best}]（全部过验证器；被拒候选
    不返回——需要拒绝明细时直接调 face_band+verify_and_measure）。
    r_scan/rim_min 会同步传给 measure_rings 的孔面积/顶帽预筛，两域
    各自传物理值（合成=规格表域/1.0；真实凹陷套筒例 rim_min≤0）。
    闭运算上限=0.35×r_lo（结构元≪最小孔径，防小孔被桥死；
    2026-10-25 真实诊断：稀疏端面 f0≈0.5 → rad 6 桥死 Ø27 孔）。
    face_pk：端面轴向位置旁路（真实两级端面/凸台时防粗峰锁错层——
    传视图构造阶段的同一峰值；内部换算到 rel 空间）。
    """
    r_lo, r_hi = float(r_scan.min()), float(r_scan.max())
    pk_rel = None if face_pk is None \
        else (float(face_pk) - float(face_x)) * sign
    props = [(float(r["c"][0]), float(r["c"][1]))
             for r in measure_rings(pts, face_x, sign=sign, band=band,
                                    pitch=pitch, r_lo=r_lo, r_hi=r_hi,
                                    rim_min=rim_min,
                                    close_rad_max=max(2, int(0.35 * r_lo)),
                                    pk=pk_rel, occ_dens_thr=occ_dens_thr,
                                    h_gate=h_gate)]
    for c in extra_cands:
        cy, cz = float(c[0]), float(c[1])
        if all(np.hypot(cy - p[0], cz - p[1]) > dedup_rad for p in props):
            props.append((cy, cz))
    fb = face_band(pts, face_x, sign, h_lo, h_hi, pk=pk_rel)
    if fb is None:
        return []
    out = []
    for cy, cz in props:
        v, _ = verify_and_measure(fb[0], fb[1], fb[2], cy, cz,
                                  r_scan=r_scan, rim_min=rim_min, **vw)
        if v is not None:
            out.append(v)
    return out
