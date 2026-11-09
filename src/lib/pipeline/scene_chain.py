# -*- coding: utf-8 -*-
"""场景级（多构件）检测头联调·冒烟切片（T-040 修复清单#4）。

**定位**：M4（docs/t037_scene_face_pipeline.md §8，DoD=锚 A1-A4）的
第一切片——"部署级裁剪前置"路线：整场景 → 预测材质分割 → 钢筋类点
DBSCAN 分簇 → 簇群分组=构件 → 逐构件轴对齐裁剪 → 构件级链（生长面+
三头+融合）→ member_table GT 参数级评估。非免裁剪路线（面片不配对成
构件的 scene_fullchain 完整形态）——那是 M4 后续卡的工作。

**限制（如实声明）**：
- 只处理近水平构件（|轴·z|<0.5）；立柱（轴≈重力）端面/顶面语义不同，
  留待 M4 正式卡（skip=vertical_member 入统计）。
- 套筒可见性未启用（sl_vis=None，分母=全部 GT 套筒，遮挡件拉低召回）；
  场景分割/构件归属全用预测链，GT（member_table/member_id/class_id）
  只进评估。
- 外露 GT=p1 翘曲尖口径（P-059 部署差异照旧）。

用法（cwd=src）：论文入口=experiments.exp2_validate scene（批量+池化经
scene_batch）；本模块保留单场景 CLI 作调试直入口：
  python -m lib.pipeline.scene_chain --scene mock/scene/v1/s3/xxx.npz \
    [--limit-members 2] [--out out/scene_fullchain_smoke.json]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from lib.utils import anchor_project_path
from lib.pipeline.single_chain import run_one, aggregate
from lib.pipeline.real_chain import (load_heads, MEMBER_LAT, MEMBER_BACK,
                                        TIP_MARGIN, CLUSTER_EPS, CLUSTER_MIN,
                                        MIN_CLUSTER_N)


def _member_segment_mask(pts, lbls, c, seed_mean):
    """重力系单构件分割：z 直方图分带（堆叠）→ 带内平面栅格连通域（并排）。

    为什么不用任意"场景轴"的垂直平面栅格：堆叠点云是 z 向楼梯状，
    全局 SVD 轴天生带 ~7° 倾斜（0.084×6000mm=504mm 漂移实锤），任何
    1-2° 以上的轴倾斜都会跨过 47mm 垫块缝把连通块焊死。重力 z 没有
    倾斜问题；垫块在 z 向桥接直方图，但它是小块——计数阈值（<带中位
    计数 5%）即可切出缝。水平外构件（z 带内并排）由平面连通域分离。

    连通域加最小面积门（M4 ③，s4 m0 实锤：带内 pred 混凝土碎裂 →
    opening 后种子落 48 点碎片，裁剪整件报废）——碎片不够格，取离
    种子最近的合格块。返回 (构件点 mask 函数或 None, (ya, za))。"""
    from scipy import ndimage
    con = pts[lbls == 1]
    zc = con[:, 2]
    BIN = 10.0
    z0, z1 = float(zc.min()), float(zc.max())
    nb = int((z1 - z0) / BIN) + 1
    hist, _ = np.histogram(zc, bins=np.linspace(z0, z0 + nb * BIN, nb + 1))
    thr = max(20.0, float(np.median(hist[hist > 0])) * 0.01)
    occ = hist > thr
    # 种子 z 所在带：从种子位置向两侧扩到第一个空档
    zi = int(np.clip((float(seed_mean[2]) - z0) / BIN, 0, nb - 1))
    lo = hi = zi
    while lo > 0 and occ[lo - 1]:
        lo -= 1
    while hi < nb - 1 and occ[hi + 1]:
        hi += 1
    Z_MARG = 30.0        # 带外扩半缝宽（缝~47mm，不触邻构件）
    zb_lo, zb_hi = z0 + lo * BIN - Z_MARG, z0 + (hi + 1) * BIN + Z_MARG

    # 带内平面栅格（世界 xy；构件之间有真实缝隙/无垫块桥接）
    inb = (con[:, 2] >= zb_lo) & (con[:, 2] < zb_hi)
    px, py = con[inb, 0] - c[0], con[inb, 1] - c[1]
    CELL = 30.0
    iu = np.floor(px / CELL).astype(np.int64)
    iv = np.floor(py / CELL).astype(np.int64)
    u0, v0 = iu.min(), iv.min()
    nu, nv = int(iu.max() - u0 + 3), int(iv.max() - v0 + 3)
    grid = np.zeros((nu, nv), bool)
    grid[iu - u0 + 1, iv - v0 + 1] = True
    grid = ndimage.binary_opening(grid, structure=np.ones((3, 3)),
                                  iterations=1)
    lab, _n = ndimage.label(grid)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    su, sv = float(seed_mean[0] - c[0]), float(seed_mean[1] - c[1])
    cu = int(np.clip(int(su / CELL) - u0 + 1, 0, nu - 1))
    cv = int(np.clip(int(sv / CELL) - v0 + 1, 0, nv - 1))
    MIN_CELLS = 30       # ~30 cell=2.7 万 mm² 足迹；真构件 ≥10×100 cell
    target = lab[cu, cv]
    if sizes[target] < MIN_CELLS:
        cand = [cid for cid in range(1, len(sizes))
                if sizes[cid] >= MIN_CELLS]
        if not cand:
            return None, None
        cells = np.argwhere(np.isin(lab, cand))
        d = np.abs(cells - [cu, cv]).sum(1)
        target = lab[tuple(cells[int(np.argmin(d))])]
    # 膨胀 1 格（30mm）：筋点在保护层外缘/端面外，不扩会掉边筋
    comp = ndimage.binary_dilation(lab == target, iterations=1)

    def mask(q):
        qz = (q[:, 2] >= zb_lo) & (q[:, 2] < zb_hi)
        qi = np.floor((q[:, 0] - c[0]) / CELL).astype(np.int64) - u0 + 1
        qj = np.floor((q[:, 1] - c[1]) / CELL).astype(np.int64) - v0 + 1
        ok = (qi >= 0) & (qj >= 0) & (qi < nu) & (qj < nv)
        out = np.zeros(len(q), bool)
        cell_ok = np.zeros(int(ok.sum()), bool)
        cell_ok = comp[qi[ok], qj[ok]]
        out[ok] = cell_ok
        return out & qz

    return mask, (zb_lo, zb_hi)


PROTRU_MARG = 1500.0    # 轴向外延（外露筋段最长 ~1.25m；mask 的平面
                        # 足迹只覆盖混凝土，外露段必须靠轴窗容纳——
                        # 763mm 外露量测 bug 的根因）
CROSS_MARG = 105.0      # 横截面门半外扩（筋点截面内缩 ~45+翘曲 ~40+余量）
PROTRU_GAP = 250.0      # 端别守卫：种子伸出混凝土体 ≥250mm 才认定外露端
                        # （外露 400-1500、翘曲 σ≤40、端面糙度 ~10）
OCC_BIN = 20.0          # 轴向占据 bin（须小于最小缝隙 47mm，bin 骑缝会
                        # 切不开同轴堆叠——s3 m2 实锤 40mm bin 走穿）
OCC_CUT = 0.12          # 占据切断线（按"截断处 bin 占中位比"分签名：
                        # 真末端/47mm 垫块缝 ≈0-2%、s6 邻构件薄片 ≈6.5%
                        # →切；部分可见凹陷 19-24%（s4 m0 实锤带被截
                        # 4.7m）→过。0.25 时凹陷误切，0.10 以下薄片混入）
SEED_BOX_MIN = 300.0    # 种子自体 (u,v) 盒的最小双向跨度（单筋种子是
                        # 条线，达不到→缩选面带布局点盒源）
V2_DEBUG = False        # 双符号管线逐步打印（裁剪故障取证）

# ── 构件级候选择优（T-041①，用户 2026-10-29 拍板方向）─────────────────
COV_MIN = 0.9           # 种子覆盖率门：裁剪须罩住自家筋笼
EXCESS_MAX = 0.3        # 过剩钢门：裁剪钢点在种子包络盒外的占比
EXCESS_MARG = 100.0     # 包络盒外扩（截面 ±105 布局门同量级）


def _crop_quality(cand, pts, lbls, seed_all):
    """候选裁剪的部署可用质量（不碰 GT）：(faces_ok, cov, excess)。

    cov=种子并集点落入裁剪窗比例（罩不住自己=截断/错盒）；excess=
    裁剪类2钢点落在**种子包络盒**（t/u/v bbox+EXCESS_MARG）外的占比
    ——合并裁剪吞邻件钢筋（外构件钢在盒外）实锤 0.92 vs 干净
    0.00-0.21；不用点距 60mm 邻域：m4 预测标签稀疏、近半钢筋没进
    DBSCAN 簇，自家未聚类钢会被误计为外钢（0.35 假阳性）。"""
    from lib.e2e.face_pointgrow import extract_faces
    sub, lbls_sub, widx, xa, t_face, t_tip, n_cl = cand
    win = np.zeros(len(pts), bool)
    win[widx] = True
    gidx_u = np.unique(np.concatenate(seed_all))
    cov = float(win[gidx_u].mean()) if len(gidx_u) else 0.0
    st = sub[lbls_sub == 2]
    seed_xyz = pts[gidx_u]
    if len(st) == 0 or len(seed_xyz) == 0:
        excess = 1.0
    else:
        h = np.array([0.0, 0.0, 1.0]) if abs(float(xa[2])) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(h, xa)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(xa, e1)
        basis = np.vstack([xa, e1, e2]).astype(np.float64)
        c0 = seed_xyz.mean(0)
        qs = (seed_xyz - c0) @ basis.T
        qp = (st - c0) @ basis.T
        lo = qs.min(0) - EXCESS_MARG
        hi = qs.max(0) + EXCESS_MARG
        excess = float(((qp < lo) | (qp > hi)).any(1).mean())
    faces = None
    fr = member_frame(sub, xa)          # extract_faces 契约=构件系
    if fr is not None:
        faces = extract_faces(fr[0], lbls_sub, axis_hint=(1.0, 0.0, 0.0))
    return {"faces": faces, "cov": cov, "excess": excess}


def crop_member_best(pts, lbls, c, seed_list, gidx_list):
    """同构件多种子候选择优 claim（主入口，T-041①）。

    候选序列（先到先试，全过门即早退；门 = faces 提取成功 ∧
    cov≥COV_MIN ∧ excess≤EXCESS_MAX，全不门过按 (faces, −excess, cov)
    尽力取最优）：
    ① merged(默认路径)——committed 语义跑并集种子（≥2 种子时）；
    ② 逐种子(默认路径)——committed 语义（s3/s4/s6 已证明的配方）；
    ③ merged 变体 (seedbox)/(seedbox+union band)（直呼 v2）——m4
      半截盒 / m3 外露截带的救援路径。
    返回 (crop, faces, info) 或 None；crop 为 crop_one_member 元组。"""
    seed_all = [np.asarray(g, np.int64) for g in gidx_list]
    merged = np.concatenate(seed_list)
    trials = []
    if len(seed_list) >= 2:
        trials.append(("merged", merged, {}))
    trials += [(f"seed{i}", s, {}) for i, s in enumerate(seed_list)]
    trials += [("merged+seedbox", merged, {"box_from_seed": True}),
               ("merged+sb+union", merged,
                {"box_from_seed": True, "band_union": True})]
    best = None                   # (key tuple, tag, crop, q)
    for tag, seed, kw in trials:
        if kw:
            got = _crop_cluster_anchored(pts, lbls, seed, **kw)
        else:
            got = crop_one_member(pts, lbls, c, seed)
        if got is None:
            if tag == "merged+seedbox":
                break             # 种子 (u,v) 是条线，两变体都不适用
            continue
        q = _crop_quality(got, pts, lbls, seed_all)
        if q["faces"] is not None and q["cov"] >= COV_MIN \
                and q["excess"] <= EXCESS_MAX:
            return got, q["faces"], {"tag": tag, **{
                k: round(v, 3) for k, v in q.items() if k != "faces"}}
        key = (q["faces"] is not None, -q["excess"], q["cov"])
        if best is None or key > best[0]:
            best = (key, tag, got, q)
    if best is None:
        return None
    _key, tag, got, q = best
    return got, q["faces"], {"tag": tag, **{
        k: round(v, 3) for k, v in q.items() if k != "faces"}}

ST_EXT = 200.0          # 箍筋腿外扩上限（发夹腿=表面外伸 ≤140mm 杆径）
ST_EXT_MIN_N = 15       # 单侧界外类3钢点数门（防散噪误扩）
ST_EXT_GAP = 25.0       # 连续程间隙门（自身腿贴盒边连续；邻构件腿隔
                        # 着层间/平铺间隙 ~57mm+ 起步，s4 实锤被此门拒）


def _expand_box_stirrup(uq, vq, tq, lbls, box, t_lo, t_hi):
    """横截面盒向真实箍筋钢点方向外扩（T-041②，2026-10-29 实锤）。

    裁剪盒按混凝土分位（重力路径 ±40）/筋布局 ±105 取，会把从表面
    外伸的发夹腿切在裁剪外——s4 实锤：盒内类3钢点 s 最大 47.2（腿尖
    应 ~96，全量点云腿完整可见到 t≈100），腿缺失 → measure_height
    的 z_top 系统性偏低 ~50mm，场景高度 p50 82mm（v2.4 链 3.27，其
    裁剪=整件无盒）。向**类3钢点**方向扩，两道守卫：
    ①连续程——界外钢点须从盒边 ≤ST_EXT_GAP 起步、沿途无 >ST_EXT_GAP
      间隙（自身腿从面长出贴边连续；邻构件的腿隔着构件间隙起步，
      s4 实锤 v+ 侧 4321 点全属 m1、最近 ~97mm，被此门拒）；
    ②点数门——程内 ≥ST_EXT_MIN_N 点才认。扩幅=程末端 ∓10，上限
      ST_EXT。"""
    u_lo, u_hi, v_lo, v_hi = box
    st = (lbls == 3) & (tq > t_lo) & (tq < t_hi)
    if int(st.sum()) < ST_EXT_MIN_N:
        return box
    su, sv = uq[st], vq[st]
    for q, side in ((su, "u"), (sv, "v")):
        lo, hi = (u_lo, u_hi) if side == "u" else (v_lo, v_hi)
        ext_lo = _run_extent(lo - q[q < lo])          # 盒边→界外的距离
        if ext_lo is not None:
            lo = max(lo - ST_EXT, lo - ext_lo - 10.0)
        ext_hi = _run_extent(q[q > hi] - hi)
        if ext_hi is not None:
            hi = min(hi + ST_EXT, hi + ext_hi + 10.0)
        if side == "u":
            u_lo, u_hi = lo, hi
        else:
            v_lo, v_hi = lo, hi
    return u_lo, u_hi, v_lo, v_hi


def _run_extent(dist):
    """界外点距盒边的连续程长度：须 ≤ST_EXT_GAP 起步，首个 >ST_EXT_GAP
    间隙截断。非连续（邻构件腿）返回 None。"""
    d = np.sort(dist)
    if len(d) < ST_EXT_MIN_N or d[0] > ST_EXT_GAP:
        return None
    gap = np.flatnonzero(np.diff(d) > ST_EXT_GAP)
    return float(d[gap[0]] if len(gap) else d[-1])


def _crop_cluster_anchored(pts, lbls, seed, box_from_seed=False,
                           band_union=False):
    """筋簇锚定单构件裁剪 v2（M4 ①③：免重力缝隙分割）。

    旧重力缝隙分割（z 分带+平面连通域）两处原理性失效都出在"依赖
    混凝土自身成块"：接触堆叠无缝可切（s6 m2-m4 同裁剪）、pred 混凝土
    碎裂后选种落碎片（s4 m0）。v2 改从筋簇锚定：
    ①轴=粗门（筋簇线 ±750 横向）内混凝土两轮 SVD——体点占 5000:1，
      收敛到体轴，治种子轴 1-2° 倾斜（0.018×6000=108mm 漂移桥接缝）；
    ②横截面门=端面筋点布局 bbox±105mm——门来自筋点，与混凝土完整性
      无关（碎裂免疫）。种子簇是**单根筋**（DBSCAN eps=60 切不开布局
      间距 >60mm 的各根，s6 m0 实锤单筋种子把门缩成 210×210 角盒），
      布局 dots 必须从面带全量筋点重组：(u,v) DBSCAN(eps=200) 分组，
      取含种子的组（布局内根距 50-200 成组，邻构件面点错层分开）；
    ③轴向占据带=门内混凝土 t 直方图（40mm bin，<25% 中位=空）从端面
      向后走到第一个空档——47mm 垫块缝整 bin 落空可切（s3 不回归），
      接触堆叠的邻构件截面薄片（42mm/650≈6%）过不了 25% 门（s6 解放）；
    ④轴向定向=双符号各跑一遍②③取带长者——近场密度判别不成立：
      端对端平铺 47mm 缝后就是邻构件体（s4 实锤 ±240 窗两侧皆满），
      同轴堆叠筋尖穿进邻构件体（s3）。正确符号带=自身体长，错误符号
      从筋尖回走=外露段无混凝土带≈0（穿入情形带也长，由面带布局点数
      决胜：整布局面 ~16×150 点 >> 筋尖端几百）。
    原理性边界：共轴接触堆叠（无缝无错位）t 占据连续，仍会合并。

    T-041① 候选变体（默认关=逐位 committed 行为，由上层候选择优挑）：
    box_from_seed——横截面门改种子自体 (u,v) 外扩（全笼/多种子种子的
      (u,v) 即完整截面，与符号无关；单筋种子的 (u,v) 是条线，变体不
      适用返回 None）。m4-merged 实锤：面带最近组在共面邻柱布局交错
      下缩成半截盒（cov 0.29），种子盒全截面。
    band_union——占据带 ∪ 自家筋（低端面在筋尖侧时尖端到体之间无混
      凝土，m3 实锤纯混凝土带被外露区截成 1500 残段 → 裁剪只剩
      100mm 体段，faces=None）。"""
    from sklearn.cluster import DBSCAN
    con = pts[lbls == 1]
    if len(con) < 5000:
        return None
    cm = seed.mean(0)
    _, _, V = np.linalg.svd(seed - cm, full_matrices=False)
    ax = V[0]
    for _ in range(2):
        rel = con - cm
        tc = rel @ ax
        lat = np.linalg.norm(rel - np.outer(tc, ax), axis=1)
        g = (lat < 750.0) & (tc > -12000.0) & (tc < 12000.0)
        if int(g.sum()) < 5000:
            return None
        _, _, V = np.linalg.svd(con[g] - con[g].mean(0), full_matrices=False)
        ax = V[0]
    if abs(float(ax[2])) > 0.7:           # 立柱/劣质轴 → 交上层处理
        return None
    zt = np.array([0.0, 0.0, 1.0])
    za = zt - float(zt @ ax) * ax
    za /= np.linalg.norm(za)
    ya = np.cross(za, ax)
    b0 = con[(tc > -12000.0) & (tc < 12000.0)
             & (np.linalg.norm(rel - np.outer(tc, ax), axis=1)
                < 750.0)].mean(0)
    rel = pts - b0
    tq, uq, vq = rel @ ax, rel @ ya, rel @ za
    tcl = (seed - b0) @ ax
    lat_all = np.hypot(uq, vq)
    su_all, sv_all = (seed - b0) @ ya, (seed - b0) @ za
    if box_from_seed:
        # 变体盒=种子自体 (u,v)；单筋种子（双向跨度有一维 < 门限）不适
        if (float(su_all.max() - su_all.min()) < SEED_BOX_MIN
                or float(sv_all.max() - sv_all.min()) < SEED_BOX_MIN):
            return None

    best = None
    for s in (1.0, -1.0):
        T = tq * s                        # s 系深度（体在面后方=负向）
        TF = float(tcl.min() * s)
        if box_from_seed:
            du, dv = su_all, sv_all
        else:
            fb = (lbls == 2) & (np.abs(T - TF) < 80.0) & (lat_all < 2200.0)
            n_fb = int(fb.sum())
            if n_fb >= 4:
                uv = np.column_stack([uq[fb], vq[fb]])
                dcl = DBSCAN(eps=200.0, min_samples=3).fit(uv)
                ti = int(np.argmin(np.abs(T[fb] - TF)))
                sd = np.array([uq[fb][ti], vq[fb][ti]])
                gi = int(np.argmin(np.linalg.norm(uv - sd, axis=1)))
                gg = dcl.labels_[gi]
                if gg >= 0:
                    dsel = uv[dcl.labels_ == gg]
                else:                     # 全噪声组：退种子自带面点
                    dd = seed[np.abs(tcl * s - TF) < 40.0]
                    dsel = np.column_stack([(dd - b0) @ ya, (dd - b0) @ za])
                du, dv = dsel[:, 0], dsel[:, 1]
            else:
                dd = seed[np.abs(tcl * s - TF) < 40.0]
                du, dv = (dd - b0) @ ya, (dd - b0) @ za
        if len(du) == 0:
            continue
        u_lo = float(np.min(du)) - CROSS_MARG
        u_hi = float(np.max(du)) + CROSS_MARG
        v_lo = float(np.min(dv)) - CROSS_MARG
        v_hi = float(np.max(dv)) + CROSS_MARG
        inm = (lbls == 1) & (uq > u_lo) & (uq < u_hi) & (vq > v_lo)             & (vq < v_hi)
        tc_in = T[inm]
        tc_in = tc_in[tc_in < TF]
        if len(tc_in) < 3000:
            continue
        edges = np.linspace(TF - 10500.0, TF, int(10500.0 / OCC_BIN) + 1)
        hist, _ = np.histogram(tc_in, bins=edges)
        med = float(np.median(hist[hist > 0]))
        occ = hist > OCC_CUT * med
        if band_union:
            sinm = (tcl * s > edges[0]) & (tcl * s < TF)                 & (su_all > u_lo) & (su_all < u_hi)                 & (sv_all > v_lo) & (sv_all < v_hi)
            shist, _ = np.histogram(tcl[sinm] * s, bins=edges)
            occ = occ | (shist > 0)
        bi = len(hist) - 1
        while bi >= 0 and occ[bi]:
            bi -= 1
        t_lo = float(edges[bi + 1])
        band = TF - t_lo
        if V2_DEBUG:
            print(f'    [v2 s={s:+.0f} b={"S" if box_from_seed else "d"}'
                  f'{"u" if band_union else "-"}] TF={TF:.0f} '
                  f'box u[{u_lo:.0f},{u_hi:.0f}]v[{v_lo:.0f},{v_hi:.0f}] '
                  f'门内con={len(tc_in)} band={band:.0f} med={med:.0f}',
                  flush=True)
        key = (1 if band >= 800.0 else 0, n_fb if not box_from_seed else 0,
               band)
        if best is None or key > best[0]:
            best = (key, s, TF, t_lo, (u_lo, u_hi, v_lo, v_hi))
    if best is None or best[0][0] == 0:   # 双符号都无有效占据带
        return None
    _key, s, TF, t_lo, (u_lo, u_hi, v_lo, v_hi) = best
    u_lo, u_hi, v_lo, v_hi = _expand_box_stirrup(
        uq, vq, tq, lbls, (u_lo, u_hi, v_lo, v_hi),
        t_lo, TF + PROTRU_MARG)
    T = tq * s
    win = (T > t_lo - 60.0) & (T < TF + PROTRU_MARG)         & (uq > u_lo) & (uq < u_hi) & (vq > v_lo) & (vq < v_hi)
    if win.sum() < 20000:
        return None
    rb_m = win & (lbls == 2)
    if int(rb_m.sum()) < MIN_CLUSTER_N:
        return None
    bm = win & (lbls == 1) & (T > t_lo) & (T < TF)
    _, _, V = np.linalg.svd(pts[bm] - pts[bm].mean(0), full_matrices=False)
    xa = V[0]
    if xa[int(np.argmax(np.abs(xa)))] < 0:
        xa = -xa
    if abs(float(xa[2])) > 0.7:
        return None
    return pts[win], lbls[win], np.flatnonzero(win), xa, t_lo,         TF + PROTRU_MARG, int(rb_m.sum())


def _crop_gravity(pts, lbls, seg_mask):
    """重力缝隙分割裁剪（老路径主体，s3 4/4 满分的已证明配方）。"""
    body_m = seg_mask(pts) & (lbls == 1)
    body = pts[body_m]
    if len(body) < 5000:
        return None
    b0 = body.mean(0)
    _, _, V = np.linalg.svd(body - b0, full_matrices=False)
    xa = V[0]
    if xa[int(np.argmax(np.abs(xa)))] < 0:
        xa = -xa
    if abs(float(xa[2])) > 0.7:                  # 立柱/劣质盒 → 交上层处理
        return None
    zt = np.array([0.0, 0.0, 1.0])
    za = zt - float(zt @ xa) * xa
    za /= np.linalg.norm(za)
    ya = np.cross(za, xa)
    tb, ub, vb = (body - b0) @ xa, (body - b0) @ ya, (body - b0) @ za
    t_lo, t_hi = np.percentile(tb, [0.5, 99.5])
    u_lo, u_hi = np.percentile(ub, [0.5, 99.5])
    v_lo, v_hi = np.percentile(vb, [0.5, 99.5])
    MARG = 40.0          # 横截面外扩（保护层外的筋/量化斜边）
    tq = (pts - b0) @ xa
    uq = (pts - b0) @ ya
    vq = (pts - b0) @ za
    u_lo, u_hi, v_lo, v_hi = _expand_box_stirrup(
        uq, vq, tq, lbls,
        (u_lo - MARG, u_hi + MARG, v_lo - MARG, v_hi + MARG),
        t_lo - MEMBER_BACK, t_hi + PROTRU_MARG)
    win = (tq > t_lo - MEMBER_BACK) & (tq < t_hi + PROTRU_MARG) \
        & (uq > u_lo) & (uq < u_hi) \
        & (vq > v_lo) & (vq < v_hi)
    if win.sum() < 20000:
        return None
    rb_m = win & (lbls == 2)
    if int(rb_m.sum()) < MIN_CLUSTER_N:
        return None
    t_face = t_lo
    t_tip = t_hi
    return pts[win], lbls[win], np.flatnonzero(win), xa, t_face, t_tip, \
        int(rb_m.sum())


def crop_one_member(pts, lbls, c, seed_cluster, seg_axis=None):
    """单构件裁剪：重力缝隙分割（主）→ 筋簇锚定 v2（兜底）。

    重力分割对有缝工况是已证明路径（s3 4/4 满分），主位不动；只在
    ①z 带吞并邻构件层（接触堆叠无缝：带高 >> 种子层高，s6 m2-m4 实锤
    2100 vs ~600）或 ②自身失败（碎片选种 s4 m0）时交 v2。v2=筋簇锚定
    （轴两轮 SVD / 布局重组门 / 双符号占据带，见 _crop_cluster_anchored）。
    返回 (sub, lbls_sub, win_idx, xa, t_face, t_tip, n_rebar) 或 None。"""
    seg_mask, band = _member_segment_mask(pts, lbls, c,
                                          seed_cluster.mean(0))
    degenerate = False
    if seg_mask is not None:
        zc = seed_cluster[:, 2]
        if (band[1] - band[0]) > float(zc.max() - zc.min()) + 900.0:
            degenerate = True      # z 带吞并邻层 → 缝隙分割失效
    if seg_mask is not None and not degenerate:
        got = _crop_gravity(pts, lbls, seg_mask)
        if got is not None:
            return got
    got = _crop_cluster_anchored(pts, lbls, seed_cluster)
    if got is not None:
        return got
    if seg_mask is not None:       # v2 也失败 → 老路径尽力
        return _crop_gravity(pts, lbls, seg_mask)
    return None


def member_frame(crop, xa):
    """轴对齐质心系（x=轴, z=重力）→ (p, R, c0)；立柱返回 None。"""
    zt = np.array([0.0, 0.0, 1.0])
    za = zt - float(zt @ xa) * xa
    nz = float(np.linalg.norm(za))
    if nz < 0.5:                       # 轴≈重力（立柱）：本切片不处理
        return None
    za /= nz
    ya = np.cross(za, xa)
    R = np.vstack([xa, ya, za])
    c0 = crop.mean(0)
    return ((crop - c0) @ R.T).astype(np.float32), R, c0


def member_meta(mt, R, c0):
    """member_table 条目（世界系）→ eval_one 所需 meta（构件质心系）。"""
    def xform(pl):
        return (np.asarray(pl, np.float64) - c0) @ R.T

    rbs = mt.get("rebar_gt") or []
    p0s = xform([g["p0"] for g in rbs]) if rbs else np.zeros((0, 3))
    p1s = xform([g["p1"] for g in rbs]) if rbs else np.zeros((0, 3))
    sls = mt.get("sleeve_gt") or []
    slc = xform([g["center"] for g in sls]) if sls else np.zeros((0, 3))
    hps = mt.get("hairpin_gt") or []
    hpl = xform([g["foot_l"] for g in hps]) if hps else np.zeros((0, 3))
    hpr = xform([g["foot_r"] for g in hps]) if hps else np.zeros((0, 3))
    end_x = float(p0s[:, 0].mean()) if len(p0s) else 0.0
    sl_x = float(slc[:, 0].mean()) if len(slc) else end_x - 1000.0
    return {
        "rebar_positions_yz": p0s[:, 1:].astype(np.float32),
        "rebar_gt": [{"p1": p} for p in p1s],
        "rebar_end_x": end_x,
        "sleeve_end_x": sl_x,
        "rebar_diam": mt.get("rebar_diam") or 16.0,
        "n_rebars": int(mt.get("n_rebars", len(rbs))),
        "sleeve_gt": [{"center_x": float(slc[i, 0]),
                       "center_yz": [float(slc[i, 1]), float(slc[i, 2])],
                       "inner_diam": g.get("inner_diam", 40.0),
                       "outer_diam": g.get("outer_diam", 60.0)}
                      for i, g in enumerate(sls)],
        "hairpin_gt": [{"p_foot_l": hpl[i].tolist(),
                        "p_foot_r": hpr[i].tolist(),
                        "height": g.get("height"), "diam": g.get("diam")}
                       for i, g in enumerate(hps)],
    }


def _fix_end_flip(faces, p, lbls, seed_xyz, c0, xa):
    """端别翻转守卫（T-041①，m4 实锤）：extract_faces 按 n_rb（端面
    外向 (0,300) 薄层钢点数）排出筋/套筒端，两处场景病会把它翻反——
    ①垫片（混凝土类）贴在套筒端面前 → _outward 把外向翻进体内，薄层
    采到的是嵌入筋（布局网格，量大）；②预测标签稀疏（m4 近半钢筋没进
    簇）→ 真出筋面的外露筋计数偏低。守卫判据（部署可用，只用种子+
    裁剪）：**种子轴向范围超出混凝土体的一侧=外露端**（出筋端筋尖伸
    出体外 400-1500mm；套筒端种子与体端齐平）。超出量 ≥PROTRU_GAP
    才纠正（两端齐平/种子缺尖时不动，回退 n_rb 原判定）。"""
    con_m = lbls == 1
    if seed_xyz is None or int(con_m.sum()) < 100:
        return faces
    ts = (seed_xyz - c0) @ xa
    tc = (p[con_m] - c0) @ xa
    c_lo, c_hi = np.percentile(tc, [0.5, 99.5])
    gap_lo = float(c_lo - ts.min())       # 种子伸出体低端的长
    gap_hi = float(ts.max() - c_hi)       # 种子伸出体高端的长
    if max(gap_lo, gap_hi) < PROTRU_GAP:
        return faces
    tip_sign = -1.0 if gap_lo > gap_hi else 1.0   # 外露端在构件系 ±x
    nrb = faces["rebar"]["n3"]
    if float(nrb[0]) * tip_sign < 0:      # 现判出筋端在非外露侧 → 互换
        faces = dict(faces)
        faces["rebar"], faces["sleeve"] = faces["sleeve"], faces["rebar"]
        faces["axis"] = faces["rebar"]["n3"]
        print(f"  [端别守卫] 翻转（外露 gap {max(gap_lo, gap_hi):.0f}mm）",
              flush=True)
    return faces


def _sleeve_visible_face(p, lbls, meta):
    """套筒可见性（面环带口径，T-041①）：套筒盘域（平面距≤外径/2+30、
    |x−cx|≤40mm=环壁轴向带宽）内混凝土点 ≥30 才算可见。比 v2.4 链的
    柱域口径（|x−cx|≤80）紧——堆叠工况套筒端面被邻构件挡掉时，柱域
    里仍有体混凝土（m3 实锤误判可见），面环带才是融合真实依赖的点。"""
    conc = lbls == 1
    out = []
    for g in meta.get("sleeve_gt") or []:
        cx = float(g["center_x"])
        cy, cz = float(g["center_yz"][0]), float(g["center_yz"][1])
        r = float(g.get("outer_diam", 60.0)) / 2 + 30.0
        m = (conc & (np.abs(p[:, 0] - cx) <= 40.0)
             & ((p[:, 1] - cy) ** 2 + (p[:, 2] - cz) ** 2 <= r * r))
        out.append(bool(int(m.sum()) >= 30))
    return out


def run_member(sub, lbls_sub, mt, models, device, xa, sl_fuse=True,
               no_roll=False, faces0=None, seed_xyz=None):
    """单构件：轴对齐 → 生长面 → 箍筋面认证 → 三头 → eval_one dict。

    sub 已是单构件（main.crop_member_best 产物），不再二次裁剪；
    faces0 可传入候选择优时已算好的 extract_faces 结果（免重算）；
    seed_xyz=该构件种子并集（世界系），用于端别翻转守卫。
    箍筋面认证（M4 ②，用户 2026-10-29 定线）：extract_faces 的箍筋面
    候选=钢点薄层计数最大的面（stface）；侧面钢点 ≥500 且 >2.5×顶面才
    认证。认证通过→faces["top"] 换成该面（带法向），箍筋视图按面法向
    参数化（top_view n3 参数）——构件系全程保持重力系，正立件逐位不变，
    rb/sl 链零风险（BEV 峰视图对旋转敏感，动系会掉环：m1-s4 实锤
    8/8→3/8）。未认证=扫描/分割上游问题（低于 500 线），回默认顶面并
    留痕 st_cert。"""
    from lib.e2e.face_pointgrow import extract_faces
    from lib.pipeline.single_chain import eval_one
    fr = member_frame(sub, xa)
    if fr is None:
        return {"skip": True, "reason": "vertical_member"}
    p0, R0, c00 = fr
    if faces0 is None:
        faces0 = extract_faces(p0, lbls_sub, axis_hint=(1.0, 0.0, 0.0))
    if faces0 is None:
        return {"skip": True, "reason": "face_fail"}
    faces0 = _fix_end_flip(faces0, p0, lbls_sub, seed_xyz, c00, xa)
    stf = faces0.get("stface")
    # 箍筋面认证（用户 2026-10-29 定线）：候选侧面钢点 ≥500 且
    # >2.5× 顶面才认证为箍筋面。低于线=扫描/分割上游问题（场景真箍
    # 筋面 GT 仅 146-1405 点，噪音案例 38-212），不硬测，留痕 st_cert。
    cert = (stf is not None and stf["kind"] == "side"
            and stf["n_st"] >= 500.0
            and stf["n_st"] > 2.5 * float(stf.get("top_n_st") or 0.0))
    p, R, c0 = p0, R0, c00
    faces = faces0
    if cert and not no_roll:
        faces = dict(faces0)
        faces["top"] = {"n3": np.asarray(stf["n3"], np.float64),
                        "d": float(stf["d"])}
        print(f"  [认证] 箍筋面=侧面 n3={faces['top']['n3'].round(2)} "
              f"钢点 {stf['n_st']}/顶 {stf.get('top_n_st')}", flush=True)
    x_rb = float(faces["rebar"]["c3"][0])
    x_sl = float(faces["sleeve"]["c3"][0])
    m1 = (lbls_sub == 2) | ((lbls_sub == 1) & (np.abs(p[:, 0] - x_rb) < 60.0))
    m2 = (lbls_sub == 1) & (np.abs(p[:, 0] - x_sl) < 60.0)
    top = faces.get("top")
    m3 = None
    if top is not None:
        n3t = np.asarray(top.get("n3", (0.0, 0.0, 1.0)), np.float64)
        nn = n3t[1:] / max(float(np.linalg.norm(n3t[1:])), 1e-9)
        s = p[:, 1] * nn[0] + p[:, 2] * nn[1] - float(top["d"])
        m3 = ((lbls_sub == 1) | (lbls_sub == 3)) & (s > -60.0)
    res = run_one(p, None, faces, models, device, face_pts=(m1, m2, m3),
                  conc_mask=(lbls_sub == 1) | (lbls_sub == 2),
                  sl_fuse=sl_fuse, rbst="point",
                  st_steel=(lbls_sub == 3))
    if res is None or res.get("face_fail"):
        return {"skip": True, "reason": "face_fail"}
    meta = member_meta(mt, R, c0)
    if not len(meta["rebar_positions_yz"]):
        return {"skip": True, "reason": "no_rebar_gt"}
    # 诊断量：端别/检出数/检出-GT 间距分位（m3 0/14 类故障的定位针）
    det_rb = np.asarray(res["rebar"]["uv_world"], np.float32).reshape(-1, 2)
    gt_rb = meta["rebar_positions_yz"]
    if len(det_rb) and len(gt_rb):
        D = np.linalg.norm(det_rb[:, None, :] - gt_rb[None, :, :], axis=2)
        near = D.min(1)
        print(f"  [诊断] end_pick={res['rebar']['end_pick']} "
              f"sl_end={res['sleeve']['end_pick']} rb检出={len(det_rb)} "
              f"GT={len(gt_rb)} 检出→GT最近距 p50="
              f"{float(np.median(near)):.0f}/p90={float(np.percentile(near, 90)):.0f}mm",
              flush=True)
    # 套筒可见性分母（T-040 遗留口径缺口，T-041①启用）：堆叠工况端面
    # 被邻构件/垫块挡掉一半是常态（m2 实锤 GT8-15 面带内点=0，检出环
    # 全部毫米级匹配）——不可见套筒不计入召回分母，与 v2.4 链同口径。
    ev = eval_one(res, meta, sl_vis=_sleeve_visible_face(p0, lbls_sub, meta),
                  st_off=(0.0, 0.0))
    ev["rb_gt"] = int(meta["n_rebars"])
    ev["st_cert"] = {"certified": bool(cert and not no_roll),
                     "n_st": int(stf["n_st"]) if stf else 0,
                     "top_n_st": (int(stf.get("top_n_st") or 0)
                                  if stf else 0)}
    return {"eval": ev, "res": {"rb_n": res["rebar"]["n"],
                                "sl_n": res["sleeve"]["n"],
                                "exposed_mm": round(float(
                                    meta["rebar_gt"][0]["p1"][0]
                                    - meta["rebar_end_x"]), 0)}}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="场景 npz 路径")
    ap.add_argument("--ckpt", default="out/exp1_train/head_bev/best.pth")
    ap.add_argument("--pt-ckpt", default="out/exp1_train/head_point/best.pth")
    ap.add_argument("--seg-ckpt", default="")
    ap.add_argument("--limit-members", type=int, default=0)
    ap.add_argument("--no-roll", action="store_true",
                    help="禁用滚转（箍筋面择向）消融")
    ap.add_argument("--refresh-labels", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--label-cache-dir", default="out/cache/scene_batch",
                    help="分割标签缓存目录（命中则跳过分割，确定性保证）")
    return ap


def run_scene(args):
    """单场景全链（原 main 本体，参数化）。返回 summary。"""
    from lib.segmentation.inference import load_model, predict_labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(anchor_project_path(args.scene), allow_pickle=True)
    vis = z["visible"].astype(bool)
    pts = z["points"][vis].astype(np.float32)
    member_id = z["member_id"][vis].astype(np.int64) if "member_id" \
        in z.files else None
    mts = json.loads(str(z["member_table_json"])) if "member_table_json" \
        in z.files else []
    print(f"[场景] {args.scene}：{len(pts)} 点，{len(mts)} 构件 GT",
          flush=True)

    out_path = anchor_project_path(args.out or "out/scene_fullchain_smoke.json")
    lbl_dir = anchor_project_path(args.label_cache_dir)
    os.makedirs(lbl_dir, exist_ok=True)
    lbl_cache = os.path.join(
        lbl_dir, os.path.basename(out_path).replace(".json", "_labels.npz"))
    if os.path.exists(lbl_cache) and not args.refresh_labels:
        lbls = np.load(lbl_cache)["lbls"]
        print(f"[分割] 读缓存 {lbl_cache}", flush=True)
    else:
        seg_model, seg_dev = load_model(args.seg_ckpt or None)
        t0 = time.time()
        lbls = predict_labels(pts, seg_model, seg_dev)
        print(f"[分割] {time.time()-t0:.0f}s，类分布 "
              f"{np.bincount(lbls, minlength=4)}", flush=True)
        np.savez(lbl_cache, lbls=lbls)

    models = load_heads(args.ckpt, args.pt_ckpt, device)
    # 场景全局轴（分割/轴窗专用）：全量 pred 混凝土 SVD——种子簇轴带
    # 1-2° 倾斜会在场景尺度把垫块缝桥死（0.018×6000mm=108mm 漂移实锤）
    _con = pts[lbls == 1]
    _cc = _con.mean(0)
    _, _, _V = np.linalg.svd(_con[::5] - _cc, full_matrices=False)
    seg_axis = _V[0]
    if seg_axis[int(np.argmax(np.abs(seg_axis)))] < 0:
        seg_axis = -seg_axis
    # 筋簇分组（预测）→ 逐构件种子。归属=种子簇自身的 member_id 多数票
    # （一根筋只属于一个构件；旧"整窗多数票"在堆叠场景把窗内高邻居
    # 投成归属，m0/m2 永远输给 m1/m3——2026-10-29 实锤 17 簇 13 归 m1）
    from sklearn.cluster import DBSCAN
    rb_idx = np.flatnonzero(lbls == 2)
    rb = pts[rb_idx]
    cl = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(rb)
    seeds, seed_gidx = [], []
    for k in np.unique(cl.labels_):
        if k >= 0 and (cl.labels_ == k).sum() >= MIN_CLUSTER_N:
            gidx = rb_idx[cl.labels_ == k]
            seeds.append(pts[gidx])
            seed_gidx.append(gidx)
    print(f"[分簇] 预测筋簇 {len(seeds)} 个，seg_axis={seg_axis.round(3)}",
          flush=True)

    # 按构件归组（T-041①：同构件多种子候选择优 claim，替代旧的
    # DBSCAN label 顺序先到先得——首种子 claim 到合并/碎片裁剪时整件
    # 报废，s6 m2-m4 实锤）
    by_mid = {}
    for si, (seed, gidx) in enumerate(zip(seeds, seed_gidx)):
        gv = member_id[gidx]
        gv = gv[gv >= 0]
        if not len(gv):
            print(f"[seed{si}] 种子簇无 GT 归属（背景噪声簇）", flush=True)
            continue
        mid = int(np.bincount(gv, minlength=len(mts)).argmax())
        by_mid.setdefault(mid, []).append((si, seed, gidx))

    per_file, used_gt = [], []
    for mid in sorted(by_mid):
        if args.limit_members and len(used_gt) >= args.limit_members:
            break
        lst = by_mid[mid]
        tags = ",".join(f"seed{si}" for si, _s, _g in lst)
        got = crop_member_best(pts, lbls, pts.mean(0),
                               [s for _si, s, _g in lst],
                               [g for _si, _s, g in lst])
        if got is None:
            print(f"[构件 m{mid}] crop 失败（{tags}）", flush=True)
            continue
        (sub, lbls_sub, win_idx, xa, t_face, t_tip, n_cl), faces0, info = got
        used_gt.append(mid)
        t0 = time.time()
        r = run_member(sub, lbls_sub, mts[mid], models, device, xa,
                       no_roll=args.no_roll, faces0=faces0,
                       seed_xyz=pts[np.unique(np.concatenate(
                           [g for _si, _s, g in lst]))])
        dt = time.time() - t0
        if r is None:
            per_file.append({"file": f"m{mid}", "skip": True,
                             "reason": "unknown"})
            continue
        if r.get("skip"):
            per_file.append({"file": f"m{mid}", "skip": True,
                             "reason": r["reason"],
                             "crop": info})
            print(f"[构件 m{mid}] skip={r['reason']} crop={info}",
                  flush=True)
            continue
        r["file"] = f"m{mid}"
        r["sec"] = round(dt, 1)
        r["crop"] = info
        per_file.append(r)
        ev = r["eval"]
        print(f"[构件 m{mid}] crop={info['tag']} 簇 {n_cl} "
              f"外露 {r['res']['exposed_mm']}mm "
              f"rb {ev['rb_match']}/{ev['rb_gt']} sl {ev['sl_match']}/"
              f"{ev['sl_total_gt']} st {ev['st_match']}/{ev['st_total_gt']}"
              f"（{dt:.0f}s）", flush=True)
    agg = aggregate(per_file)
    summary = {"scene": args.scene, "aggregate": agg,
               "构件GT数": len(mts), "覆盖构件数": len(used_gt),
               "筋簇种子": len(seeds), "per_file": per_file}
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fo:
            json.dump(summary, fo, ensure_ascii=False, indent=1,
                      default=str)
        print(f"→ {out_path}")
    return summary


def main():
    run_scene(build_parser().parse_args())


if __name__ == "__main__":
    main()
