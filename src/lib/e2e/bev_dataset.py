# -*- coding: utf-8 -*-
"""BEV 柱化数据集（PointPillars 柱化 + CenterNet 热图）。

三个裁剪视图（坐标系沿用 crop_dataset 的端面原点约定，绝对尺度 mm）：
- 钢筋端 (t,u,v)：t∈(-300,1750)，外露朝 +t，热图峰=每根纵筋；
- 套筒端 (t,u,v)：t∈(-300,80)，热图峰=每个套筒环；
- 箍筋顶面 (x,y,z)：x∈[发夹筋范围±200]，|y-yc|<512，z>top_z-50，
  热图峰=每条腿（角点定位，同箍两腿 x 相同配对）。

设计定案（2026-10-23 讨论）：
- 网络只做检测：热图（根数=峰数）+ 亚格偏移（位置）+ 规格分类（直径/内径档）；
- 连续量全部几何量测、不走网络：外露长度=检出 (u,v) 柱查询的 t 顶端
  （基准=裁剪原点端面），箍筋高度=腿柱 z 顶端 − 邻域表面基准；
- 直径/内径保留分类头：可见弧半径是名义半径 0.45-1.0 随机因子（T-031
  教训），可见尺寸与规格档范围重叠，几何分不出规格。

增广（保尺度，pillarize 之前施加）：截面内旋转/翻转、随机丢点、jitter。
"""

import json
import os
import zlib

import numpy as np

from lib.e2e.crop_dataset import (REBAR_WIN, SLEEVE_WIN, _end_face_center,
                                  load_split)

CELL = 8.0
# 固定网格（mm 原点相对量）：端面网格覆盖最大截面（T 梁 1000 宽）+ 余量；
# 箍筋顶面网格 3584×1024mm 覆盖 0.85L+400 窗（L≤4000）
END_ORIGIN = (-608.0, -608.0)
END_WH = (152, 152)
ST_ORIGIN = (0.0, -512.0)  # 原点=裁剪窗 (x0, yc)，缓存时一致使用
ST_WH = (448, 128)  # 3584×1024mm 覆盖 0.85L+400 窗（L≤4000）

REBAR_CAP = 150_000   # 缓存时随机下采样上限（保密度比例，控制柱化耗时；
SLEEVE_CAP = 80_000   # 8mm 格平均 ~9 点，足够统计；纯提速无信息损失）
STIRRUP_CAP = 150_000

EXPOSED_MIN = 300.0   # 合成 min_exposed 分治下限（T-032 口径）


# ── 裁剪样本构造 ────────────────────────────────────────────────────────
def _face_points(pts, lbls, meta):
    """面类点集（face_band_labels 单一事实源）：返回三类掩码 (m1,m2,m3)。"""
    from lib.e2e.face_sem import face_band_labels
    lab = face_band_labels(pts, lbls, meta)
    return lab == 1, lab == 2, lab == 3


def make_rebar_crop(pts, lbls, meta, end="rebar", clean=True):
    """纵筋面视图点云 + GT：峰 (u,v)、根数、外露（构件级标量）、直径。

    (u,v) 以端面中心为原点（端面带混凝土点中位数，与训练网格中心一致）。
    v6d 起点集=纵筋面类（面片+外露筋全段+垫片——"面片带梳子一体"，
    用户 2026-10-24 对账），窗仅作深度 t 的采样范围。
    end="sleeve" 时取套筒面类点按同一 t 范围——钢筋头交叉负样本视图。
    """
    end_x = float(meta[f"{end}_end_x"])
    other = "sleeve" if end == "rebar" else "rebar"
    flip = -1.0 if float(meta[f"{other}_end_x"]) > end_x else 1.0
    t = flip * (pts[:, 0] - end_x)
    cy, cz = _end_face_center(pts, lbls, end_x)
    m = (t > REBAR_WIN[0]) & (t < REBAR_WIN[1])
    if clean:
        m1, m2, _ = _face_points(pts, lbls, meta)
        m = m & (m2 if end == "sleeve" else m1)
    p = np.column_stack([t[m], pts[m, 1] - cy, pts[m, 2] - cz])
    if end != "rebar":
        return p, np.zeros((0, 2), np.float32), 0, 0.0, 0.0
    uv = np.asarray(meta["rebar_positions_yz"], dtype=np.float32)
    uv = np.column_stack([uv[:, 0] - cy, uv[:, 1] - cz])
    expo = flip * (float(meta["rebar_gt"][0]["p1"][0]) - end_x)
    return p, uv, int(meta["n_rebars"]), expo, float(meta["rebar_diam"])


def make_sleeve_crop(pts, lbls, meta, end="sleeve", clean=True):
    """套筒面视图点云 + GT：峰 (u,v)、环数、内径。

    end="rebar" 时取纵筋面类点按同一 t 范围（梳齿根部+端面片）——
    套筒头交叉负样本视图（热图应全 0）。
    """
    end_x = float(meta[f"{end}_end_x"])
    other = "sleeve" if end == "rebar" else "rebar"
    flip = -1.0 if float(meta[f"{other}_end_x"]) > end_x else 1.0
    t = flip * (pts[:, 0] - end_x)
    cy, cz = _end_face_center(pts, lbls, end_x)
    m = (t > SLEEVE_WIN[0]) & (t < SLEEVE_WIN[1])
    if clean:
        m1, m2, _ = _face_points(pts, lbls, meta)
        m = m & (m1 if end == "rebar" else m2)
    p = np.column_stack([t[m], pts[m, 1] - cy, pts[m, 2] - cz])
    if end != "sleeve":
        return p, np.zeros((0, 3), np.float32), 0
    gt = np.array([[g["center_yz"][0] - cy, g["center_yz"][1] - cz,
                    float(g["inner_diam"])] for g in meta["sleeve_gt"]],
                  dtype=np.float32)
    return p, gt, len(gt)


def make_stirrup_crop(pts, meta, lbls=None):
    """箍筋面（顶面片+发夹）视图点云 + GT：每箍 (x_l,y_l,x_r,y_r,h,diam)。

    腿脚锚点用生成器翘曲同步后的 p_foot_l/r（builder 已写入 hairpin_gt）。
    lbls 给定时点集=箍筋面类（v6d 干净口径）。返回 None=非叠合构件。
    """
    hps = meta.get("hairpin_gt") or []
    if not hps:
        return None
    x0 = min(float(g["p_foot_l"][0]) for g in hps) - 200.0
    y1s = [float(g["p_foot_l"][1]) for g in hps]
    y2s = [float(g["p_foot_r"][1]) for g in hps]
    yc = (min(y1s) + max(y2s)) / 2.0
    top_z = float(hps[0]["top_z"])
    # v2.3（P-043 修复）：x 窗口按发夹实际范围自适应——发夹排实测最长
    # 6.7m，旧固定 3584mm 使 20/24 件有腿 GT 落窗外无点（评估分母失真）。
    x_hi = max(max(float(g["p_foot_r"][0]) for g in hps),
               max(float(g["p_foot_l"][0]) for g in hps)) + 200.0
    span = max(ST_WH[0] * CELL, x_hi - x0)
    win = ((pts[:, 0] > x0) & (pts[:, 0] < x0 + span)
           & (np.abs(pts[:, 1] - yc) < ST_WH[1] * CELL / 2)
           & (pts[:, 2] > top_z - 50.0))
    if lbls is not None:
        _, _, m3 = _face_points(pts, lbls, meta)
        win = win & m3
    p = pts[win] - np.array([x0, yc, 0.0], dtype=np.float32)
    gt = np.array([[float(g["p_foot_l"][0]) - x0, float(g["p_foot_l"][1]) - yc,
                    float(g["p_foot_r"][0]) - x0, float(g["p_foot_r"][1]) - yc,
                    float(g["height"]), float(g["diam"])] for g in hps],
                  dtype=np.float32)
    return p, gt, top_z


# ── 柱化与 GT 编码 ──────────────────────────────────────────────────────
def pillarize(p, iu, iv, idd, origin, wh, d_shift=0.0, cell=CELL,
              d_scale=1000.0, ext_from=None):
    """(N,3) → (7或8,H,W)：[logcnt, d_min, d_max, d_mean, d_std, u_off,
    v_off(, ext)]。

    d 为池化轴坐标（d_scale 归一：端面 t/1000、世界系 z/100——量级不同）；
    u/v_off 为格内均值相对格心的偏移（亚格信息）。空格全零。
    ext_from 给定时加第 8 通道 ext=clip(d_max − ext_from, 0)（归一单位）
    ——外露长度先验（用户 2026-10-24 提案）：端面视图里只有外露筋能
    延伸超过端面内侧厚度，显式编码弥补密度柱特征不鲜明（bev_v6b 诊断
    真峰/假峰分数混居的解法之一，与 v6c 干净视图互补）。
    """
    W, H = wh
    u = p[:, iu] - origin[0]
    v = p[:, iv] - origin[1]
    d = (p[:, idd] - d_shift).astype(np.float64) / d_scale
    gx = np.floor(u / cell).astype(np.int64)
    gy = np.floor(v / cell).astype(np.int64)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    nch = 7 if ext_from is None else 8
    feat = np.zeros((nch, H, W), dtype=np.float32)
    if not ok.any():
        return feat
    gx, gy = gx[ok], gy[ok]
    u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok]
    flat = gy * W + gx
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    uniq, start = np.unique(fs, return_index=True)
    cnt = np.diff(np.append(start, len(fs))).astype(np.float64)
    dso, uo, vo = d[order], u[order], v[order]
    dmean = np.add.reduceat(dso, start) / cnt
    dvar = np.add.reduceat(dso ** 2, start) / cnt - dmean ** 2
    cy, cx = uniq // W, uniq % W
    feat[0, cy, cx] = np.log1p(cnt) / np.log1p(64.0)
    feat[1, cy, cx] = np.minimum.reduceat(dso, start)
    feat[2, cy, cx] = np.maximum.reduceat(dso, start)
    feat[3, cy, cx] = dmean
    feat[4, cy, cx] = np.sqrt(np.maximum(dvar, 0))
    feat[5, cy, cx] = np.add.reduceat(uo, start) / cnt - (cx + 0.5) * cell
    feat[6, cy, cx] = np.add.reduceat(vo, start) / cnt - (cy + 0.5) * cell
    feat[5:7] /= cell
    if ext_from is not None:
        feat[7, cy, cx] = np.clip(
            np.maximum.reduceat(dso, start) - ext_from / d_scale, 0.0, None)
    return feat


def encode_gt(uvs, origin, wh, cell=CELL, spec_ids=None, sigma=1.2):
    """CenterNet GT：高斯热图（亚格中心）+ 偏移 + 规格类别（仅峰格）。"""
    W, H = wh
    heat = np.zeros((H, W), dtype=np.float32)
    off = np.zeros((H, W, 2), dtype=np.float32)
    cls = np.full((H, W), -1, dtype=np.int64)
    for k, (u, v) in enumerate(np.asarray(uvs, dtype=np.float64)):
        cu = (u - origin[0]) / cell
        cv = (v - origin[1]) / cell
        cx, cy = int(np.floor(cu)), int(np.floor(cv))
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        off[cy, cx] = (cu - cx, cv - cy)
        cls[cy, cx] = 0 if spec_ids is None else int(spec_ids[k])
        r = int(np.ceil(3 * sigma))
        x0, x1 = max(cx - r, 0), min(cx + r + 1, W)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, H)
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        d2 = (xs[None, :] - cu) ** 2 + (ys[:, None] - cv) ** 2
        g = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
        heat[cy, cx] = 1.0  # 峰格置 1（focal 正样本掩码 = heat==1）
    return heat, off, cls


# ── 保尺度增广（pillarize 之前） ────────────────────────────────────────
def rot_flip_stirrup(p, gt, rng, rot=True, flip=True):
    """箍筋顶面增广：绕 z 旋转/沿 y 翻转，同步旋转两腿 (x,y)。"""
    if rot:
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = x * c - y * s
        p[:, 1] = x * s + y * c
        legs = gt[:, :4].reshape(-1, 2)
        lx, ly = legs[:, 0].copy(), legs[:, 1].copy()
        legs[:, 0] = lx * c - ly * s
        legs[:, 1] = lx * s + ly * c
        gt[:, :4] = legs.reshape(-1, 4)
    if flip and rng.random() < 0.5:
        p[:, 1] = -p[:, 1]
        gt[:, 1] = -gt[:, 1]
        gt[:, 3] = -gt[:, 3]
    return p, gt


def _rot_flip_points(p, iu, iv, rng, rot=True, flip=True, gt=None, gt_uv=None):
    """截面内旋转/翻转，同步旋转 GT 点。gt_uv: (K,2) 或含多列的 (K,≥4)。"""
    if rot:
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        u, v = p[:, iu].copy(), p[:, iv].copy()
        p[:, iu] = u * c - v * s
        p[:, iv] = u * s + v * c
        if gt is not None:
            gu, gv = gt[:, gt_uv].copy(), gt[:, gt_uv + 1].copy()
            gt[:, gt_uv] = gu * c - gv * s
            gt[:, gt_uv + 1] = gu * s + gv * c
    if flip and rng.random() < 0.5:
        p[:, iv] = -p[:, iv]
        if gt is not None:
            gt[:, gt_uv + 1] = -gt[:, gt_uv + 1]
    return p, gt


def _drop_jitter(p, rng, drop=(0.1, 0.5), jitter=1.5):
    if drop is not None:
        p = p[rng.random(len(p)) > rng.uniform(*drop)]
    if jitter > 0:
        p = p + rng.normal(0, rng.uniform(0, jitter), p.shape).astype(np.float32)
    return p


# ── 几何量测（连续量的唯一来源，2026-10-23 定案） ────────────────────────
def prefilter_far(p, iu, iv, idd, far_thr=None, cell=CELL):
    """删除触及窗口远端的柱的点——地面/背景长条结构一刀切。

    依据：钢筋端窗 (−300,1750) 的远端只可能被贯穿全窗的地面/背景触到
    （构件最长外露 1500 + 翘曲 ≤1550，够不到 1650）；若构件柱真能伸到
    远端，裁剪窗本身早已截断——本规则不比窗口更强的尺寸限制。
    阈值随窗缩放（REBAR_WIN[1]−100）。对套筒（远端 +80）/箍筋（z 远端
    ~+200）裁剪，对任意构件尺寸均为无操作。
    （2026-10-24 修订：初版"跨度>1600 删"会误删长外露筋——其格子含端面
    点(t≈0)+筋体点(t≤外露)，跨度可达 ~1800。）
    """
    if far_thr is None:
        far_thr = REBAR_WIN[1] - 100.0
    if len(p) == 0:
        return p
    gx = np.floor(p[:, iu] / cell).astype(np.int64)
    gy = np.floor(p[:, iv] / cell).astype(np.int64)
    gx -= gx.min()
    gy -= gy.min()
    flat = gy * (int(gx.max()) + 1) + gx
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    uniq, start = np.unique(fs, return_index=True)
    dmax = np.maximum.reduceat(p[order, idd], start)
    bad = uniq[dmax > far_thr]
    return p[~np.isin(flat, bad)]


def measure_exposed(p, uv, radius=22.0, topk=15):
    """外露长度：检出 (u,v) 柱查询的 t 顶端（top-k 均值，抗单点噪声）。

    基准=裁剪原点（端面 t=0）；柱半径 22mm < 最小筋距 120mm/2，够不到
    地面（保护层 ≥30mm）；垫片终止于筋尖不影响 t_max。
    """
    if len(p) == 0:
        return 0.0
    d2 = (p[:, 1] - uv[0]) ** 2 + (p[:, 2] - uv[1]) ** 2
    sel = np.sort(p[d2 < radius * radius, 0])[::-1]
    if len(sel) == 0:
        return 0.0
    return float(sel[:min(topk, len(sel))].mean())


def measure_exposed_local(p, uv, radius=22.0, topk=15,
                          r_in=26.0, r_out=56.0):
    """外露长度，局部混凝土平面基准（P-030 原则：面的基准来自混凝土点
    自身，不用面参数 d）。

    筋顶端同 measure_exposed；基准面=筋外环带 (r_in,r_out) 内点的
    t 中位数——环带落在端面混凝土上（最近邻筋距 120mm，56mm 环带
    够不到邻筋），局部中位数把面参数 d 的系统偏移（点级生长面 d 偏到
    构件外 p50 +6~9mm，外露量测继承之）与局部翘曲一并吸收。
    环带点 <20 退化回全局 t=0 基准（measure_height 同款兜底模式）。"""
    tip = measure_exposed(p, uv, radius, topk)
    if len(p) == 0:
        return tip
    d2 = (p[:, 1] - uv[0]) ** 2 + (p[:, 2] - uv[1]) ** 2
    ring = (d2 > r_in * r_in) & (d2 < r_out * r_out)
    if int(ring.sum()) >= 20:
        return tip - float(np.median(p[ring, 0]))
    return tip


def measure_height(p, leg_xy, top_z, bar_r, topk=10, r_in=30.0, r_out=90.0):
    """箍筋高度：腿柱 z 顶端（top-k 均值）− 邻域混凝土表面基准 − 杆半径。

    表面基准取腿周围环带（30-90mm）内 z < z_top−25 的点中位数（label-free
    排除 U 形自身）；低频翘曲在环带尺度内被中位数吸收。
    """
    if len(p) == 0:
        return 0.0
    d2 = (p[:, 0] - leg_xy[0]) ** 2 + (p[:, 1] - leg_xy[1]) ** 2
    col = np.sort(p[d2 < 12.0 * 12.0, 2])[::-1]
    if len(col) == 0:
        return 0.0
    z_top = float(col[:min(topk, len(col))].mean())
    ring = (d2 > r_in * r_in) & (d2 < r_out * r_out) & (p[:, 2] < z_top - 25.0)
    if ring.sum() < 10:  # 兜底放宽半径，但仍排除腿芯柱（自身点会拉高基准）
        ring = (d2 > 15.0 * 15.0) & (d2 < r_out * r_out * 4) \
            & (p[:, 2] < z_top - 25.0)
    surf = float(np.median(p[ring, 2])) if ring.sum() else top_z
    return max(z_top - surf - bar_r, 0.0)


# ── 缓存构建与读取 ──────────────────────────────────────────────────────
def _cap(p, cap, rng):
    if len(p) > cap:
        p = p[rng.choice(len(p), cap, replace=False)]
    return p


def build_split_cache(split_paths, out_path):
    """裁剪全部构件的三端样本并缓存（一次）。

    v6：额外缓存双端交叉负样本视图（rbn=套筒端按钢筋窗、sln=出筋端按
    套筒窗）——端别由检测头自决的前提（两头都看两端，正确端才点火）。
    """
    rb = {"pts": [], "uv": [], "n": [], "expo": [], "diam": []}
    sl = {"pts": [], "gt": [], "n": []}
    st = {"pts": [], "gt": [], "topz": []}
    rbn, sln = [], []
    for f in split_paths:
        d = np.load(f, allow_pickle=False)
        pts = d["points"].astype(np.float32)
        lbls = d["labels"].astype(np.int64)
        meta = json.loads(str(d["meta_json"]))
        rng = np.random.default_rng(zlib.crc32(f.encode("utf-8")))
        p, uv, n, expo, diam = make_rebar_crop(pts, lbls, meta)
        rb["pts"].append(_cap(p, REBAR_CAP, rng).astype(np.float16))
        rb["uv"].append(uv.astype(np.float32))
        rb["n"].append(n)
        rb["expo"].append(np.float32(expo))
        rb["diam"].append(np.float32(diam))
        sp, sg, sn = make_sleeve_crop(pts, lbls, meta)
        sl["pts"].append(_cap(sp, SLEEVE_CAP, rng).astype(np.float16))
        sl["gt"].append(sg.astype(np.float32))
        sl["n"].append(sn)
        pn, _, _, _, _ = make_rebar_crop(pts, lbls, meta, end="sleeve")
        rbn.append(_cap(pn, REBAR_CAP, rng).astype(np.float16))
        pn2, _, _ = make_sleeve_crop(pts, lbls, meta, end="rebar")
        sln.append(_cap(pn2, SLEEVE_CAP, rng).astype(np.float16))
        s = make_stirrup_crop(pts, meta, lbls=lbls)
        if s is None:
            st["pts"].append(np.zeros((0, 3), np.float16))
            st["gt"].append(np.zeros((0, 6), np.float32))
            st["topz"].append(np.float32(0.0))
        else:
            sp2, sg2, top_z = s
            st["pts"].append(_cap(sp2, STIRRUP_CAP, rng).astype(np.float16))
            st["gt"].append(sg2.astype(np.float32))
            st["topz"].append(np.float32(top_z))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        rb_pts=np.array(rb["pts"], dtype=object),
                        rb_uv=np.array(rb["uv"], dtype=object),
                        rb_n=np.array(rb["n"]), rb_expo=np.array(rb["expo"]),
                        rb_diam=np.array(rb["diam"]),
                        sl_pts=np.array(sl["pts"], dtype=object),
                        sl_gt=np.array(sl["gt"], dtype=object),
                        sl_n=np.array(sl["n"]),
                        st_pts=np.array(st["pts"], dtype=object),
                        st_gt=np.array(st["gt"], dtype=object),
                        st_topz=np.array(st["topz"]),
                        rbn_pts=np.array(rbn, dtype=object),
                        sln_pts=np.array(sln, dtype=object))
    return out_path


def load_bev_cache(root, split):
    return np.load(os.path.join(root, f"{split}.npz"), allow_pickle=True)
