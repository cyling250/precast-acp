# -*- coding: utf-8 -*-
"""
v6 端到端全链入口：整构件点云 → 连接面 → 面视图检测 → 五组装配参数。

验证顺序②（docs/handover.md T-033）：阶段 B 用经典面先串起来出全链
数字——面来源可切换：
  --face oracle   GT 面（机制验证+性能上限，对照 bev_v1/v2 oracle 口径）
  --face classic  经典面（teeth 出筋面 + 轴外推套筒面 + 法向聚类顶面；
                  本机复核：出筋 75%、套筒经典法缺口由轴外推补、顶 100%）
  --face sem      阶段 A 语义面（步骤③，需先训练 train_e2e_face）

端别由检测头自决：两端面视图都跑钢筋头+套筒头，钢筋头点火=出筋面、
套筒头点火=套筒面（训练已含交叉负样本）。箍筋头跑顶面视图。

套筒段（--sleeve，T-038 接入 T-035 定稿级联融合）：
  raw     裸 bev_v7 头（基线口径，套筒召回天花板 ~0.58）
  fusion  密度孔提议 ∪ 裸头峰 → 统一物理验证器 → Kåsa+扇区内缘精测
          （face_x/sign 从面参数来——面失败时回退裸头输出；
          融合输入=混凝土类点，材质来源与各口径的分割一致）

评估：合成 test 56 件，五组参数（根数/位置/规格/外露/箍高）对照 GT，
与 bev_v1/v2 的 oracle 投影口径可比 → "面投影代价"= 两口径之差。

用法（cwd=src）：论文入口=experiments.exp2_validate single / exp3_ablation；
本模块保留同参 CLI 作调试直入口：
  python -m lib.pipeline.single_chain --face classic [--limit 20]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from lib.utils import anchor_project_path
from lib.e2e.plane_classic import find_teeth_and_faces, extract_planes
from lib.e2e.bev_dataset import (CELL, END_ORIGIN, END_WH, ST_ORIGIN, ST_WH,
                                 REBAR_WIN, SLEEVE_WIN, pillarize,
                                 measure_exposed, measure_height,
                                 prefilter_far, REBAR_CAP, SLEEVE_CAP,
                                 STIRRUP_CAP)
from lib.e2e.model_pillar import (PillarBEVNet, decode, match_dets,
                                  STIRRUP_DIAM_TABLE, REBAR_DIAM_TABLE,
                                  SLEEVE_IDIAM_TABLE)
from lib.e2e.point_head import PointCircleNet, detect, make_seeds

POS_GATE = 120.0
SLEEVE_PEAK_THR = 0.15   # 融合峰工作点=T-035 定稿 FUSED@0.15（阈值只管
SLEEVE_PEAK_TOPK = 64    # 召回、精度由物理验证器把守；0.3 掉 3-4pp 召回）
SEED_VOX_END = 10.0      # 点级头种子体素（与 train_e2e_point 同源约定）
SEED_VOX_TOP = 20.0


# ── 面提取（三种来源，统一返回质心系） ──────────────────────────────────
def oracle_faces(pts, meta, c0):
    """GT 面（质心系 pts；end_x 世界系须减 c0——曾在质心系上直接用
    世界 end_x 致带全空/面心 NaN，全链 oracle 全面 0 检出的根因）。"""
    a = np.array([1.0, 0.0, 0.0])
    out = {}
    for key, x_key, sign in (("rebar", "rebar_end_x", 1.0),
                             ("sleeve", "sleeve_end_x", -1.0)):
        end_x = float(meta[x_key]) - c0[0]
        n3 = np.array([sign, 0.0, 0.0])
        band = np.abs(pts[:, 0] - end_x) < 60.0
        if band.sum() < 10:
            band = np.abs(pts[:, 0] - end_x) < 150.0
        c3 = pts[band].mean(0)
        out[key] = {"n3": n3, "d": float(c3 @ n3), "c3": c3}
    hps = meta.get("hairpin_gt") or []
    out["top"] = ({"d": float(hps[0]["top_z"]) - c0[2]} if hps else None)
    out["axis"] = a
    return out


def classic_faces(pts):
    """teeth 出筋面 + 轴外推套筒端 + 法向聚类顶面（返回 None=提取失败）。"""
    teeth, faces = find_teeth_and_faces(pts)
    if not faces:
        return None
    f = faces[0]
    n3, d_rb, c3 = f["n3"], f["d"], f["c3"]
    # 轴外推套筒端面：出筋面法向即构件轴；套筒端=沿 −n3 的点云极值带
    s = pts @ n3  # 出筋方向为正
    smin = float(np.percentile(s, 0.2))
    band = s < smin + 200.0
    n3s = -n3
    c3s = pts[band].mean(0)
    # 顶面：法向聚类中 n_z>0 的最高水平面
    planes = extract_planes(pts)
    top = None
    for p in planes:
        if p["n"][2] > 0.9 and (top is None or p["d"] > top["d"]):
            top = p
    return {"rebar": {"n3": n3, "d": d_rb, "c3": c3},
            "sleeve": {"n3": n3s, "d": float(c3s @ n3s), "c3": c3s},
            "top": ({"d": float(top["d"])} if top is not None else None),
            "axis": n3}


def sem_faces(pts, model, device):
    """阶段 A 语义面：分类（一次前向）→ 逐类掩码（kNN 传播）→
    原始点 refine RANSAC（法向约束+概率引导，v6d 部署几何链）。

    返回 (faces, (m1,m2,m3))；faces 各键为平面列表（多面实例）。
    """
    from scipy.spatial import cKDTree
    from lib.e2e.face_sem import predict_bands, refine_face_planes
    cents, nrm, prob = predict_bands(model, pts, device)
    tree = cKDTree(cents)
    _, ix = tree.query(pts, k=1, workers=-1)
    prob_pt = prob[ix]
    masks = tuple(prob_pt[:, cls] > 0.5 for cls in (1, 2, 3))
    faces = {}
    for key, cls, mk in zip(("rebar", "sleeve", "top"), (1, 2, 3), masks):
        faces[key] = refine_face_planes(pts, mk, cents, nrm, prob, cls)
    out = {}
    for key in ("rebar", "sleeve"):
        if not faces[key]:
            return None, masks
        pf = faces[key][0]
        # 视图原点 c3 只用面片带内点（|t|<60）：掩码含外露筋全段（面片+
        # 梳子一体），均值质心被梳齿拉偏 → (u,v) 原点系统偏移、检测位置
        # 整体平移（2026-10-26 诊断：sem 34mm vs oracle 5.9mm 的主项）。
        n3 = pf["n"] / np.linalg.norm(pf["n"])
        t_in = pf["inliers"] @ n3 - pf["d"]
        band = np.abs(t_in) < 60.0
        c3 = (pf["inliers"][band].mean(0) if band.sum() >= 50
              else pf["inliers"].mean(0))
        out[key] = {"n3": pf["n"], "d": pf["d"], "c3": c3}
    out["top"] = ({"d": faces["top"][0]["d"]} if faces["top"] else None)
    out["axis"] = out["rebar"]["n3"]
    return out, masks


def sem_class_masks(pts, model, device):
    """部署侧逐类点掩码（面片+附属结构一体；视图点集=对应类）。"""
    from lib.e2e.face_sem import predict_class_masks
    masks = predict_class_masks(model, pts, device)
    return masks[:, 1], masks[:, 2], masks[:, 3]


# ── 面视图投影与检测 ────────────────────────────────────────────────────
def face_view(pts, face, win, cap, rng, rebar_far=True):
    """端面 → (t,u,v) 视图（面参数系）。返回 (P,3), 基 (e1,e2)。

    u 轴手性必须与训练视图构造同向（make_rebar_crop/make_sleeve_crop：
    u=y−cy、v=z−cz、t 沿外法向）——叉积 e1=cross(ẑ,n3) 在朝 −x 的
    端面上给出 (0,−1,0)，须翻转到 +y，否则视图左右镜像：P-057 实锤
    同权重裸头套筒召回 0.926（缓存视图）→0.575（镜像视图）。v 轴在
    右手系下恒为 +z 不动，翻转只作用于 e1。"""
    n3 = face["n3"] / np.linalg.norm(face["n3"])
    e1 = np.cross([0.0, 0.0, 1.0], n3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n3, e1)
    if n3[0] < 0:
        e1 = -e1
    c3 = face["c3"]
    t = pts @ n3 - face["d"]
    u = pts @ e1 - c3 @ e1
    v = pts @ e2 - c3 @ e2
    p = np.column_stack([t, u, v]).astype(np.float32)
    m = (p[:, 0] > win[0]) & (p[:, 0] < win[1])
    p = p[m]
    if len(p) > cap:
        p = p[rng.choice(len(p), cap, replace=False)]
    if rebar_far:
        p = prefilter_far(p, 1, 2, 0)
    return p, (e1, e2)


def top_view(pts, top_d, rng, band=True, keep=None, n3=(0.0, 0.0, 1.0)):
    """箍筋面视图：(x−x0, v−yc, s)，s=面法向深度、v=面内横向。

    n3=箍筋面法向（默认 ẑ=重力顶面，与训练缓存同构逐位一致）。滚转
    构件的箍筋面=竖直侧面（s4/s6 实锤），传入该面法向即得侧视图——
    检测头对视图平面内的旋转不变（2026-10-29 认证帧实验 st 0→14-20
    实证），构件系全程保持重力系，rb/sl 链零改动（BEV 峰视图对旋转
    敏感，动系会掉环——m1-s4 实锤 8/8→3/8）。

    窗推导（T-040 箍筋修复定稿，两处缺陷均由复合梁件解剖实锤）：
    - x 窗用带点全范围分位（T-040 修复清单#1：above 子集可比带点短
      ~900mm，窗末端少 2-3 榀）。
    - s 门槛：带点输入（band=True，oracle/pointgrow/pred）取
      min(top_d−50, 带点 s 2% 分位)——grown 顶面被翘曲高区抬高时
      （004 实锤 top d 比 GT 顶面高 +80mm，固定 top_d−50 把正常顶面/
      低翘曲端腿柱整段切出视图），带点本身已按材质+GT-锚定 s 界定，
      取较松者只放不收；全点输入（band=False，classic）保持 top_d−50
      （全点 2% 分位是深身体点，min 会把窗放开到全身——classic 臂
      2026-10-29 实锤腿位 5.9→54.5mm，即此回退的原因）。
    - yc = 带点 v 中位：旧 above（s>top_d+10）子集在 pred 带被翘曲侧边
      混凝土污染（004/012 实锤 yc 偏 −508/+428mm，窗整体错位一半腿出
      窗）；带点 v 跨构件全宽，中位=截面中心，对污染稳健。
    - keep：箍筋钢点 bool 掩码（与 pts 同长）。窗变大后 150k 均匀下采
      样把箍筋钢点稀释（腿位 4.33→5.28 的主项），保留采样=钢点全保、
      余量给混凝土。
    """
    if len(pts) < 100:
        return None
    n3 = np.asarray(n3, np.float64)
    nn = (n3[1:] / max(float(np.linalg.norm(n3[1:])), 1e-9)).astype(
        np.float32)
    ev = np.array([nn[1], -nn[0]], np.float32)   # 面内横向（n3=ẑ 时=+y）
    s = pts[:, 1] * nn[0] + pts[:, 2] * nn[1] - top_d   # float32 保持
    if band:
        s_lo = min(top_d - 50.0, float(np.percentile(s, 2.0)))
    else:
        s_lo = top_d - 50.0
    m = s > s_lo
    if m.sum() < 100:
        return None
    q = pts[m]
    x0 = float(np.percentile(q[:, 0], 0.5)) - 200.0
    x1 = float(np.percentile(q[:, 0], 99.5)) + 200.0
    span = max(ST_WH[0] * CELL, x1 - x0)
    v = q[:, 1] * ev[0] + q[:, 2] * ev[1]
    yc = float(np.median(v))
    win = ((pts[:, 0] > x0) & (pts[:, 0] < x0 + span)
           & (np.abs(pts[:, 1] * ev[0] + pts[:, 2] * ev[1] - yc)
              < ST_WH[1] * CELL / 2)
           & (s > s_lo))
    p = np.column_stack([pts[win, 0] - x0,
                         pts[win, 1] * ev[0] + pts[win, 2] * ev[1] - yc,
                         s[win]]).astype(np.float32)
    if len(p) > STIRRUP_CAP:
        if keep is not None:
            kp = np.asarray(keep, bool)[win]
            idx_steel = np.flatnonzero(kp)
            n_steel = len(idx_steel)
            if n_steel >= STIRRUP_CAP:
                sel = rng.choice(idx_steel, STIRRUP_CAP, replace=False)
            else:
                n_rest = STIRRUP_CAP - n_steel
                rest_pool = np.flatnonzero(~kp)
                rest = rng.choice(rest_pool,
                                  min(n_rest, len(rest_pool)),
                                  replace=False)
                sel = np.sort(np.concatenate([idx_steel, rest]))
            p = p[sel]
        else:
            p = p[rng.choice(len(p), STIRRUP_CAP, replace=False)]
    return p, (float(x0), float(yc))


def uv_to_world(det_uv, face, basis):
    """面视图 (u,v) → 质心系 (y,z)（GT positions_yz 同系）。"""
    e1, e2 = basis
    c3 = face["c3"]
    u = det_uv[:, 0] + c3 @ e1
    v = det_uv[:, 1] + c3 @ e2
    return np.column_stack([u, v])


# ── 单件全链推理 ────────────────────────────────────────────────────────
@torch.no_grad()
def run_one(pts, meta, faces, models, device, face_pts=None, thresh=0.3,
            conc_mask=None, sl_fuse=False, rbst="point", fuse_kwargs=None,
            rb_thr=0.5, st_steel=None):
    """三面视图 → 三检测头（端别自决）→ 五组参数。

    face_pts=(m1,m2,m3) 给定时每面视图点集=对应面类点（面片+附属结构
    一体，v6d 口径；oracle=GT 标签、sem=分类预测掩码；classic=None
    保持全点+prefilter_far 启发式基线本貌）。
    rbst="point"：钢筋/箍筋段用 point_v2b 点级圆检测头（T-035 per-kind
    定稿接进全链——此前链条里一直是 bev_v7 的 rb/st，定稿从未接入）；
    "bev" 保留 BEV 头路径作消融。套筒段无论何种模式都走 bev_v7+融合。
    sl_fuse=True 时套筒段走级联融合（T-038）：conc_mask=混凝土类点
    （与各口径材质来源一致），face_x/sign 从 faces["sleeve"] 面参数
    导出（质心系，n3 已 snap 到构件轴）；融合无环时回退裸头输出。
    """
    if faces is None:
        return None
    rng = np.random.default_rng(0)
    model_rb, model_sl, model_st = models
    m1, m2, m3 = face_pts if face_pts is not None else (None, None, None)
    pts_by_face = {"rebar": pts if m1 is None else pts[m1],
                   "sleeve": pts if m2 is None else pts[m2]}
    res = {"face_fail": True}
    # ── 两端面 × 两头部（端别自决）──
    views = {}
    for key in ("rebar", "sleeve"):
        face = faces[key]
        pq = pts_by_face[key]
        p, basis = face_view(pq, face, REBAR_WIN, REBAR_CAP, rng)
        if rbst == "point":
            seeds, nb, extra, nb2 = make_seeds(p, seed_vox=SEED_VOX_END,
                                               rng=rng)
            det_rb = detect(model_rb, seeds, nb, extra, nb2,
                            REBAR_DIAM_TABLE, thr=rb_thr, device=device,
                            pts_uv=p[:, 1:3])
        else:
            feat = pillarize(p, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
            det_rb = decode(model_rb(torch.from_numpy(
                feat[None]).to(device)), CELL, END_ORIGIN,
                thresh=thresh)[0]
        p2, _ = face_view(pq, face, SLEEVE_WIN, SLEEVE_CAP, rng,
                          rebar_far=False)
        feat2 = pillarize(p2, 1, 2, 0, END_ORIGIN, END_WH, ext_from=250.0)
        det_sl = decode(model_sl(torch.from_numpy(feat2[None]).to(device)),
                        CELL, END_ORIGIN, thresh=thresh)[0]
        views[key] = {"p_rb": p, "basis": basis,
                      "rb": det_rb, "sl": det_sl, "p_sl": p2}
    # 端别自决：钢筋头点火面=出筋面；套筒头点火面=套筒面
    # （都点火/都不点火时取热图峰值高者）
    def score(d):
        return float(d[:, 2].max()) if len(d) else 0.0

    key_rb = max(views, key=lambda k: (len(views[k]["rb"]) > 0,
                                       score(views[k]["rb"])))
    key_sl = max(views, key=lambda k: (len(views[k]["sl"]) > 0,
                                       score(views[k]["sl"])))
    vw, face = views[key_rb], faces[key_rb]
    det = vw["rb"]
    res["rebar"] = {
        "n": len(det),
        "uv_world": uv_to_world(det[:, :2], face, vw["basis"]),
        "spec": [REBAR_DIAM_TABLE[int(i)] for i in det[:, 3:].argmax(1)],
        "expo": ([measure_exposed(vw["p_rb"], d) for d in det[:, :2]]
                 if len(det) else []),
        "end_pick": key_rb}
    vw, face = views[key_sl], faces[key_sl]
    det = vw["sl"]
    res["sleeve"] = {
        "n": len(det),
        "uv_world": uv_to_world(det[:, :2], face, vw["basis"]),
        "spec": [SLEEVE_IDIAM_TABLE[int(i)] for i in det[:, 3:].argmax(1)],
        "end_pick": key_sl}
    # ── 套筒级联融合（T-035 定稿，T-038 接入）：低阈值峰 ∪ 密度孔提议 →
    # 统一物理验证器 → 精测。face_x=套筒面沿轴位置、sign=外向（n3 已
    # snap 轴）；环候选与输出同在质心系真 (y,z)，主循环统一加回世界系。
    # 两个接入实锤（2026-10-28 诊断）：
    # ① 融合输入必须全量混凝土点——翘曲可把真实端面推出名义位置
    #   +45~500mm，端区窗口裁剪会把端面切掉（几何提议 0.796→0.484）；
    # ② 峰必须换算回真 (y,z)——面基 e1 在 −x 端=(0,−1,0)，uv_to_world
    #   返回的是镜像坐标，直接喂会落不到孔上被验证器全杀。
    # 融合环只在套筒端面物理成立（出筋端无环，验证器天然全拒）；无环=
    # 端面不可融合，回退裸头（含原端别）。
    if sl_fuse and faces.get("sleeve") is not None \
            and conc_mask is not None:
        from lib.measure.sleeve_fusion import fuse_rings
        fsl = faces["sleeve"]
        e1, e2 = views["sleeve"]["basis"]
        sign = 1.0 if float(fsl["n3"][0]) > 0 else -1.0
        face_x = float(fsl["c3"][0])
        conc = pts[conc_mask]
        if len(conc) >= 200:
            feat_pk = pillarize(views["sleeve"]["p_sl"], 1, 2, 0,
                                END_ORIGIN, END_WH, ext_from=250.0)
            det_pk = decode(model_sl(torch.from_numpy(
                feat_pk[None]).to(device)), CELL, END_ORIGIN,
                thresh=SLEEVE_PEAK_THR, topk=SLEEVE_PEAK_TOPK)[0]
            if len(det_pk):
                uw = det_pk[:, 0] + float(fsl["c3"] @ e1)
                vw = det_pk[:, 1] + float(fsl["c3"] @ e2)
                peaks = np.column_stack([uw * float(e1[1]), vw])
            else:
                peaks = ()
            rings = fuse_rings(conc, face_x, sign, extra_cands=peaks,
                               **(fuse_kwargs or {}))
            if rings:
                res["sleeve"] = {
                    "n": len(rings),
                    "uv_world": np.asarray([r["c"] for r in rings],
                                           np.float32),
                    "spec": [SLEEVE_IDIAM_TABLE[int(np.argmin(
                        np.abs(np.asarray(SLEEVE_IDIAM_TABLE)
                               - 2.0 * float(r["r_in"]))))]
                        for r in rings],
                    "r_in": [round(2.0 * float(r["r_in"]), 2)
                             for r in rings],
                    "end_pick": "sleeve",
                    "src": "fusion"}
    # ── 顶面 → 箍筋头（面法向参数化：正立=重力顶面逐位不变，滚转构件
    # =认证侧面视图，构件系保持重力系）──
    res["stirrup"] = None
    if faces.get("top") is not None:
        topf = faces["top"]
        n3t = np.asarray(topf.get("n3", (0.0, 0.0, 1.0)), np.float64)
        nn = n3t[1:] / max(float(np.linalg.norm(n3t[1:])), 1e-9)
        ev2 = (float(nn[1]), -float(nn[0]))
        pq = pts if m3 is None else pts[m3]
        tvw = top_view(pq, float(topf["d"]), rng, band=m3 is not None,
                       keep=(st_steel[m3] if st_steel is not None
                             and m3 is not None else None),
                       n3=tuple(n3t))
        if tvw is not None:            # 带点过少时 top_view 返回 None
            tv, (x0, yc) = tvw
            if rbst == "point":
                # 深度回 0 轴（P-042 轴序约定）：视图 (x−x0,v−yc,s)
                # 重排 (s,x,v)，检测平面=轴1,2=(x,v)
                p_det = np.column_stack([tv[:, 2], tv[:, 0], tv[:, 1]])
                seeds, nb, extra, nb2 = make_seeds(p_det,
                                                   seed_vox=SEED_VOX_TOP,
                                                   rng=rng)
                det_st = detect(model_st, seeds, nb, extra, nb2,
                                STIRRUP_DIAM_TABLE, thr=0.5, device=device,
                                pts_uv=p_det[:, 1:3])
            else:
                feat = pillarize(tv, 0, 1, 2, ST_ORIGIN, ST_WH,
                                 d_shift=0.0)
                det_st = decode(model_st(torch.from_numpy(
                    feat[None]).to(device)), CELL, ST_ORIGIN,
                    thresh=thresh, topk=96)[0]
            res["stirrup"] = {"n_det": len(det_st),
                              "dets": det_st[:, :3].tolist(),
                              "spec": [STIRRUP_DIAM_TABLE[int(i)]
                                       for i in det_st[:, 3:].argmax(1)],
                              "tv": tv, "x0": x0, "yc": yc,
                              "ev2": ev2,
                              "top_d": float(topf["d"]),
                              "tv_pts": len(tv)}
    res["face_fail"] = False
    return res


# ── 评估（对照 meta GT） ────────────────────────────────────────────────
def sleeve_visible(pts, lbls, meta):
    """逐 GT 套筒可见性：套筒柱域（|x−cx|≤80、平面距≤外径/2+30mm）内
    混凝土点 ≥30 视为可见（端面被遮挡/完全缺点的套筒不计入召回分母）。"""
    conc = lbls == 1
    out = []
    for g in meta.get("sleeve_gt") or []:
        cx = float(g["center_x"])
        cy, cz = float(g["center_yz"][0]), float(g["center_yz"][1])
        r = float(g.get("outer_diam", 60.0)) / 2 + 30.0
        m = (conc & (np.abs(pts[:, 0] - cx) <= 80.0)
             & ((pts[:, 1] - cy) ** 2 + (pts[:, 2] - cz) ** 2 <= r * r))
        out.append(bool(int(m.sum()) >= 30))
    return out


def eval_one(res, meta, sl_vis=None, st_off=None):
    """全链结果 vs GT：五组参数逐面统计。

    sl_vis 给定时（逐 GT 套筒可见性），套筒召回/匹配只对可见套筒——
    端面完全看不见的套筒不拉低指标（用户 2026-10-27 口径）。
    st_off=(c0_x, c0_y)：箍筋 GT 腿（世界系 p_foot）→ 质心系视图坐标
    的云原点修正——视图窗 x0/yc 是质心系，不修正则 GT 全被推出窗外
    （~c0 量级偏移，匹配全废）。"""
    out = {}
    # 钢筋：GT positions_yz（世界 y,z；全链在质心系——评估前调用方先减 c0）
    uv_gt = np.asarray(meta["rebar_positions_yz"], np.float32)
    det = np.asarray(res["rebar"]["uv_world"], np.float32).reshape(-1, 2)
    si, gi = match_dets(det, uv_gt, gate=POS_GATE)
    out["rb_n_ok"] = int(len(det) == len(uv_gt))
    out["rb_end_ok"] = int(res["rebar"]["end_pick"] == "rebar")
    out["rb_match"] = len(si)
    out["rb_pos"] = [float(np.linalg.norm(det[a] - uv_gt[b]))
                     for a, b in zip(si, gi)]
    diam = float(meta["rebar_diam"])
    out["rb_spec"] = [int(abs(res["rebar"]["spec"][a] - diam) < 0.1)
                      for a, _ in zip(si, gi)]
    out["rb_diam_err"] = [float(res["rebar"]["spec"][a] - diam)
                          for a, _ in zip(si, gi)]
    out["rb_expo_err"] = []
    if si:
        expo_gt = None
        # GT 外露=逐根 p1.x−end_x（取筋长一致的构件级口径：meta protrusion
        # 与逐根 p1 接近；沿用 bev 缓存口径取第 0 根）
        g0 = meta["rebar_gt"][0]
        end_x = float(meta["rebar_end_x"])
        flip = -1.0 if float(meta["sleeve_end_x"]) > end_x else 1.0
        expo_gt = flip * (float(g0["p1"][0]) - end_x)
        out["rb_expo_err"] = [abs(res["rebar"]["expo"][a] - expo_gt)
                              for a, _ in zip(si, gi)]
    # 套筒（可见性分母）
    uv_gt_s = np.asarray([[g["center_yz"][0], g["center_yz"][1]]
                          for g in meta["sleeve_gt"]], np.float32)
    if sl_vis is not None and len(sl_vis):
        vis_m = np.asarray(sl_vis, bool)
    else:
        vis_m = np.ones(len(uv_gt_s), bool)
    out["sl_all_n"] = int(len(uv_gt_s))
    out["sl_total_gt"] = int(vis_m.sum())
    uv_gt_s = uv_gt_s[vis_m]
    det_s = np.asarray(res["sleeve"]["uv_world"], np.float32).reshape(-1, 2)
    si_s, gi_s = match_dets(det_s, uv_gt_s, gate=POS_GATE)
    out["sl_n_ok"] = int(len(det_s) == len(uv_gt_s))
    out["sl_end_ok"] = int(res["sleeve"]["end_pick"] == "sleeve")
    out["sl_match"] = len(si_s)
    out["sl_pos"] = [float(np.linalg.norm(det_s[a] - uv_gt_s[b]))
                     for a, b in zip(si_s, gi_s)]
    out["sl_spec"] = []
    out["sl_diam_err"] = []
    vis_gt = [g for g, v in zip(meta["sleeve_gt"], vis_m) if v]
    for a, b in zip(si_s, gi_s):
        gt_d = float(vis_gt[b]["inner_diam"])
        out["sl_spec"].append(
            int(abs(res["sleeve"]["spec"][a] - gt_d) < 0.1))
        out["sl_diam_err"].append(float(res["sleeve"]["spec"][a] - gt_d))
    # 箍筋（有发夹筋的构件）：榀级召回 + 腿位/高度量测（用户 2026-10-28
    # 口径：弃"榀数全对"，改 召回/腿位p50/高度p50 三指标）
    hps = meta.get("hairpin_gt") or []
    out["st_total_gt"] = len(hps)
    out["st_match"] = 0
    out["st_leg_err"] = []
    out["st_h_err"] = []
    if res["stirrup"] is not None and hps:
        st = res["stirrup"]
        dets = np.asarray(st["dets"], np.float32).reshape(-1, 3)
        x0, yc = float(st["x0"]), float(st["yc"])
        tv, top_d = st["tv"], float(st["top_d"])
        ev2 = st.get("ev2") or (1.0, 0.0)   # 面内横向（重力顶面=(1,0)）
        specs = st.get("spec") or []
        order = np.argsort(dets[:, 0]) if len(dets) else []
        pairs, used = [], np.zeros(len(dets), bool)
        for a in range(len(order)):
            if used[order[a]]:
                continue
            for b in range(a + 1, len(order)):
                k = order[b]
                if not used[k] and abs(dets[k, 0] - dets[order[a], 0]) < 30.0:
                    pairs.append((order[a], k))
                    used[order[a]] = used[k] = True
                    break
        # 榀级匹配：检出对中心 x ↔ GT 对中心 x，贪心最近 ≤60mm
        ox, oy = (float(st_off[0]), float(st_off[1])) \
            if st_off is not None else (0.0, 0.0)
        gt_cx = [((float(g["p_foot_l"][0]) + float(g["p_foot_r"][0])) / 2
                  - ox - x0) for g in hps]
        det_cx = [(dets[a, 0] + dets[k, 0]) / 2 for a, k in pairs]
        usedg = np.zeros(len(hps), bool)
        for pi in np.argsort(det_cx):
            c = det_cx[pi]
            if not len(gt_cx):
                break
            j = int(np.argmin(np.abs(np.asarray(gt_cx) - c)))
            if usedg[j] or abs(gt_cx[j] - c) > 60.0:
                continue
            usedg[j] = True
            out["st_match"] += 1
            g = hps[j]
            a, k = pairs[pi]

            def _leg_v(py, pz):
                # GT 腿横向坐标投影到视图 v 轴（重力顶面 ev2=(1,0) 即 y）
                return py * ev2[0] + pz * ev2[1]

            gl = sorted([(float(g["p_foot_l"][0]) - ox - x0,
                          _leg_v(float(g["p_foot_l"][1]) - oy,
                                 float(g["p_foot_l"][2])) - yc),
                         (float(g["p_foot_r"][0]) - ox - x0,
                          _leg_v(float(g["p_foot_r"][1]) - oy,
                                 float(g["p_foot_r"][2])) - yc)])
            dl = sorted([(float(dets[a, 0]), float(dets[a, 1])),
                         (float(dets[k, 0]), float(dets[k, 1]))])
            dia = (float(specs[a]) if a < len(specs) else 12.0)
            for (dx, dy), (gx, gy) in zip(dl, gl):
                dd = float(np.hypot(dx - gx, dy - gy))
                if dd < 100.0:
                    out["st_leg_err"].append(dd)
                    h_hat = measure_height(tv, (dx, dy), top_d, dia / 2)
                    out["st_h_err"].append(abs(h_hat - float(g["height"])))
    return out


def aggregate(per_file):
    med = lambda x: round(float(np.median(x)), 2) if x else None
    agg = {}
    tot = len(per_file)
    ok = [r for r in per_file if not r.get("skip")]
    agg["面提取成功"] = f"{len(ok)}/{tot}"
    if not ok:
        return agg
    for pre, name in (("rb", "钢筋"), ("sl", "套筒")):
        agg[name] = {
            "根数全对": round(np.mean([r["eval"][f"{pre}_n_ok"]
                                      for r in ok]), 3),
            "端别正确": round(np.mean([r["eval"][f"{pre}_end_ok"]
                                      for r in ok]), 3),
            "召回": round(sum(r["eval"][f"{pre}_match"] for r in ok)
                          / max(sum(r["eval"]["sl_total_gt"]
                                    if pre == "sl" else r["eval"]["rb_gt"]
                                    for r in ok), 1), 3),
            "位置p50": med([e for r in ok for e in r["eval"][f"{pre}_pos"]]),
            "直径误差p50": med([e for r in ok
                               for e in r["eval"][f"{pre}_diam_err"]]),
            "直径误差p90": (round(float(np.percentile(
                [abs(e) for r in ok
                 for e in r["eval"][f"{pre}_diam_err"]], 90)), 2)
                if any(r["eval"][f"{pre}_diam_err"] for r in ok) else None)}
    agg["套筒"]["套筒可见占比"] = round(
        sum(r["eval"]["sl_total_gt"] for r in ok)
        / max(sum(r["eval"]["sl_all_n"] for r in ok), 1), 3)
    agg["钢筋"]["外露量测误差p50"] = med(
        [e for r in ok for e in r["eval"]["rb_expo_err"]])
    st_ok = [r for r in ok if r["eval"]["st_total_gt"] > 0]
    if st_ok:
        agg["箍筋"] = {
            "召回": round(sum(r["eval"]["st_match"] for r in st_ok)
                          / max(sum(r["eval"]["st_total_gt"]
                                    for r in st_ok), 1), 3),
            "腿位p50": med([e for r in ok
                            for e in r["eval"].get("st_leg_err", [])]),
            "高度p50": med([e for r in ok
                            for e in r["eval"].get("st_h_err", [])])}
    return agg


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", choices=("oracle", "classic", "sem",
                                       "pointgrow", "pointgrow-pred"),
                    default="classic")
    ap.add_argument("--sleeve", choices=("raw", "fusion"), default="fusion",
                    help="套筒段：fusion=T-035 级联融合（定稿），raw=裸头")
    ap.add_argument("--rbst", choices=("point", "bev"), default="point",
                    help="钢筋/箍筋段：point=point_v2b 点级头（T-035 定稿），"
                         "bev=bev_v7 BEV 头（消融）")
    ap.add_argument("--pt-ckpt", type=str,
                    default="out/exp1_train/head_point/best.pth",
                    help="rbst=point 时的点级头权重（ref_radial 从 ckpt 读）")
    ap.add_argument("--ckpt", type=str, default="out/exp1_train/head_bev/best.pth")
    ap.add_argument("--face-ckpt", type=str,
                    default="out/exp1_train/face_sem/last.pth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seg-ckpt", type=str, default="",
                    help="pointgrow-pred 臂的分割权重（空=默认 "
                         "out/exp1_train/seg/best.pth；v2 消融对比用）")
    ap.add_argument("--out", type=str, default="")
    return ap


def run_single(args):
    """单件全链批跑（原 main 本体，参数化）。返回
    {"aggregate":…, "per_file":[…]}。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(anchor_project_path(args.ckpt), map_location=device)
    if args.rbst == "point":
        # 点级头权重（v2b/v3 通用）：ref_radial 从 ckpt 读取（向后兼容 0）
        pt = torch.load(anchor_project_path(args.pt_ckpt),
                        map_location=device)
        ref_rad = int(pt.get("ref_radial", 0))
        model_rb = PointCircleNet(len(REBAR_DIAM_TABLE), bn=True,
                                  coarse=True, refine=True, fg=True,
                                  in_extra=7, ref_radial=ref_rad).to(device)
        model_rb.load_state_dict(pt["rb"])
        model_st = PointCircleNet(len(STIRRUP_DIAM_TABLE), bn=True,
                                  coarse=True, refine=True, fg=True,
                                  in_extra=7, ref_radial=ref_rad).to(device)
        model_st.load_state_dict(pt["st"])
    else:
        model_rb = PillarBEVNet(len(REBAR_DIAM_TABLE), in_ch=8).to(device)
        model_st = PillarBEVNet(len(STIRRUP_DIAM_TABLE)).to(device)
        model_rb.load_state_dict(ck["rb"])
        model_st.load_state_dict(ck["st"])
    model_sl = PillarBEVNet(len(SLEEVE_IDIAM_TABLE), in_ch=8).to(device)
    model_sl.load_state_dict(ck["sl"])
    for m in (model_rb, model_sl, model_st):
        m.eval()
    face_model = None
    if args.face == "sem":
        from lib.e2e.face_sem import MiniPointNet
        face_model = MiniPointNet().to(device)
        face_model.load_state_dict(torch.load(
            anchor_project_path(args.face_ckpt), map_location=device))
        face_model.eval()
    seg_model = seg_dev = None
    if args.face == "pointgrow-pred":
        from lib.segmentation.inference import load_model, predict_labels
        # load_model 对显式路径不接 PROJ 前缀（相对 cwd 解析，cwd=src
        # 时 out/... 落空——2026-10-29 夜链实锤），统一过锚定解析
        seg_model, seg_dev = load_model(
            anchor_project_path(args.seg_ckpt) if args.seg_ckpt else None)

    files = sorted(glob.glob(anchor_project_path("mock/inuse/test/*.npz")))
    if args.limit:
        files = files[:args.limit]
    per_file, times = [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        pts = d["points"].astype(np.float32)
        lbls = d["labels"].astype(np.int64)
        meta = json.loads(str(d["meta_json"]))
        c0 = pts.mean(0)
        p = pts - c0
        uv_gt_offset = c0[1:]  # 评估把 uv_world(质心系) 加回世界系
        t0 = time.time()
        face_pts = None
        st_steel = None
        # 融合输入的端面候选混凝土点：1∪2 类（T-039 阶段 2 定稿）——
        # 套筒端物理上无钢筋（决策 A：点级无套筒类），分割误标为 2 类的
        # 环壁内缘点必须回收，否则 r_in 系统偏大（规格 0.79→0.91 的主项）
        conc_mask = (lbls == 1) | (lbls == 2)
        if args.face == "oracle":
            faces = oracle_faces(p, meta, c0)
            from lib.e2e.face_sem import face_band_labels
            lab = face_band_labels(pts, lbls, meta)
            face_pts = (lab == 1, lab == 2, lab == 3)  # 干净上限（GT 类）
            st_steel = lbls == 3
        elif args.face == "pointgrow":
            # 点级生长面（T-037 §7.5 新架构；oracle 材质分割）。
            # 面视图点集与 oracle 同用 GT 带（face_band_labels），唯一
            # 变量 = 面参数来源 → 与 --face oracle 直接可比。
            from lib.e2e.face_pointgrow import extract_faces
            faces = extract_faces(p, lbls)
            from lib.e2e.face_sem import face_band_labels
            lab = face_band_labels(pts, lbls, meta)
            face_pts = (lab == 1, lab == 2, lab == 3)
            st_steel = lbls == 3
        elif args.face == "pointgrow-pred":
            # 部署链复验：材质标签用**分割预测**（非 GT）→ 生长面 →
            # 检测头。视图点集同样来自预测标签（face_band_labels 只认
            # 材质 id，直接换 pred 标签输入）。
            from lib.e2e.face_pointgrow import extract_faces
            lbls_pred = predict_labels(pts, seg_model, seg_dev)
            faces = extract_faces(p, lbls_pred)
            from lib.e2e.face_sem import face_band_labels
            lab = face_band_labels(pts, lbls_pred, meta)
            face_pts = (lab == 1, lab == 2, lab == 3)
            st_steel = lbls_pred == 3
            conc_mask = (lbls_pred == 1) | (lbls_pred == 2)  # 部署链同上
        elif args.face == "classic":
            faces = classic_faces(p)  # 经典全链：全点+启发式（基线本貌）
        else:
            faces, mks = sem_faces(p, face_model, device)
            if faces is not None:
                face_pts = mks
        res = run_one(p, meta, faces,
                      (model_rb, model_sl, model_st), device,
                      face_pts=face_pts, conc_mask=conc_mask,
                      sl_fuse=(args.sleeve == "fusion"), rbst=args.rbst,
                      st_steel=st_steel)
        times.append(time.time() - t0)
        if res is None or res.get("face_fail"):
            per_file.append({"file": f, "skip": True,
                             "reason": "face_fail"})
            continue
        # 质心系 uv → 世界系（与 meta positions_yz 可比）
        res["rebar"]["uv_world"] += uv_gt_offset
        res["sleeve"]["uv_world"] += uv_gt_offset
        sl_vis = sleeve_visible(pts, lbls, meta)
        ev = eval_one(res, meta, sl_vis=sl_vis,
                      st_off=(float(c0[0]), float(c0[1])))
        # rb_gt 供聚合召回分母
        ev["rb_gt"] = int(meta["n_rebars"])
        per_file.append({"file": f, "eval": ev, "res": {
            "rb_n": res["rebar"]["n"], "sl_n": res["sleeve"]["n"]}})
    agg = aggregate(per_file)
    agg["耗时/件s"] = round(float(np.mean(times)), 2)
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    if args.out:
        out_path = anchor_project_path(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fo:
            json.dump({"aggregate": agg, "per_file": per_file}, fo,
                      ensure_ascii=False, indent=1, default=str)
        print(f"→ {out_path}")
    return {"aggregate": agg, "per_file": per_file}


def main():
    run_single(build_parser().parse_args())


if __name__ == "__main__":
    main()
