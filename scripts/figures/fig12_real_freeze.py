# -*- coding: utf-8 -*-
"""Fig12 真实场景识别效果——冻结管线现跑检出并缓存（Fig11 模式第一步）。

管线段逐字复刻 src/lib/pipeline/real_chain.py::run_real（141-286 行语义
+ 291-372 行 GT 对齐评估），把指标输出换成绘图工件落盘：
  全场景点 / 真值筋点（世界系）、构件系端面带点、检出（钢筋/套筒）、
  人工标注真值（对齐到检出系）、套筒 RANSAC 变换。

用法（项目根）：.venv/Scripts/python.exe -I temp/fig12/freeze_real.py
产出：out/cache/fig12_real_deploy.npz（rb_thr=0.3 定稿工作点）
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lib.utils import anchor_project_path
from lib.e2e.model_pillar import match_dets
from lib.pipeline.real_chain import (CLUSTER_EPS, CLUSTER_MIN, MIN_CLUSTER_N,
                                     MEMBER_BACK, MEMBER_LAT, TIP_MARGIN,
                                     load_heads, _rig_ransac_2d)
from lib.pipeline.single_chain import POS_GATE, run_one

RB_THR = 0.3                      # 定稿工作点（12/12 无假检）
R_SCAN_REAL = np.arange(9.0, 22.1, 2.0)


def main():
    import open3d as o3d
    from sklearn.cluster import DBSCAN
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from lib.training.train_bev import _fit_rigid_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # ── 场景与 GT 载入（同 run_real §1） ──
    pts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/scene_full_mm - Cloud.ply")).points).astype(np.float32)
    gts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/rebar_only.ply")).points).astype(np.float32)
    lbls = np.load(anchor_project_path("out/cache/real_labels.npz"))["lbls"]
    print(f"[场景] {len(pts)} 点，筋真值 {len(gts)} 点（分割读缓存）",
          flush=True)

    # ── 场景轴（同 §3） ──
    c = pts.mean(0)
    _, _, V = np.linalg.svd(pts[::20] - c, full_matrices=False)
    xa = V[0]
    if xa[int(np.argmax(np.abs(xa)))] < 0:
        xa = -xa
    zt = np.array([0.0, 0.0, 1.0])
    za = zt - float(zt @ xa) * xa
    za /= np.linalg.norm(za)
    ya = np.cross(za, xa)
    R = np.vstack([xa, ya, za])

    # ── 裁单构件（同 §4） ──
    rb = pts[lbls == 2]
    cl = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(rb)
    clusters = []
    for k in np.unique(cl.labels_):
        if k < 0:
            continue
        gk = rb[cl.labels_ == k]
        if len(gk) >= MIN_CLUSTER_N:
            tk = (gk - c) @ xa
            clusters.append((tk.min(), tk.max(), len(gk), gk.mean(0)))
    clusters.sort(key=lambda q: -q[2])
    anchor = clusters[0]
    grp = [q for q in clusters
           if q[0] <= anchor[1] + 800.0 and q[1] >= anchor[0] - 800.0]
    t_face = float(np.median([q[0] for q in grp]))
    t_tip = float(max(q[1] for q in grp))
    rb_center = np.mean([q[3] for q in grp], axis=0)
    t_all = (pts - c) @ xa
    rel = pts - rb_center
    t_rbc = rel @ xa
    lat = np.linalg.norm(rel - np.outer(t_rbc, xa), axis=1)
    win = (t_all > t_face - MEMBER_BACK) & (t_all < t_tip + TIP_MARGIN) \
        & (lat < MEMBER_LAT)
    crop, lbls_crop = pts[win], lbls[win]
    print(f"[裁剪] 簇群 {len(grp)}，构件子集 {len(crop)} 点", flush=True)

    # ── 构件系 + 面提取 + 检测（同 §5-8） ──
    c0 = crop.mean(0)
    p = ((crop - c0) @ R.T).astype(np.float32)
    g2 = ((gts - c0) @ R.T).astype(np.float64)
    from lib.e2e.face_pointgrow import extract_faces
    faces = extract_faces(p, lbls_crop, axis_hint=(1.0, 0.0, 0.0))
    assert faces is not None, "extract_faces 失败（canonical 从未失败，停下查）"
    x_rb = float(faces["rebar"]["c3"][0])
    x_sl = float(faces["sleeve"]["c3"][0])
    m1 = (lbls_crop == 2) | ((lbls_crop == 1) & (np.abs(p[:, 0] - x_rb) < 60.0))
    m2 = (lbls_crop == 1) & (np.abs(p[:, 0] - x_sl) < 60.0)
    top = faces.get("top")
    m3 = (((lbls_crop == 1) | (lbls_crop == 3))
          & (p[:, 2] > float(top["d"]) - 60.0)) if top is not None else None
    con_pk = p[lbls_crop == 1]
    near = con_pk[np.abs(con_pk[:, 0] - x_sl) < 400.0]
    face_pk = None
    if len(near) >= 100:
        edges = np.arange(x_sl - 400.0, x_sl + 401.0, 1.0)
        hist, _ = np.histogram(near[:, 0], bins=edges)
        face_pk = float(edges[int(np.argmax(hist))] + 0.5)
    fuse_kw = {"r_scan": R_SCAN_REAL, "rim_min": -9.0,
               "h_lo": -40.0, "h_hi": 45.0, "h_gate": 45.0, "face_pk": face_pk}
    models = load_heads("out/exp1_train/head_bev/best.pth",
                        "out/exp1_train/head_point/best.pth", device)
    res = run_one(p, None, faces, models, device, face_pts=(m1, m2, m3),
                  conc_mask=(lbls_crop == 1) | (lbls_crop == 2),
                  sl_fuse=True, rbst="point", fuse_kwargs=fuse_kw,
                  rb_thr=RB_THR)
    assert res is not None and not res.get("face_fail")
    det_rb = np.asarray(res["rebar"]["uv_world"], np.float32).reshape(-1, 2)
    expo_rb = np.asarray(res["rebar"]["expo"], np.float64)
    det_sl = np.asarray(res["sleeve"]["uv_world"], np.float32).reshape(-1, 2)
    r_in = np.asarray(res["sleeve"].get("r_in") or res["sleeve"]["spec"],
                      np.float64)
    print(f"[检测] 钢筋 {len(det_rb)}、套筒 {len(det_sl)}"
          f"（src={res['sleeve'].get('src', '裸头')}）", flush=True)

    # ── 钢筋 GT 对齐（同 §9：手标圆心 ↔ 筋点簇，刚体拟合 ≤30mm 采纳） ──
    with open(anchor_project_path("real_resource/rebar_gt_manual.json"),
              encoding="utf-8") as f:
        circ = np.array([[q["h2"], q["h3"]]
                         for q in json.load(f)["rebars"]], np.float64)
    clg = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(
        gts - gts.mean(0))
    glabs = [k for k in np.unique(clg.labels_) if k >= 0]
    gsizes = sorted(((int((clg.labels_ == k).sum()), k) for k in glabs),
                    reverse=True)
    gt_uv, gt_expo = [], []
    for _s, k in gsizes[:len(circ)]:
        gk = g2[clg.labels_ == k]
        bk = np.abs(gk[:, 0] - x_rb) < 80.0
        if bk.sum() < 5:
            bk = np.abs(gk[:, 0] - x_rb) < 150.0
        if bk.sum() < 5:
            bk = np.ones(len(gk), bool)
        gt_uv.append([gk[bk, 1].mean(), gk[bk, 2].mean()])
        gt_expo.append(float(gk[:, 0].max() - x_rb))
    gt_uv = np.array(gt_uv, np.float32)
    Rm, tm = np.eye(2), np.zeros(2)
    for _ in range(2):
        ri, ci = linear_sum_assignment(cdist(circ @ Rm.T + tm, gt_uv))
        Rm, tm = _fit_rigid_2d(circ[ri], gt_uv[ci])
    fit_res = float(np.linalg.norm(
        (circ @ Rm.T + tm)[ri] - gt_uv[ci], axis=1).mean())
    if fit_res < 30.0:
        gt_uv = (circ @ Rm.T + tm).astype(np.float32)
    si, gi = match_dets(det_rb, gt_uv, gate=POS_GATE)
    pos_err = [float(np.linalg.norm(det_rb[a] - gt_uv[b]))
               for a, b in zip(si, gi)]
    print(f"[钢筋] 检出{len(det_rb)} 匹配{len(si)} 拟合残差{fit_res:.1f}mm "
          f"位置p50={np.median(pos_err) if pos_err else float('nan'):.2f}mm",
          flush=True)

    # ── 套筒 GT 对齐（同 §10：成对 RANSAC，镜像可能） ──
    with open(anchor_project_path("real_resource/sleeve_gt_manual.json"),
              encoding="utf-8") as f:
        sj = json.load(f)
    sc = np.array([[q["h2"], q["h3"]] for q in sj["circles"]], np.float64)
    s_id = np.array([2.0 * float(q["r_mm"]) for q in sj["circles"]])
    rr = (_rig_ransac_2d(det_sl.astype(np.float64), sc, thr=8.0)
          if len(det_sl) >= 3 else None)
    inl, sl_R, sl_t, sl_mirror, pairs = rr
    D = cdist(det_sl.astype(np.float64) @ sl_R.T + sl_t, sc)
    ri2, ci2 = linear_sum_assignment(D)
    sel = D[ri2, ci2] < 8.0
    res_p = D[ri2[sel], ci2[sel]]
    id_err = [abs(float(r_in[a]) - float(s_id[b]))
              for a, b in zip(ri2[sel], ci2[sel])]
    print(f"[套筒] 检出{len(det_sl)} 内点{int(inl)} "
          f"残差p50={np.median(res_p):.2f}mm ID误差p50="
          f"{np.median(id_err):.2f}mm mirror={sl_mirror}", flush=True)

    # ── 绘图工件落盘 ──
    band_rb = p[np.abs(p[:, 0] - x_rb) < 60.0][:, 1:]
    band_sl = p[np.abs(p[:, 0] - x_sl) < 60.0][:, 1:]
    # 检出钢筋的世界系杆段（底=端面，顶=端面+外露长）
    base3 = np.column_stack([np.full(len(det_rb), x_rb), det_rb])
    tip3 = np.column_stack([x_rb + expo_rb, det_rb])
    seg_w = np.stack([(base3 @ R + c0), (tip3 @ R + c0)], axis=1)
    # 真值筋杆段（世界系）：筋点簇沿轴的 min→max
    gt_seg_w = []
    for _s, k in gsizes[:len(circ)]:
        gk = g2[clg.labels_ == k]
        gk_w = gk @ R + c0
        gt_seg_w.append([gk_w[:, 0].min(), gk_w[:, 0].max()])
    out = os.path.join(ROOT, "out", "cache", "fig12_real_deploy.npz")
    np.savez(out, pts=pts, gts=gts,
             band_rb=band_rb.astype(np.float32),
             band_sl=band_sl.astype(np.float32),
             det_rb=det_rb, expo_rb=expo_rb, det_sl=det_sl, r_in=r_in,
             gt_rb_uv=gt_uv, fit_res=np.float64(fit_res),
             gt_sl_sc=sc, sl_R=np.asarray(sl_R, np.float64),
             sl_t=np.asarray(sl_t, np.float64),
             sl_mirror=np.int64(sl_mirror),
             sl_pairs=np.array(pairs, np.int64).reshape(-1, 2),
             seg_w=seg_w.astype(np.float64),
             gt_seg_w=np.array(gt_seg_w, np.float64),
             x_rb=np.float64(x_rb), x_sl=np.float64(x_sl),
             R=R, c0=c0)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
