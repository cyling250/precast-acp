# -*- coding: utf-8 -*-
"""真实场景全链（T-040 修复清单#3，A4"裁单构件前置"降级路径）。

场景几何（2026-10-29 布局诊断实锤，俯视/侧视渲染 + 筋簇统计）：
  场景=单根躺放柱（截面 ~450×650mm，重力=z）+12 根外露带弯钩纵筋
  （外露 1520mm）+少量游离噪点，无地面密扫描；钢筋端面=筋簇 t-min
  （嵌入端），套筒端=构件另一端（t-min 侧，两级端面：基面+凸台，
  环顶凸出 10-30mm）。**train_e2e_point.eval_real_pt 的套筒窗锚在筋
  尖端（t_all.max()）系错锚**——历史真实套筒数字全部出自
  script_20261025_193000 的 t_min 侧配方，本脚本按后者实现。

评估协议（沿用 193000 脚本，画幅原点差 ~70mm 故映射式不可用）：
  钢筋=eval_real_pt 协议（rebar_only 簇心↔手绘圆两轮匈牙利+2D 刚体
  拟合，残差<30mm 采纳）；套筒=检出 vs 手绘 12 圆**成对 RANSAC 刚体
  配准（含镜像）**，指标=内点数/残差 p50/内点精度/ID 误差。

管线：整场景 ply → 预测材质分割（部署级、无 GT；标签落盘缓存）→
钢筋类点 DBSCAN 分簇定构件 → 轴对齐质心系裁剪（裁单构件前置）→
点级生长面 extract_faces → 复用 e2e_fullchain.run_one 三头+融合 →
手绘 GT 出数。A4 锚原文="免裁剪全链"（docs/t037_scene_face_pipeline.md
§7.2），免裁剪场景级路线=T-040 修复清单#4 另卡。

用法（cwd=src）：论文入口=experiments.exp5_real；本模块保留同参 CLI
作调试直入口：
  python -m lib.pipeline.real_chain --rb-thr 0.3 [--out results/real_fullchain.json]
  （canonical：results/real_fullchain.json，rb_thr=0.3 定稿工作点）
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from lib.utils import anchor_project_path
from lib.e2e.model_pillar import (match_dets, PillarBEVNet,
                                  REBAR_DIAM_TABLE, SLEEVE_IDIAM_TABLE,
                                  STIRRUP_DIAM_TABLE)
from lib.e2e.point_head import PointCircleNet
from lib.pipeline.single_chain import POS_GATE, run_one

MEMBER_LAT = 900.0      # 裁剪横向半径（截面 ~450 → 横向半跨 900 容噪）
MEMBER_BACK = 4500.0    # 出筋端面向 −t 回看距离（覆盖 ≤4.5m 构件长）
TIP_MARGIN = 300.0      # 筋簇尖端外延（簇群归并修复后 t_tip=真筋尖）
CLUSTER_EPS = 60.0
CLUSTER_MIN = 5
MIN_CLUSTER_N = 800     # 筋簇最小点数（实测 12 簇 7.4k-13.4k）


def load_heads(ckpt_path, pt_ckpt, device):
    """与 e2e_fullchain.main 同构的三头载入（钢筋/箍筋=点级头，套筒=BEV）。"""
    ck = torch.load(anchor_project_path(ckpt_path), map_location=device)
    pt = torch.load(anchor_project_path(pt_ckpt), map_location=device)
    ref_rad = int(pt.get("ref_radial", 0))
    model_rb = PointCircleNet(len(REBAR_DIAM_TABLE), bn=True, coarse=True,
                              refine=True, fg=True, in_extra=7,
                              ref_radial=ref_rad).to(device)
    model_rb.load_state_dict(pt["rb"])
    model_st = PointCircleNet(len(STIRRUP_DIAM_TABLE), bn=True, coarse=True,
                              refine=True, fg=True, in_extra=7,
                              ref_radial=ref_rad).to(device)
    model_st.load_state_dict(pt["st"])
    model_sl = PillarBEVNet(len(SLEEVE_IDIAM_TABLE), in_ch=8).to(device)
    model_sl.load_state_dict(ck["sl"])
    for m in (model_rb, model_sl, model_st):
        m.eval()
    return model_rb, model_sl, model_st


def _rig_ransac_2d(A, B, thr=8.0):
    """成对 RANSAC 刚体（含镜像），返回 (内点数, R, t, mirror, 内点对)。
    端口自 script_20261025_193000（真实 9/12 锚点协议）。"""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    best = None
    for mirror in (False, True):
        Am = A * np.array([1.0, -1.0]) if mirror else A
        n = len(Am)
        for i in range(n):
            for j in range(i + 1, n):
                d1 = Am[j] - Am[i]
                n1 = float(np.linalg.norm(d1))
                if n1 < 30:
                    continue
                for k in range(len(B)):
                    for l in range(k + 1, len(B)):
                        d2 = B[l] - B[k]
                        n2 = float(np.linalg.norm(d2))
                        if n2 < 30 or abs(n1 - n2) / n1 > 0.15:
                            continue
                        cc = float(np.dot(d1, d2) / (n1 * n2))
                        ss = (d1[0] * d2[1] - d1[1] * d2[0]) / (n1 * n2)
                        R1 = np.array([[cc, -ss], [ss, cc]])
                        tv = B[k] - R1 @ Am[i]
                        D = cdist(Am @ R1.T + tv, B)
                        ri, ci = linear_sum_assignment(D)
                        inl = int((D[ri, ci] < thr).sum())
                        if best is None or inl > best[0]:
                            sel = D[ri, ci] < thr
                            P = Am[ri[sel]] @ R1.T + tv
                            Q = B[ci[sel]]
                            mu_p, mu_q = P.mean(0), Q.mean(0)
                            H = (P - mu_p).T @ (Q - mu_q)
                            U, _s, Vt = np.linalg.svd(H)
                            dg = np.sign(np.linalg.det(Vt.T @ U.T))
                            R2 = Vt.T @ np.diag([1.0, dg]) @ U.T
                            t2 = mu_q - R2 @ mu_p
                            best = (inl, R2 @ R1, R2 @ tv + t2, mirror,
                                    (ri[sel], ci[sel]))
    if best is None:
        return None
    inl, R, t, mirror, pairs = best
    return inl, R, t, mirror, pairs


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/exp1_train/head_bev/best.pth")
    ap.add_argument("--pt-ckpt", default="out/exp1_train/head_point/best.pth")
    ap.add_argument("--seg-ckpt", default="",
                    help="空=lib.segmentation.inference 默认权重")
    ap.add_argument("--sleeve", choices=("fusion", "raw"), default="fusion")
    ap.add_argument("--rb-thr", type=float, default=0.3,
                    help="钢筋点级头阈值（真实域工作点 0.2-0.3 实测定稿"
                         " 0.3：12/12 无假检；0.2 多 2 假检换 0.85mm）")
    ap.add_argument("--refresh-labels", action="store_true",
                    help="强制重跑分割（默认读 out/cache/real_labels.npz）")
    ap.add_argument("--out", default="",
                    help="结果 json 路径（canonical=results/real_fullchain.json，"
                         "默认只打印不落盘防覆盖）")
    return ap


def run_real(args):
    """真实场景全链（原 main 本体，参数化）。返回 out dict。"""
    import open3d as o3d
    from sklearn.cluster import DBSCAN
    from lib.segmentation.inference import load_model, predict_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ── 1. 场景与 GT 载入 ──
    pts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/scene_full_mm - Cloud.ply")).points).astype(np.float32)
    gts = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/rebar_only.ply")).points).astype(np.float32)
    print(f"[场景] {len(pts)} 点，筋 GT {len(gts)} 点", flush=True)

    # ── 2. 预测材质分割（部署级；落盘缓存） ──
    lbl_cache = anchor_project_path("out/cache/real_labels.npz")
    if os.path.exists(lbl_cache) and not args.refresh_labels:
        lbls = np.load(lbl_cache)["lbls"]
        print(f"[分割] 读缓存 {lbl_cache}", flush=True)
    else:
        seg_model, seg_dev = load_model(args.seg_ckpt or None)
        t0 = time.time()
        lbls = predict_labels(pts, seg_model, seg_dev)
        print(f"[分割] predict_labels {time.time()-t0:.0f}s，类分布 "
              f"{np.bincount(lbls, minlength=4)}", flush=True)
        np.savez(lbl_cache, lbls=lbls)

    # ── 3. 场景轴（全场景 SVD，无 GT；z 为重力） ──
    c = pts.mean(0)
    _, _, V = np.linalg.svd(pts[::20] - c, full_matrices=False)
    xa = V[0]
    if xa[int(np.argmax(np.abs(xa)))] < 0:
        xa = -xa
    zt = np.array([0.0, 0.0, 1.0])
    za = zt - float(zt @ xa) * xa
    za /= np.linalg.norm(za)
    ya = np.cross(za, xa)
    R = np.vstack([xa, ya, za])            # p' = (p−c0) @ R.T

    # ── 4. 裁单构件前置：预测筋簇 → 构件轴窗 + 横向半径 ──
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
    print(f"[裁剪] 预测筋簇 {len(clusters)} 个（≥{MIN_CLUSTER_N} 点）",
          flush=True)
    if not clusters:
        print("[裁剪] 无筋簇 → 终止")
        return
    # 同构件簇群：t 区间与锚簇重叠/间隙 <800mm 归并（首版用
    # min(|lo−hi|,|hi−lo|) 把重度重叠区间判成远距，P-061 跨系家族）
    anchor = clusters[0]
    grp = [q for q in clusters
           if q[0] <= anchor[1] + 800.0 and q[1] >= anchor[0] - 800.0]
    t_face = float(np.median([q[0] for q in grp]))    # 出筋端面=簇 t-min 中位
    t_tip = float(max(q[1] for q in grp))
    rb_center = np.mean([q[3] for q in grp], axis=0)
    print(f"[裁剪] 簇群 {len(grp)}，端面 t≈{t_face:.0f}，tip {t_tip:.0f}"
          f"（外露 {t_tip - t_face:.0f}mm）", flush=True)
    # 轴窗与簇统计同用 c 系（首版 t_all 用 rb_center 系、t_face 用 c 系，
    # 窗整体错位 ~2.8m——P-061「跨系传参第一次运行就该断言」再实锤）；
    # 横向距离在 rb_center 系自内洽（轴线=过筋簇质心的 xa 方向线，
    # c 被 56% 背景点拖离轴线，不能用作横向参考原点）
    t_all = (pts - c) @ xa
    rel = pts - rb_center
    t_rbc = rel @ xa
    lat = np.linalg.norm(rel - np.outer(t_rbc, xa), axis=1)
    win = (t_all > t_face - MEMBER_BACK) & (t_all < t_tip + TIP_MARGIN) \
        & (lat < MEMBER_LAT)
    crop, lbls_crop = pts[win], lbls[win]
    print(f"[裁剪] 构件子集 {len(crop)} 点（全场景 "
          f"{len(crop)/len(pts):.0%}）", flush=True)

    # ── 5. 轴对齐质心系（x=轴, z=重力） ──
    c0 = crop.mean(0)
    p = ((crop - c0) @ R.T).astype(np.float32)
    g2 = ((gts - c0) @ R.T).astype(np.float64)     # 筋 GT 同系

    # ── 6. 面提取（点级生长；失败回退簇/极值端面） ──
    from lib.e2e.face_pointgrow import extract_faces
    faces = extract_faces(p, lbls_crop, axis_hint=(1.0, 0.0, 0.0))
    if faces is None:
        print("[面] extract_faces 失败 → 回退簇/极值端面", flush=True)
        # 端面位置一律在裁剪系内由预测筋/混凝土点自估（跨系换算曾少加
        # c@xa 项致端面错位 3m，2026-10-29 首跑实锤）
        rb_p = p[lbls_crop == 2]
        x_rb = (float(np.percentile(rb_p[:, 0], 1.0)) if len(rb_p)
                else float(np.percentile(p[:, 0], 2.0)))
        band = np.abs(p[:, 0] - x_rb) < 60.0
        c3r = p[band].mean(0) if band.sum() >= 10 else p.mean(0)
        con = p[lbls_crop == 1]
        smin = float(np.percentile(con[:, 0], 0.2)) if len(con) \
            else float(np.percentile(p[:, 0], 0.2))
        bands = p[:, 0] < smin + 200.0
        c3s = p[bands].mean(0)
        top_d = float(np.percentile(con[:, 2], 99)) if len(con) else None
        faces = {"rebar": {"n3": np.array([1.0, 0.0, 0.0]),
                           "d": float(c3r @ np.array([1.0, 0.0, 0.0])),
                           "c3": c3r},
                 "sleeve": {"n3": np.array([-1.0, 0.0, 0.0]),
                            "d": float(c3s @ np.array([-1.0, 0.0, 0.0])),
                            "c3": c3s},
                 "top": ({"d": top_d} if top_d is not None else None),
                 "axis": np.array([1.0, 0.0, 0.0])}
    print(f"[面] rebar d={faces['rebar']['d']:.0f} "
          f"sleeve d={faces['sleeve']['d']:.0f} "
          f"top={'有' if faces.get('top') else '无'}", flush=True)

    # ── 7. 部署带掩码（无 meta 的 face_band_labels 适配） ──
    x_rb = float(faces["rebar"]["c3"][0])
    x_sl = float(faces["sleeve"]["c3"][0])
    m1 = (lbls_crop == 2) | ((lbls_crop == 1)
                             & (np.abs(p[:, 0] - x_rb) < 60.0))
    m2 = (lbls_crop == 1) & (np.abs(p[:, 0] - x_sl) < 60.0)
    top = faces.get("top")
    m3 = (((lbls_crop == 1) | (lbls_crop == 3))
          & (p[:, 2] > float(top["d"]) - 60.0)) if top is not None else None

    # ── 8. 三头检测 + 融合（真实域融合常量：script_20261025_193000
    #      实证——孔 r≈13.4、环有凹陷个体、两级端面带跨基面+平台+环顶、
    #      h_gate=45 为几何源 1→9 的关键修复；face_pk=端部轴向直方图
    #      峰值层（平台层旁路，防粗峰锁错层）） ──
    con_pk = p[lbls_crop == 1]
    x_sl0 = float(faces["sleeve"]["c3"][0])
    near = con_pk[np.abs(con_pk[:, 0] - x_sl0) < 400.0]
    face_pk = None
    if len(near) >= 100:
        edges = np.arange(x_sl0 - 400.0, x_sl0 + 401.0, 1.0)
        hist, _ = np.histogram(near[:, 0], bins=edges)
        face_pk = float(edges[int(np.argmax(hist))] + 0.5)
    R_SCAN_REAL = np.arange(9.0, 22.1, 2.0)
    fuse_kw = {"r_scan": R_SCAN_REAL, "rim_min": -9.0,
               "h_lo": -40.0, "h_hi": 45.0, "h_gate": 45.0,
               "face_pk": face_pk}
    print(f"[融合] face_pk={face_pk}（face_x={x_sl0:.0f}）", flush=True)
    model_rb, model_sl, model_st = load_heads(args.ckpt, args.pt_ckpt, device)
    t0 = time.time()
    res = run_one(p, None, faces, (model_rb, model_sl, model_st), device,
                  face_pts=(m1, m2, m3),
                  conc_mask=(lbls_crop == 1) | (lbls_crop == 2),
                  sl_fuse=(args.sleeve == "fusion"), rbst="point",
                  fuse_kwargs=fuse_kw, rb_thr=args.rb_thr)
    print(f"[检测] 全链 {time.time()-t0:.1f}s，钢筋 {res['rebar']['n']}、"
          f"套筒 {res['sleeve']['n']}"
          f"（src={res['sleeve'].get('src', '裸头')}）", flush=True)
    if res.get("face_fail"):
        print("[检测] face_fail → 终止")
        return

    # ── 9. 评估：钢筋（eval_real_pt 协议，旋转系） ──
    out = {"裁剪簇群": len(grp), "构件点数": int(len(crop)),
           "分割类分布": np.bincount(lbls_crop, minlength=4).tolist()}
    clg = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(
        gts - gts.mean(0))
    glabs = [k for k in np.unique(clg.labels_) if k >= 0]
    gsizes = sorted(((int((clg.labels_ == k).sum()), k) for k in glabs),
                    reverse=True)
    with open(anchor_project_path("real_resource/rebar_gt_manual.json"),
              encoding="utf-8") as f:
        circ = np.array([[q["h2"], q["h3"]]
                         for q in json.load(f)["rebars"]], np.float64)
    fr_x = x_rb
    gt_uv, gt_expo = [], []
    for _s, k in gsizes[:len(circ)]:
        gk = g2[clg.labels_ == k]
        bk = np.abs(gk[:, 0] - fr_x) < 80.0
        if bk.sum() < 5:
            bk = np.abs(gk[:, 0] - fr_x) < 150.0
        if bk.sum() < 5:               # 端面带内无 GT 点 → 整簇 (y,z) 均值
            bk = np.ones(len(gk), bool)
        gt_uv.append([gk[bk, 1].mean(), gk[bk, 2].mean()])
        gt_expo.append(float(gk[:, 0].max() - fr_x))
    gt_uv = np.array(gt_uv, np.float32)
    from lib.training.train_bev import _fit_rigid_2d
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    Rm, tm = np.eye(2), np.zeros(2)
    for _ in range(2):
        ri, ci = linear_sum_assignment(cdist(circ @ Rm.T + tm, gt_uv))
        Rm, tm = _fit_rigid_2d(circ[ri], gt_uv[ci])
    fit_res = float(np.linalg.norm(
        (circ @ Rm.T + tm)[ri] - gt_uv[ci], axis=1).mean())
    if fit_res < 30.0:
        gt_uv = (circ @ Rm.T + tm).astype(np.float32)
        gt_n = len(circ)
    else:
        gt_n = len(gt_uv)
    det_rb = np.asarray(res["rebar"]["uv_world"], np.float32).reshape(-1, 2)
    si, gi = match_dets(det_rb, gt_uv, gate=POS_GATE)
    pos_err = [float(np.linalg.norm(det_rb[a] - gt_uv[b]))
               for a, b in zip(si, gi)]
    expo_err = [abs(float(res["rebar"]["expo"][a]) - gt_expo[b])
                for a, b in zip(si, gi)]
    med = lambda x: (round(float(np.median(x)), 2) if len(x) else None)
    out["钢筋"] = {"GT根数": gt_n, "检出": len(det_rb), "匹配": len(si),
                   "标注拟合残差mm": round(fit_res, 1),
                   "位置p50mm": med(pos_err),
                   "位置p90mm": (round(float(np.percentile(pos_err, 90)), 2)
                                 if pos_err else None),
                   "外露量测误差p50mm": med(expo_err)}
    print(f"[钢筋] {out['钢筋']}", flush=True)

    # ── 10. 评估：套筒（成对 RANSAC 配准，193000 协议） ──
    det_sl = np.asarray(res["sleeve"]["uv_world"], np.float32).reshape(-1, 2)
    with open(anchor_project_path("real_resource/sleeve_gt_manual.json"),
              encoding="utf-8") as f:
        sj = json.load(f)
    sc = np.array([[q["h2"], q["h3"]] for q in sj["circles"]], np.float64)
    s_id = np.array([2.0 * float(q["r_mm"]) for q in sj["circles"]])
    rr = (_rig_ransac_2d(det_sl.astype(np.float64), sc, thr=8.0)
          if len(det_sl) >= 3 else None)
    if rr is None:
        out["套筒"] = {"检出": len(det_sl), "GT环数": len(sc), "配准内点": 0}
    else:
        inl, Rs, ts, mirror, _pairs = rr
        D = cdist(det_sl.astype(np.float64) @ Rs.T + ts, sc)
        ri2, ci2 = linear_sum_assignment(D)
        sel = D[ri2, ci2] < 8.0
        res_p = D[ri2[sel], ci2[sel]]
        # ID 用连续内径（融合 r_in）优先——离散规格表在真实域差 ~9mm
        # （真实孔 ID≈27mm 不在合成表档位附近，首跑实锤）
        id_hat = (np.asarray(res["sleeve"]["r_in"], float)
                  if res["sleeve"].get("r_in")
                  else np.asarray(res["sleeve"]["spec"], float))
        id_err = [abs(float(id_hat[a]) - float(s_id[b]))
                  for a, b in zip(ri2[sel], ci2[sel])] if sel.any() else []
        out["套筒"] = {"检出": len(det_sl), "GT环数": len(sc),
                       "配准内点": int(inl), "内点残差p50mm": med(res_p),
                       "内点精度": round(inl / max(len(det_sl), 1), 3),
                       "ID误差p50mm": med(id_err)}
    print(f"[套筒] {out['套筒']}", flush=True)

    # 箍筋链状态（真实场景无箍筋 GT，只记录链路健康度）
    out["箍筋链"] = ({"n_det": res["stirrup"]["n_det"]}
                     if res.get("stirrup") else None)
    if args.out:
        op = anchor_project_path(args.out)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op, "w", encoding="utf-8") as fo:
            json.dump(out, fo, ensure_ascii=False, indent=1, default=str)
        print(f"→ {op}")
    return out


def main():
    run_real(build_parser().parse_args())


if __name__ == "__main__":
    main()
