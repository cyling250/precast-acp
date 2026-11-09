# -*- coding: utf-8 -*-
"""论文定性图渲染（一次性工具，复用冻结管线函数跑检出，不改动管线本体）。

产出（--out-dir，默认 manuscript/figures/qualitative/）：
  single_<stem>.png   图A：单件部署链——材质预测染色侧视 + 两端面检出 vs GT
  faces_<stem>.png    图C：面实例可视化——三连接面位置/法向（构件系侧视）
  scene_<mode>.png    图B：场景链——场景俯视材质染色 + 逐构件端面检出 vs GT
  real_scene.png      图D：真实场景——扫描云侧视 + 手绘 GT 筋点（canonical
                      数字在正文表，检出叠加不做：帧变换收益低）

检出全部由 lib.pipeline 冻结函数现跑（部署口径 pointgrow-pred + fusion），
只用 GT 画对照——图不进任何指标表。GPU 空闲时运行（分割/检测推理）。

用法（cwd=src）：
  python -m scripts.render_qualitative [--singles 2] [--scenes s3,s4] [--real]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from lib.utils import anchor_project_path

C_CONC = "#c8c8c8"
C_RB = "#d62728"
C_ST = "#ff9f1c"
C_GT = "#222222"
C_DET = "#1f77b4"
C_SL = "#2ca02c"


def _scatter(ax, p, c, s=1.0, alpha=0.5, **kw):
    ax.scatter(p[:, 0], p[:, 1], c=c, s=s, alpha=alpha,
               linewidths=0, **kw)


def _material_colors(lbls):
    palette = {0: "#eeeeee", 1: C_CONC, 2: C_RB, 3: C_ST}
    return [palette.get(int(l), "#dddddd") for l in lbls]


def _subsample(pts, cap=60000, seed=0):
    if len(pts) <= cap:
        return np.arange(len(pts))
    rng = np.random.default_rng(seed)
    return rng.choice(len(pts), cap, replace=False)


def load_models(device):
    from lib.pipeline.real_chain import load_heads
    from lib.segmentation.inference import load_model
    models = load_heads("out/exp1_train/head_bev/best.pth",
                        "out/exp1_train/head_point/best.pth", device)
    seg_model, seg_dev = load_model()
    return models, seg_model, seg_dev


def run_single_deploy(pts, meta, models, seg_model, seg_dev, device):
    """单件部署口径现跑（=single_chain pointgrow-pred 臂的组成，零新逻辑）。"""
    from lib.pipeline.single_chain import run_one
    from lib.e2e.face_pointgrow import extract_faces
    from lib.e2e.face_sem import face_band_labels
    c0 = pts.mean(0)
    p = pts - c0
    lbls = predict(pts, seg_model, seg_dev)
    faces = extract_faces(p, lbls)
    if faces is None:
        return None
    lab = face_band_labels(pts, lbls, meta)
    face_pts = (lab == 1, lab == 2, lab == 3)
    res = run_one(p, meta, faces, models, device, face_pts=face_pts,
                  conc_mask=(lbls == 1) | (lbls == 2), sl_fuse=True,
                  rbst="point", st_steel=(lbls == 3))
    if res is None or res.get("face_fail"):
        return None
    res["rebar"]["uv_world"] = np.asarray(
        res["rebar"]["uv_world"], np.float32) + c0[1:]
    res["sleeve"]["uv_world"] = np.asarray(
        res["sleeve"]["uv_world"], np.float32) + c0[1:]
    return {"res": res, "lbls": lbls, "c0": c0, "p": p, "faces": faces}


def predict(pts, model, device):
    from lib.segmentation.inference import predict_labels
    return predict_labels(pts, model, device)


def plot_endface(ax, pts, x_band, gt_yz, gt_inner, det_yz, det_diam, title):
    """端面 (y,z) 视图：点 + GT（黑虚线圆/黑+）vs 检出（实线圆）。"""
    band = np.abs(pts[:, 0] - x_band) < 60.0
    idx = _subsample(pts[band], 20000)
    q = pts[band][idx]
    _scatter(ax, q[:, 1:], C_CONC, s=1.2)
    for (gy, gz), di in zip(np.asarray(gt_yz, np.float32),
                            np.asarray(gt_inner, np.float32)):
        ax.add_patch(plt.Circle((gy, gz), di / 2, fill=False, ls="--",
                                ec=C_GT, lw=1.0, alpha=0.9))
        ax.plot([gy], [gz], "+", color=C_GT, ms=7, mew=1.4)
    for (dy, dz), dd in zip(np.asarray(det_yz, np.float32).reshape(-1, 2),
                            np.asarray(det_diam, np.float32)):
        ax.add_patch(plt.Circle((dy, dz), dd / 2, fill=False, ec=C_DET,
                                lw=1.6))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("y (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_single(f_path, models, seg_model, seg_dev, device, out_dir):
    from lib.e2e.face_sem import face_band_labels
    stem = os.path.splitext(os.path.basename(f_path))[0]
    d = np.load(f_path, allow_pickle=False)
    pts = d["points"].astype(np.float32)
    meta = json.loads(str(d["meta_json"]))
    got = run_single_deploy(pts, meta, models, seg_model, seg_dev, device)
    if got is None:
        print(f"[single] {stem}: 链失败，跳过")
        return None
    res, lbls, c0, p = got["res"], got["lbls"], got["c0"], got["p"]
    idx = _subsample(pts)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axs[0]
    _scatter(ax, np.column_stack([pts[idx, 0], pts[idx, 2]]),
             [_material_colors(lbls[idx])[i] for i in range(len(idx))], s=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{stem}\n部署链材质预测（灰=混凝土 红=钢筋 橙=箍筋）",
                 fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    # 套筒端面
    x_sl = float(meta["sleeve_end_x"])
    gt_yz = [[g["center_yz"][0], g["center_yz"][1]]
             for g in meta.get("sleeve_gt") or []]
    gt_inner = [g.get("inner_diam", 40.0) for g in meta.get("sleeve_gt") or []]
    plot_endface(axs[1], pts, x_sl, gt_yz, gt_inner,
                 res["sleeve"]["uv_world"], res["sleeve"]["spec"],
                 f"套筒端面：检出 {res['sleeve']['n']} 环（蓝实线）"
                 f" vs GT {len(gt_yz)}（黑虚线）")
    # 出筋端面
    x_rb = float(meta["rebar_end_x"])
    gt_yz = np.asarray(meta["rebar_positions_yz"], np.float32)
    plot_endface(axs[2], pts, x_rb, gt_yz,
                 [meta["rebar_diam"]] * len(gt_yz),
                 res["rebar"]["uv_world"],
                 res["rebar"]["spec"],
                 f"出筋端面：检出 {res['rebar']['n']} 根（蓝圈）"
                 f" vs GT {len(gt_yz)}（黑+）")
    fig.tight_layout()
    out = os.path.join(out_dir, f"single_{stem}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[single] → {out}")

    # 面实例可视化（同一件，侧视 + 三连接面）
    lab = face_band_labels(pts, lbls, meta)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    band_m = (lab == 1) | (lab == 2) | (lab == 3)
    idx2 = _subsample(pts[band_m], 40000)
    q = pts[band_m][idx2]
    cols = np.array([C_RB, C_SL, C_ST])[lab[band_m][idx2] - 1]
    _scatter(ax, q[:, [0, 2]], cols, s=1.0)
    faces = got["faces"]
    for key, col, name in (("rebar", C_RB, "出筋面"),
                           ("sleeve", C_SL, "套筒面")):
        f = faces[key]
        n3 = np.asarray(f["n3"], np.float64)
        n3 = n3 / np.linalg.norm(n3)
        c3 = np.asarray(f["c3"], np.float64) + c0
        if abs(float(n3[0])) < 0.5:
            continue
        z0, z1 = np.percentile(pts[:, 2], [1, 99])
        ax.plot([c3[0], c3[0]], [z0, z1], color=col, lw=2.0)
        ax.annotate(name, (c3[0], z1), color=col, fontsize=8,
                    ha="center", va="bottom")
    top = faces.get("top")
    if top is not None:
        x0, x1 = np.percentile(pts[:, 0], [1, 99])
        ax.plot([x0, x1], [float(top["d"]) + c0[2]] * 2, color=C_ST,
                lw=2.0)
        ax.annotate("箍筋顶面", (x1, float(top["d"]) + c0[2]), color=C_ST,
                    fontsize=8, ha="right", va="bottom")
    ax.set_aspect("equal")
    ax.set_title("连接面实例（点色=GT 面带：红出筋 绿套筒 橙箍筋；"
                 "线=生长面位置）", fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(out_dir, f"faces_{stem}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[faces] → {out}")
    return got


def fig_scene(mode, name, models, seg_model, seg_dev, device, out_dir,
              max_members=3):
    """场景图：俯视材质染色 + 前 max_members 个构件端面检出 vs GT。"""
    from lib.pipeline.scene_chain import (crop_member_best, member_frame,
                                          member_meta)
    from lib.e2e.face_pointgrow import extract_faces
    from lib.pipeline.scene_chain import _fix_end_flip
    scene_npz = anchor_project_path(f"mock/scene/v1/{mode}/{name}.npz")
    z = np.load(scene_npz, allow_pickle=True)
    vis = z["visible"].astype(bool)
    pts = z["points"][vis].astype(np.float32)
    member_id = z["member_id"][vis].astype(np.int64)
    mts = json.loads(str(z["member_table_json"]))
    lbl_cache = anchor_project_path(
        f"out/cache/scene_batch/{mode}_{name}_labels.npz")
    lbls = np.load(lbl_cache)["lbls"]
    print(f"[scene {mode}] {len(pts)} 点，{len(mts)} 构件 GT", flush=True)

    # 分簇归组（与 scene_chain.main 同序）
    from sklearn.cluster import DBSCAN
    from lib.pipeline.real_chain import CLUSTER_EPS, CLUSTER_MIN, MIN_CLUSTER_N
    rb_idx = np.flatnonzero(lbls == 2)
    rb = pts[rb_idx]
    cl = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(rb)
    by_mid = {}
    for k in np.unique(cl.labels_):
        if k >= 0 and (cl.labels_ == k).sum() >= MIN_CLUSTER_N:
            gidx = rb_idx[cl.labels_ == k]
            gv = member_id[gidx]
            gv = gv[gv >= 0]
            if not len(gv):
                continue
            mid = int(np.bincount(gv, minlength=len(mts)).argmax())
            by_mid.setdefault(mid, []).append(gidx)
    c_center = pts.mean(0)
    panels = []
    for mid in sorted(by_mid)[:max_members]:
        gidx_list = by_mid[mid]
        got = crop_member_best(pts, lbls, c_center,
                               [pts[g] for g in gidx_list], gidx_list)
        if got is None:
            continue
        (sub, lbls_sub, _widx, xa, _tf, _tt, _n), _faces0, info = got
        fr = member_frame(sub, xa)
        if fr is None:
            continue
        p0, R, c0 = fr
        faces = extract_faces(p0, lbls_sub, axis_hint=(1.0, 0.0, 0.0))
        if faces is None:
            continue
        faces = _fix_end_flip(faces, p0, lbls_sub,
                              pts[np.unique(np.concatenate(gidx_list))],
                              c0, xa)
        mt = mts[mid]
        meta = member_meta(mt, R, c0)
        from lib.pipeline.single_chain import run_one
        x_rb = float(faces["rebar"]["c3"][0])
        x_sl = float(faces["sleeve"]["c3"][0])
        m1 = (lbls_sub == 2) | ((lbls_sub == 1)
                                & (np.abs(p0[:, 0] - x_rb) < 60.0))
        m2 = (lbls_sub == 1) & (np.abs(p0[:, 0] - x_sl) < 60.0)
        top = faces.get("top")
        m3 = None
        if top is not None:
            n3t = np.asarray(top.get("n3", (0.0, 0.0, 1.0)), np.float64)
            nn = n3t[1:] / max(float(np.linalg.norm(n3t[1:])), 1e-9)
            s = p0[:, 1] * nn[0] + p0[:, 2] * nn[1] - float(top["d"])
            m3 = ((lbls_sub == 1) | (lbls_sub == 3)) & (s > -60.0)
        res = run_one(p0, None, faces, models, device, face_pts=(m1, m2, m3),
                      conc_mask=(lbls_sub == 1) | (lbls_sub == 2),
                      sl_fuse=True, rbst="point", st_steel=(lbls_sub == 3))
        if res is None or res.get("face_fail"):
            continue
        panels.append({"mid": mid, "info": info, "sub": sub,
                       "lbls_sub": lbls_sub, "p0": p0, "meta": meta,
                       "res": res, "faces": faces})
        print(f"  m{mid}: rb {res['rebar']['n']} sl {res['sleeve']['n']}",
              flush=True)

    # 版面：1 行俯视 + 每构件一行两列端面
    n_mem = len(panels)
    fig = plt.figure(figsize=(12.0, 3.0 + 2.6 * n_mem))
    gs = fig.add_gridspec(n_mem + 1, 2, height_ratios=[1.2] + [1] * n_mem)
    ax = fig.add_subplot(gs[0, :])
    idx = _subsample(pts, 100000)
    _scatter(ax, pts[idx][:, [0, 1]],
             [_material_colors(lbls[idx])[i] for i in range(len(idx))], s=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"场景 {mode}/{name} 俯视：部署分割材质"
                 f"（{len(mts)} 构件 GT；灰=混凝土 红=钢筋 橙=箍筋）",
                 fontsize=10)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    for i, pa in enumerate(panels):
        # 出筋端
        ax1 = fig.add_subplot(gs[i + 1, 0])
        meta, res = pa["meta"], pa["res"]
        x_rb = float(pa["faces"]["rebar"]["c3"][0])
        band = np.abs(pa["p0"][:, 0] - x_rb) < 60.0
        idxb = _subsample(pa["p0"][band], 15000)
        _scatter(ax1, pa["p0"][band][idxb][:, 1:], C_CONC, s=1.0)
        gt_yz = np.asarray(meta["rebar_positions_yz"], np.float32)
        ax1.plot(gt_yz[:, 0], gt_yz[:, 1], "+", color=C_GT, ms=8, mew=1.5,
                 ls="none")
        det = np.asarray(res["rebar"]["uv_world"], np.float32).reshape(-1, 2)
        ax1.plot(det[:, 0], det[:, 1], "o", color=C_DET, ms=5,
                 mfc="none", mew=1.5, ls="none")
        ax1.set_aspect("equal")
        ax1.set_title(f"m{pa['mid']}（{pa['info']['tag']}）出筋端："
                      f"检出 {len(det)} vs GT {len(gt_yz)}"
                      f"（黑+=GT 蓝圈=检出）", fontsize=8)
        ax1.tick_params(labelsize=6)
        # 套筒端
        ax2 = fig.add_subplot(gs[i + 1, 1])
        x_sl = float(pa["faces"]["sleeve"]["c3"][0])
        band = np.abs(pa["p0"][:, 0] - x_sl) < 60.0
        idxb = _subsample(pa["p0"][band], 15000)
        _scatter(ax2, pa["p0"][band][idxb][:, 1:], C_CONC, s=1.0)
        sls = meta.get("sleeve_gt") or []
        for g in sls:
            gy, gz = g["center_yz"]
            ax2.add_patch(plt.Circle((gy, gz), g.get("inner_diam", 40) / 2,
                                     fill=False, ls="--", ec=C_GT, lw=1.0))
            ax2.add_patch(plt.Circle((gy, gz), g.get("outer_diam", 60) / 2,
                                     fill=False, ls=":", ec=C_GT, lw=0.8))
        det_s = np.asarray(res["sleeve"]["uv_world"],
                           np.float32).reshape(-1, 2)
        diam_s = np.asarray(res["sleeve"]["spec"], np.float32)
        for (dy, dz), dd in zip(det_s, diam_s):
            ax2.add_patch(plt.Circle((dy, dz), dd / 2, fill=False,
                                     ec=C_DET, lw=1.6))
        ax2.set_aspect("equal")
        ax2.set_title(f"m{pa['mid']} 套筒端：检出 {len(det_s)} vs GT 可见"
                      f" {len(sls)}（黑虚=GT 蓝实=检出）", fontsize=8)
        ax2.tick_params(labelsize=6)
    fig.tight_layout()
    out = os.path.join(out_dir, f"scene_{mode}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[scene {mode}] → {out}")


def fig_real(out_dir):
    """真实场景：扫描云侧视 + 手绘 GT 筋点（canonical 数字在正文表）。"""
    import open3d as o3d
    p1 = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/scene_full_mm - Cloud.ply")).points).astype(np.float32)
    p2 = np.asarray(o3d.io.read_point_cloud(anchor_project_path(
        "real_resource/rebar_only.ply")).points).astype(np.float32)
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    idx = _subsample(p1, 120000)
    _scatter(ax, p1[idx][:, [0, 2]], C_CONC, s=0.5, alpha=0.35)
    idx2 = _subsample(p2, 30000)
    _scatter(ax, p2[idx2][:, [0, 2]], C_RB, s=0.7, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("真实场景（1.73M 点）：灰=扫描云，红=外露纵筋（GT 提取点）\n"
                 "canonical：钢筋 12/12 检出全匹配（位置 p50 7.29mm），"
                 "套筒 9 检出/8 配准内点", fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(out_dir, "real_scene.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[real] → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", type=int, default=2,
                    help="单件图数量（test 排序前 N）")
    ap.add_argument("--single-files", default="",
                    help="逗号分隔指定件名（优先于 --singles）")
    ap.add_argument("--scenes", default="s3,s4",
                    help="场景工况（每工况取 canonical 第一件）")
    ap.add_argument("--max-members", type=int, default=3)
    ap.add_argument("--real", action="store_true", default=True)
    ap.add_argument("--out-dir", default="manuscript/figures/qualitative")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = anchor_project_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    models, seg_model, seg_dev = load_models(device)

    if args.single_files:
        files = [anchor_project_path("mock/inuse/test/" + f.strip() + ".npz")
                 for f in args.single_files.split(",") if f.strip()]
    else:
        files = sorted(glob.glob(
            anchor_project_path("mock/inuse/test/*.npz")))[:args.singles]
    for f in files:
        fig_single(f, models, seg_model, seg_dev, device, out_dir)

    for mode in [m for m in args.scenes.split(",") if m]:
        files_m = sorted(glob.glob(anchor_project_path(
            f"mock/scene/v1/{mode}/*.npz")))
        cfgs = {}
        for f in files_m:
            base = os.path.basename(f).replace(".npz", "")
            cfgs.setdefault(base.split("_")[2], base)
        name = cfgs[sorted(cfgs)[0]]
        fig_scene(mode, name, models, seg_model, seg_dev, device, out_dir,
                  max_members=args.max_members)

    if args.real:
        fig_real(out_dir)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DengXian", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt  # noqa: E402  (Agg 后端后再 import)
    main()
