# -*- coding: utf-8 -*-
"""Fig11 出版级重绘（人在回路迭代版，离线读冻结检出缓存，无 GPU）。

底版场景：s4/scene_s4_cfg01_0000（三场景评选最干净，见 select_scene.py）；
检出缓存：out/cache/fig11_s4_deploy.npz（冻结管线现跑产物，含箍筋面认证）。

版式（先子图后拼接；拼接由用户完成，stitch_rows 仅出内部预览）：
  (a) 通栏 场景俯视部署分割材质 + 构件滚转标注    144x36
  (b)(c) 构件① 出筋端 / 套筒端 检出对真值          72x52
  (d)(e) 构件② 出筋端 / 套筒端 检出对真值          72x52
  (f) 通栏 构件② 箍筋认证侧面视图：腿位检出对真值  144x31

图内语言：全英文（2026-11-04 用户定稿）。
图内不打 (a)(b)(c) 编号、不标统计小字（用户定稿：编号与说明在 Visio
拼接时由用户添加）；GT 等CS黑话不得出现在论文图内（交接文档红线）。

用法（项目根）：.venv/Scripts/python.exe -I temp/fig11/plot_fig11.py
产出：temp/fig11/panels_en/*.png（600dpi 交付子图）
     + temp/fig11/Fig11_en_preview.png（内部碰撞预览，不交付）
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL = os.path.join(os.environ.get("USERPROFILE", ""), ".zcode",
                     "skills", "scientific-plotting")
sys.path.insert(0, os.path.join(SKILL, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sci_style as ss

ss.apply_style()   # 禁手写 rcParams，一律走样式模块

import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(ROOT, "out", "figures")
OUT_DIR = os.path.join(OUT_ROOT, "fig11_s4_panels")
PANEL_DIR = os.path.join(OUT_ROOT, "fig11_s4_panels", "panels_en")
os.makedirs(PANEL_DIR, exist_ok=True)

# 项目定性图既定配色（Fig7/8/10/12 同款，真实场景配色优先规则）
C_CONC = "#c8c8c8"   # 混凝土
C_RB = "#d62728"     # 钢筋（材质染色）
C_ST = "#ff9f1c"     # 箍筋（材质染色）
C_GT = "#222222"     # 真值标记（黑）
C_DET = "#1f77b4"    # 检出标记（蓝）

SCENE = "scene_s4_cfg01_0000"


def _subsample(n, cap, seed=0):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, cap, replace=False)


def _mat_colors(lbls):
    palette = {0: "#eeeeee", 1: C_CONC, 2: C_RB, 3: C_ST}
    return [palette.get(int(v), "#dddddd") for v in lbls]


def panel_overview(path):
    """(a) 场景俯视：部署分割材质 + 滚转标注。144×36。"""
    z = np.load(os.path.join(ROOT, "mock", "scene", "v1", "s4",
                             f"{SCENE}.npz"), allow_pickle=True)
    vis = z["visible"].astype(bool)
    pts = z["points"][vis].astype(np.float32)
    member_id = z["member_id"][vis].astype(np.int64)
    lbls = np.load(os.path.join(ROOT, "out", "cache", "scene_batch",
                                f"s4_{SCENE}_labels.npz"))["lbls"]
    panels = np.load(os.path.join(ROOT, "out", "cache",
                                  "fig11_s4_deploy.npz"),
                     allow_pickle=True)["panels"]

    fig = ss.new_figure(144, 36, single_ax=False)
    ax = fig.add_axes([ss.MARGIN_L_MM / 144, ss.MARGIN_B_MM / 36,
                       1 - (ss.MARGIN_L_MM + ss.MARGIN_R_MM) / 144,
                       1 - (ss.MARGIN_B_MM + ss.MARGIN_T_MM) / 36])
    idx = _subsample(len(pts), 150000)
    ax.scatter(pts[idx, 0], pts[idx, 1], c=_mat_colors(lbls[idx]),
               s=0.4, alpha=0.8, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ss.no_grid(ax)

    # 构件外接框 + 滚转标注（GT member_id 分组；文字错高避让碰撞）
    offsets = [(-2400, 950), (1500, -850)]
    for pa, name, (dx, dy) in zip(panels, ["Member 1", "Member 2"], offsets):
        mid = int(pa["mid"])
        mp = pts[member_id == mid]
        x0, x1 = np.percentile(mp[:, 0], [0.5, 99.5])
        y0, y1 = np.percentile(mp[:, 1], [0.5, 99.5])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        roll = abs(float(pa["roll_deg"]))
        ax.annotate(f"{name} (rolled {roll:.0f}\N{DEGREE SIGN}, on side)",
                    xy=(cx, cy), xytext=(cx + dx, cy + dy), ha="center",
                    fontsize=8.5, color="#333333",
                    arrowprops=dict(arrowstyle="-", lw=0.7,
                                    color="#333333"))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ss.cap_axes(ax, xlim=(-500, 10500), ylim=(-400, 1900), min_ticks=5)
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def _endface_window(band_yz, gt, pad=20.0):
    """端面窗口：带点范围外包 GT，向外圆整到 50 mm（落在网格上）。"""
    lo = np.minimum(np.percentile(band_yz, 2, axis=0),
                    gt.min(axis=0)) - pad
    hi = np.maximum(np.percentile(band_yz, 98, axis=0),
                    gt.max(axis=0)) + pad
    return np.floor(lo / 50) * 50, np.ceil(hi / 50) * 50


def panel_endface(path, pa, end):
    """(b)-(e) 端面检出对真值。end='rb'|'sl'。72×52。"""
    p0 = pa["p0"]
    meta, res, faces = pa["meta"], pa["res"], pa["faces"]
    x_band = float(faces["rebar" if end == "rb" else "sleeve"]["c3"][0])
    band = np.abs(p0[:, 0] - x_band) < 60.0
    bpts = p0[band][:, 1:]
    idx = _subsample(len(bpts), 15000)

    fig, ax = ss.new_figure(72, 52)
    ax.scatter(bpts[idx, 0], bpts[idx, 1], c=C_CONC, s=0.8, alpha=0.6,
               linewidths=0, rasterized=True)
    if end == "rb":
        gt = np.asarray(meta["rebar_positions_yz"], np.float32)
        det = np.asarray(res["rebar"]["uv_world"],
                         np.float32).reshape(-1, 2)
        ax.plot(gt[:, 0], gt[:, 1], "+", color=C_GT, ms=7, mew=1.4,
                ls="none")
        ax.plot(det[:, 0], det[:, 1], "o", color=C_DET, ms=4.5,
                mfc="none", mew=1.3, ls="none")
        n_gt, n_det = len(gt), len(det)
        hit = int((np.linalg.norm(det[:, None] - gt[None], axis=-1)
                   .min(1) <= 30).sum())
    else:
        sls = meta.get("sleeve_gt") or []
        det = np.asarray(res["sleeve"]["uv_world"],
                         np.float32).reshape(-1, 2)
        diam = np.asarray(res["sleeve"]["spec"], np.float32)
        for g in sls:
            gy, gz = g["center_yz"]
            ax.add_patch(plt.Circle((gy, gz), g.get("inner_diam", 40) / 2,
                                    fill=False, ls="--", ec=C_GT, lw=1.0))
            ax.add_patch(plt.Circle((gy, gz), g.get("outer_diam", 60) / 2,
                                    fill=False, ls=":", ec=C_GT, lw=0.7))
        for (dy, dz), dd in zip(det, diam):
            ax.add_patch(plt.Circle((dy, dz), dd / 2, fill=False,
                                    ec=C_DET, lw=1.4))
        gt = np.array([g["center_yz"] for g in sls]).reshape(-1, 2)
        n_gt, n_det = len(gt), len(det)
        hit = int((np.linalg.norm(det[:, None] - gt[None], axis=-1)
                   .min(1) <= 30).sum()) if n_gt else 0

    # 四面板统一固定窗口（截面外包，构件①②同尺度可横向对比）
    ax.set_aspect("equal")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=(-200, 400), ylim=(-400, 400), min_ticks=4)
    ax.set_xticks([-200, 0, 200, 400])   # 显式 200 步长，防刻度标签挤压
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def panel_stirrup(path, pa):
    """(f) 构件②箍筋：认证侧面视图腿位检出对真值。144×31。"""
    meta, res = pa["meta"], pa["res"]
    st = res["stirrup"]
    x0, yc = float(st["x0"]), float(st["yc"])
    tv = np.asarray(st["tv"], np.float64)
    ev2 = st.get("ev2") or (1.0, 0.0)
    dets = np.asarray(st["dets"], np.float64).reshape(-1, 3)

    fig = ss.new_figure(144, 31, single_ax=False)
    ax = fig.add_axes([ss.MARGIN_L_MM / 144, ss.MARGIN_B_MM / 31,
                       1 - (ss.MARGIN_L_MM + ss.MARGIN_R_MM) / 144,
                       1 - (ss.MARGIN_B_MM + ss.MARGIN_T_MM) / 31])
    idx = _subsample(len(tv), 40000)
    ax.scatter(tv[idx, 0], tv[idx, 1], c=C_CONC, s=0.4,
               alpha=0.5, linewidths=0, rasterized=True)
    # GT 腿脚（质心系 → 视图系，口径同 eval_one）
    for g in meta.get("hairpin_gt") or []:
        for p in (g["p_foot_l"], g["p_foot_r"]):
            v = float(p[1]) * ev2[0] + float(p[2]) * ev2[1] - yc
            ax.plot(float(p[0]) - x0, v, "+", color=C_GT, ms=6, mew=1.3,
                    ls="none")
    # 检出腿已是视图系坐标（x−x0, v−yc），不再减原点
    ax.plot(dets[:, 0], dets[:, 1], "o", color=C_DET, ms=4, mfc="none",
            mew=1.2, ls="none")
    ax.set_aspect("equal")
    ax.set_xlabel("Along-member x (mm)")
    ax.set_ylabel("v (mm)")
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=(0, 4000), ylim=(-300, 300), min_ticks=5)
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def main():
    panels = np.load(os.path.join(ROOT, "out", "cache",
                                  "fig11_s4_deploy.npz"),
                     allow_pickle=True)["panels"]
    m0, m1 = panels[0], panels[1]

    p_a = os.path.join(PANEL_DIR, "a_overview.png")
    panel_overview(p_a)
    paths = {}
    for i, (pa, name) in enumerate([(m0, "1"), (m1, "2")]):
        for end in [("rb", "出筋端"), ("sl", "套筒端")]:
            p = os.path.join(PANEL_DIR,
                             f"{'bcde'[i * 2 + (0 if end[0] == 'rb' else 1)]}"
                             f"_m{name}_{end[0]}.png")
            panel_endface(p, pa, end[0])
            paths[(name, end[0])] = p
    p_f = os.path.join(PANEL_DIR, "f_m2_stirrup.png")
    panel_stirrup(p_f, m1)

    rows = [[p_a],
            [paths[("1", "rb")], paths[("1", "sl")]],
            [paths[("2", "rb")], paths[("2", "sl")]],
            [p_f]]
    out = os.path.join(OUT_ROOT, "fig11_s4_panels", "Fig_preview.png")
    ss.stitch_rows(rows, out)   # 仅内部碰撞检查预览；拼接由用户完成
    ss.report_size_mm(out)
    print("panels:", *[os.path.basename(p) for p in
                       [p_a, paths[("1", "rb")], paths[("1", "sl")],
                        paths[("2", "rb")], paths[("2", "sl")], p_f]],
          sep="\n  ")


if __name__ == "__main__":
    main()
