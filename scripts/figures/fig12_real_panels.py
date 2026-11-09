# -*- coding: utf-8 -*-
"""Fig12 真实场景识别效果——出版级子图（离线读冻结检出缓存，无 GPU）。

检出缓存：out/cache/fig12_real_deploy.npz（freeze_real.py 现跑冻结，
与 results/real_fullchain.json canonical 逐位一致：钢筋 12/12、套筒 9/8）。

版式（三面板，用户 2026-11-04 定稿；编号与说明由用户 Visio 拼接时添加）：
  侧视通栏  全场景扫描 + 真值筋点 + 检出杆段          144x36
  钢筋端面  检出蓝圈 vs 人工标注真值黑+（12/12）       72x52
  套筒端面  检出蓝圈(9) vs 人工标注环黑虚(12)          72x52

图内全英文、不打编号、不标统计小字；GT 等CS黑话不得出现（红线）。

用法（项目根）：.venv/Scripts/python.exe -I temp/fig12/plot_fig12.py
产出：temp/fig12/panels_en/*.png（600dpi 交付子图）
     + temp/fig12/Fig12_en_preview.png（内部碰撞预览，不交付）
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL = os.path.join(os.environ.get("USERPROFILE", ""), ".zcode",
                     "skills", "scientific-plotting")
sys.path.insert(0, os.path.join(SKILL, "scripts"))

import sci_style as ss

ss.apply_style()   # 禁手写 rcParams，一律走样式模块

import matplotlib.pyplot as plt

OUT_ROOT = os.path.join(ROOT, "out", "figures")
OUT_DIR = os.path.join(OUT_ROOT, "fig12_panels")
PANEL_DIR = os.path.join(OUT_ROOT, "fig12_panels", "panels_en")
os.makedirs(PANEL_DIR, exist_ok=True)

# 项目定性图既定配色（Fig7/8/10/11/12 同款）
C_CONC = "#c8c8c8"   # 混凝土/扫描云
C_RB = "#d62728"     # 真值筋点（红，沿用旧图 12 口径）
C_GT = "#222222"     # 真值标记（黑）
C_DET = "#1f77b4"    # 检出标记（蓝）


def _subsample(n, cap, seed=0):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, cap, replace=False)


def panel_side(path, z):
    """侧视通栏：扫描 + 真值筋点 + 检出杆段。144×36。"""
    pts, gts, seg = z["pts"], z["gts"], z["seg_w"]
    fig = ss.new_figure(144, 36, single_ax=False)
    ax = fig.add_axes([ss.MARGIN_L_MM / 144, ss.MARGIN_B_MM / 36,
                       1 - (ss.MARGIN_L_MM + ss.MARGIN_R_MM) / 144,
                       1 - (ss.MARGIN_B_MM + ss.MARGIN_T_MM) / 36])
    idx = _subsample(len(pts), 150000)
    ax.scatter(pts[idx, 0], pts[idx, 2], c=C_CONC, s=0.4, alpha=0.5,
               linewidths=0, rasterized=True)
    idx2 = _subsample(len(gts), 25000)
    ax.scatter(gts[idx2, 0], gts[idx2, 2], c=C_RB, s=0.5, alpha=0.8,
               linewidths=0, rasterized=True)
    for (p0, p1) in seg:   # 检出杆段：端面 → 尖端（世界系）
        ax.plot([p0[0], p1[0]], [p0[2], p1[2]], color=C_DET, lw=1.6,
                solid_capstyle="round")
    ax.set_aspect("equal")
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=(-1000, 3000), ylim=(0, 750), min_ticks=5)
    ax.set_xticks(np.arange(-1000, 3001, 500))
    ax.set_yticks((0, 250, 500, 750))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def panel_rb_face(path, z):
    """钢筋端面：检出蓝圈 vs 人工标注真值黑+。72×52。"""
    band, det, gt = z["band_rb"], z["det_rb"], z["gt_rb_uv"]
    fig, ax = ss.new_figure(72, 52)
    idx = _subsample(len(band), 15000)
    ax.scatter(band[idx, 0], band[idx, 1], c=C_CONC, s=0.8, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.plot(gt[:, 0], gt[:, 1], "+", color=C_GT, ms=7, mew=1.4, ls="none")
    ax.plot(det[:, 0], det[:, 1], "o", color=C_DET, ms=4.5, mfc="none",
            mew=1.3, ls="none")
    ax.set_aspect("equal")
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=(-300, 500), ylim=(-400, 400), min_ticks=5)
    ax.set_xticks((-300, -100, 100, 300, 500))
    ax.set_yticks((-400, -200, 0, 200, 400))
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def panel_sl_face(path, z):
    """套筒端面：检出蓝圈 vs 人工标注环黑虚（变换到检出系）。72×52。"""
    band, det, r_in = z["band_sl"], z["det_sl"], z["r_in"]
    sl_R, sl_t = z["sl_R"], z["sl_t"]
    # 人工标注环心（手标系 → 检出系）：det@R.T+t ≈ sc ⇒ sc_in_det=(sc−t)@R
    with open(os.path.join(ROOT, "real_resource",
                           "sleeve_gt_manual.json"), encoding="utf-8") as f:
        sj = json.load(f)
    sc = np.array([[q["h2"], q["h3"]] for q in sj["circles"]], np.float64)
    r_gt = np.array([float(q["r_mm"]) for q in sj["circles"]])
    sc_det = (sc - sl_t) @ sl_R
    fig, ax = ss.new_figure(72, 52)
    idx = _subsample(len(band), 15000)
    ax.scatter(band[idx, 0], band[idx, 1], c=C_CONC, s=0.8, alpha=0.6,
               linewidths=0, rasterized=True)
    for (gy, gz), gr in zip(sc_det, r_gt):
        ax.add_patch(plt.Circle((gy, gz), gr, fill=False, ls="--",
                                ec=C_GT, lw=1.0))
    for (dy, dz), rr in zip(det, r_in):
        ax.add_patch(plt.Circle((dy, dz), rr / 2, fill=False, ec=C_DET,
                                lw=1.4))
    ax.set_aspect("equal")
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=(-400, 400), ylim=(-400, 400), min_ticks=5)
    ax.set_xticks((-400, -200, 0, 200, 400))
    ax.set_yticks((-400, -200, 0, 200, 400))
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    fig.savefig(path, dpi=ss.DPI, facecolor="white")
    plt.close(fig)


def main():
    z = np.load(os.path.join(ROOT, "out", "cache",
                             "fig12_real_deploy.npz"))
    p_a = os.path.join(PANEL_DIR, "a_side.png")
    panel_side(p_a, z)
    p_b = os.path.join(PANEL_DIR, "b_rb_face.png")
    panel_rb_face(p_b, z)
    p_c = os.path.join(PANEL_DIR, "c_sl_face.png")
    panel_sl_face(p_c, z)
    out = os.path.join(OUT_ROOT, "fig12_panels", "Fig_preview.png")
    ss.stitch_rows([[p_a], [p_b, p_c]], out)   # 内部碰撞预览，不交付
    ss.report_size_mm(out)
    print("panels:", *[os.path.basename(p) for p in (p_a, p_b, p_c)],
          sep="\n  ")


if __name__ == "__main__":
    main()
