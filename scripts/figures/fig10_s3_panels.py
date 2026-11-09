# -*- coding: utf-8 -*-
"""s3 柱直叠工况定性面板（Visio Figure10 页；draft 旧编号图 10）。

数据＝冻结检出缓存 out/cache/fig10_s3_deploy.npz（冻结管线现跑产物，
scene_s3_cfg01_0000 的 m0/m1/m2，含 GT meta 与检出 res）；场景俯视
(a) 读场景 npz + 分割标签缓存。

七个面板（2026-11-04 二次定稿：全图 10.5pt；(a) 143×35 用户锚定规格、
y 向压缩非等比；端面等比绘截面、x 主刻度减为 3 个防 10.5pt 挤字、
±半程由次刻度补；点云面板内部一律不加网格；黑+=真值，蓝圈=检出；
套筒端黑虚=真值内径圈、黑点=外径圈）：
  (a) 通栏 143×35  场景俯视部署分割材质  x[-2000,8000]t2000 y[-1000,2000]t1000
  (b)-(g) 画幅统一 62.4×60，内部数据框线物理尺寸一致（不等比铺满）：
  (b) m0 出筋端   y[-500,500]  z[-500,500]
  (c) m0 套筒端   同 (b)
  (d) m1 出筋端   y[-300,300]  z[-600,400]
  (e) m1 套筒端   y[-600,400]  z[-500,500]
  (f) m2 出筋端   y[-400,400]  z[-450,300]
  (g) m2 套筒端   y[-600,400]  z[-400,400]

用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig10_s3_panels.py
产出：out/figures/fig10_s3/panels/*.png（600dpi）+ Fig_preview.png（预览）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _bootstrap import (C_CONC, C_DET, C_GT, C_RB, C_ST, OUT_ROOT, ROOT,
                        ss, subsample)

import matplotlib.pyplot as plt   # noqa: E402  （须在 _bootstrap 之后）
from matplotlib.ticker import AutoMinorLocator   # noqa: E402

OUT_DIR = os.path.join(OUT_ROOT, "fig10_s3", "panels")
os.makedirs(OUT_DIR, exist_ok=True)

SCENE = "scene_s3_cfg01_0000"
FS = 10.5   # 2026-11-04 二次定稿：全图 10.5pt（(a) 143×35 锚定规格）


def _mat_colors(lbls):
    palette = {0: "#eeeeee", 1: C_CONC, 2: C_RB, 3: C_ST}
    return [palette.get(int(v), "#dddddd") for v in lbls]


def panel_overview(path):
    z = np.load(os.path.join(ROOT, "mock", "scene", "v1", "s3",
                             f"{SCENE}.npz"), allow_pickle=True)
    vis = z["visible"].astype(bool)
    pts = z["points"][vis].astype(np.float32)
    lbls = np.load(os.path.join(ROOT, "out", "cache", "scene_batch",
                                f"s3_{SCENE}_labels.npz"))["lbls"]
    fig = ss.new_figure(143, 35, single_ax=False)   # 用户定稿规格
    ml = 20.0   # 左距 20mm：10.5pt 下 "−1000" 刻度+轴标题不截断
    ax = fig.add_axes([ml / 143, ss.MARGIN_B_MM / 35,
                       1 - (ml + 6.5) / 143,   # 右距 6.5mm 防"8000"裁边
                       1 - (ss.MARGIN_B_MM + ss.MARGIN_T_MM) / 35])
    idx = subsample(len(pts), 150000)
    ax.scatter(pts[idx, 0], pts[idx, 1], c=_mat_colors(lbls[idx]),
               s=0.4, alpha=0.8, linewidths=0, rasterized=True)
    ss.no_grid(ax)   # y 向压缩非等比（Visio 版式同款），143×35 装满
    # 下缘外扩至 −1500：−1000 刻度离底边 3.2mm，与 x 首 label 错开不叠印
    ss.cap_axes(ax, xlim=(-2000, 8000), ylim=(-1500, 2000), min_ticks=5)
    ax.set_xticks(np.arange(-2000, 8001, 2000))
    ax.set_yticks(np.arange(-1000, 2001, 1000))
    ax.tick_params(labelsize=FS, axis="x", pad=6)
    ax.set_xlabel("x (mm)", fontsize=FS)
    ax.set_ylabel("y (mm)", fontsize=FS)
    fig.savefig(path, dpi=900, transparent=True)
    plt.close(fig)


def panel_endface(path, pa, end, xlim, ylim, xticks, yticks,
                  size_w, size_h):
    p0 = pa["p0"]
    meta, res, faces = pa["meta"], pa["res"], pa["faces"]
    x_band = float(faces["rebar" if end == "rb" else "sleeve"]["c3"][0])
    band = np.abs(p0[:, 0] - x_band) < 60.0
    bpts = p0[band][:, 1:]
    idx = subsample(len(bpts), 15000)
    ml = 17.0   # 左距 17mm：10.5pt 下 "−500" 刻度+轴标题不截断
    mr = 6.0
    mb = 14.0   # 底距 14mm：x pad 9 + 刻度标签 + 轴标题不被裁（(d) 实锤）
    inner_w = size_w - ml - mr
    inner_h = size_h - mb - ss.MARGIN_T_MM
    fig = ss.new_figure(size_w, size_h, single_ax=False)
    ax = fig.add_axes([ml / size_w, mb / size_h,
                       inner_w / size_w, inner_h / size_h])
    ax.scatter(bpts[idx, 0], bpts[idx, 1], c=C_CONC, s=0.8, alpha=0.6,
               linewidths=0, rasterized=True)
    if end == "rb":
        gt = np.asarray(meta["rebar_positions_yz"], np.float32)
        det = np.asarray(res["rebar"]["uv_world"],
                         np.float32).reshape(-1, 2)
        ax.plot(gt[:, 0], gt[:, 1], "+", color=C_GT, ms=7, mew=1.4,
                ls="none")
        ax.plot(det[:, 0], det[:, 1], "o", color=C_DET, ms=5.5,
                mfc="none", mew=1.5, ls="none")
    else:
        for g in meta.get("sleeve_gt") or []:
            gy, gz = g["center_yz"]
            ax.add_patch(plt.Circle((gy, gz), g.get("inner_diam", 40) / 2,
                                    fill=False, ls="--", ec=C_GT, lw=1.0))
            ax.add_patch(plt.Circle((gy, gz), g.get("outer_diam", 60) / 2,
                                    fill=False, ls=":", ec=C_GT, lw=0.7))
        det = np.asarray(res["sleeve"]["uv_world"],
                         np.float32).reshape(-1, 2)
        diam = np.asarray(res["sleeve"]["spec"], np.float32)
        for (dy, dz), dd in zip(det, diam):
            ax.add_patch(plt.Circle((dy, dz), dd / 2, fill=False,
                                    ec=C_DET, lw=1.5))
    ax.set_aspect("equal")   # 恢复等比：画幅比例=窗口比例，内容零变形
    ss.no_grid(ax)
    ss.cap_axes(ax, xlim=xlim, ylim=ylim, cap_x=False, cap_y=False)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=FS)   # x pad=9：±500 角刻度与 x label 错开
    ax.tick_params(axis="x", pad=9)
    ax.set_xlabel("y (mm)", fontsize=FS)
    ax.set_ylabel("z (mm)", fontsize=FS)
    fig.savefig(path, dpi=900, transparent=True)
    plt.close(fig)


def main():
    z = np.load(os.path.join(ROOT, "out", "cache",
                             "fig10_s3_deploy.npz"), allow_pickle=True)
    panels = {int(pa["mid"]): pa for pa in z["panels"]}
    print("cache members:", sorted(panels))

    p_a = os.path.join(OUT_DIR, "a_overview.png")
    panel_overview(p_a)

    specs = [  # (文件名, mid, end, xlim, ylim, xticks, yticks, 宽mm, 高mm)
        # 2026-11-04 四次定稿：内容零变形（画幅比例=窗口比例），六面板
        # 共用比例尺 0.0394mm/mm；x 主刻度 3 个防 10.5pt 挤字，y/z 全套
        # (b)(c)(e) 窗口 1000×1000 → 62.4×55.9
        ("b_m0_rb.png", 0, "rb", (-500, 500), (-500, 500),
         (-500, 0, 500), (-500, -250, 0, 250, 500), 62.4, 55.9),
        ("c_m0_sl.png", 0, "sl", (-500, 500), (-500, 500),
         (-500, 0, 500), (-500, -250, 0, 250, 500), 62.4, 55.9),
        # (d) 窗口 600×1000
        ("d_m1_rb.png", 1, "rb", (-300, 300), (-600, 400),
         (-300, 0, 300), (-600, -400, -200, 0, 200, 400), 46.6, 55.9),
        ("e_m1_sl.png", 1, "sl", (-600, 400), (-500, 500),
         (-600, -200, 200), (-500, -250, 0, 250, 500), 62.4, 55.9),
        # (f) 窗口 800×750
        ("f_m2_rb.png", 2, "rb", (-400, 400), (-450, 300),
         (-400, 0, 400), (-450, -300, -150, 0, 150, 300), 54.5, 46.1),
        # (g) 与 (f) 同框（54.5×46.1）；y 窗口对称外扩至 [-535,335] 使
        # 窗口比例=框线比例（1.075 vs 1.064），圆圈保持正圆不拉伸
        ("g_m2_sl.png", 2, "sl", (-535, 335), (-400, 400),
         (-500, -100, 300), (-400, -200, 0, 200, 400), 65.4, 55.3),
    ]
    paths = [p_a]
    for fn, mid, end, xlim, ylim, xt, yt, w, h in specs:
        p = os.path.join(OUT_DIR, fn)
        panel_endface(p, panels[mid], end, xlim, ylim, xt, yt, w, h)
        paths.append(p)

    out = os.path.join(OUT_ROOT, "fig10_s3", "Fig_preview.png")
    ss.stitch_rows([[p_a],
                    [paths[1], paths[2], paths[3]],
                    [paths[4], paths[5], paths[6]]], out)
    print("panels:", *[os.path.basename(p) for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
