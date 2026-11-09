# -*- coding: utf-8 -*-
"""外露钢筋点密度分布模型图（Visio Figure5 页 (a)；draft 旧编号图 6(a)）。

三条曲线＝生成器参数域端点组合（src/lib/pointcloud_generator/_rebar.py
_prof：面端 START=0.80 线性升至峰位 t_p，再按 ((t-t_p)/(1-t_p))^1.2 衰减
到末端系数 f_tip）；灰虚水平线＝基线密度 f_0=0.80。

用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig_density_model.py
产出：out/figures/fig_density_model/{Fig.png,Fig.tiff}（72×52 排列一）
"""
import os

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _bootstrap import OUT_ROOT, ss

import matplotlib.pyplot as plt

OUT_DIR = os.path.join(OUT_ROOT, "fig_density_model")
os.makedirs(OUT_DIR, exist_ok=True)

START = 0.80   # REBAR_DENS_START_FACTOR（_config.py:92）
CURVES = [(0.10, 0.75), (0.22, 0.60), (0.35, 0.45)]   # (t_p, f_tip) 域端点


def profile(t, peak_t, tip_f):
    t = np.asarray(t, np.float64)
    up = START + (1.0 - START) * np.minimum(t / max(peak_t, 1e-9), 1.0)
    k = np.clip((t - peak_t) / max(1.0 - peak_t, 1e-9), 0.0, None)
    down = 1.0 + (tip_f - 1.0) * k ** 1.2
    return np.where(t <= peak_t, up, down)


def main():
    fig, ax = ss.new_figure(72, 52)
    t = np.linspace(0.0, 1.0, 401)
    ax.axhline(START, color=ss.GRAY, lw=1.2, ls=(0, (6, 3)), zorder=2)
    for (tp, tf), color in zip(CURVES, ss.COLORS):
        ax.plot(t, profile(t, tp, tf), color=color, lw=1.0, zorder=3,
                label=rf"$t_p$={tp:.2f}, $f_{{tip}}$={tf:.2f}")
    # 基线条目放图例末行（左下空白区），避免文字与穿行曲线相碰
    ax.plot([], [], color=ss.GRAY, lw=1.2, ls=(0, (6, 3)),
            label=rf"Baseline density $f_0$ = {START:.2f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.35, 1.05)
    ss.cap_axes(ax, xlim=(0.0, 1.0), ylim=(0.35, 1.05), min_ticks=5)
    ax.set_xlabel("Normalized axial coordinate $t$")
    ax.set_ylabel("Relative area density")
    ax.legend(loc="lower left", fontsize=9, frameon=False,
              handlelength=1.6, labelspacing=0.9, borderaxespad=0.4)
    png = os.path.join(OUT_DIR, "Fig.png")
    fig.savefig(png, dpi=ss.DPI, facecolor="white")
    ss.save_tiff(fig, os.path.join(OUT_DIR, "Fig.tiff"))
    plt.close(fig)
    ss.report_size_mm(png)


if __name__ == "__main__":
    main()
