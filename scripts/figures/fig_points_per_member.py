# -*- coding: utf-8 -*-
"""各类型每件点数图（Visio Figure6 页 (b)；draft 旧编号图 6(b)）。

数据＝mock/inuse/test 64 件，按 meta_json.type 分 8 类（每类 8 件）
取点数均值 ± 标准差。类型缩写同 draft 图 4 题注：RC＝矩形柱，LC＝
L 形柱，RB＝矩形梁，TB＝T 形梁，RCB/TCB/LCB＝矩形/T 形/L 形叠合梁，
SWP＝剪力墙板。

用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig_points_per_member.py
产出：out/figures/fig_points_per_member/{Fig.png,Fig.tiff}（72×52 排列一）
"""
import glob
import json
import os

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _bootstrap import OUT_ROOT, ROOT, ss

import matplotlib.pyplot as plt   # noqa: E402  （须在 _bootstrap 之后）

OUT_DIR = os.path.join(OUT_ROOT, "fig_points_per_member")
os.makedirs(OUT_DIR, exist_ok=True)

TYPE_NAMES = [
    ("rectangular_column", "RC"),
    ("l_shaped_column", "LC"),
    ("rectangular_beam", "RB"),
    ("t_beam", "TB"),
    ("rectangular_composite_beam", "RCB"),
    ("t_composite_beam", "TCB"),
    ("l_composite_beam", "LCB"),
    ("shear_wall_panel", "SWP"),
]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "mock", "inuse",
                                          "test", "*.npz")))
    by_type = {t: [] for t, _ in TYPE_NAMES}
    for f in files:
        z = np.load(f)
        m = json.loads(str(z["meta_json"]))
        by_type[m["type"]].append(int(len(z["points"])))
    means = np.array([np.mean(by_type[t]) for t, _ in TYPE_NAMES])
    stds = np.array([np.std(by_type[t]) for t, _ in TYPE_NAMES])
    for (t, ab), m, s in zip(TYPE_NAMES, means, stds):
        print(f"{ab:>4} {t:<28} n={len(by_type[t])} {m/1e6:.2f}±{s/1e6:.2f}M")

    fig, ax = ss.new_figure(72, 52)
    x = np.arange(8)
    ax.bar(x, means / 1e6, width=0.62, color=ss.GRAY, edgecolor="none",
           zorder=3)
    ax.errorbar(x, means / 1e6, yerr=stds / 1e6, fmt="none",
                ecolor="black", elinewidth=1.0, capsize=3, capthick=1.0,
                zorder=4)
    ax.set_xticks(x, [ab for _, ab in TYPE_NAMES])
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 4.6)
    ss.cap_axes(ax, ylim=(0, 4.6), cap_x=False, min_ticks=5)
    ax.set_xlabel("Member type")
    ax.set_ylabel("Points per member (million)")
    fig.savefig(os.path.join(OUT_DIR, "Fig.png"), dpi=ss.DPI,
                facecolor="white")
    ss.save_tiff(fig, os.path.join(OUT_DIR, "Fig.tiff"))
    plt.close(fig)
    ss.report_size_mm(os.path.join(OUT_DIR, "Fig.png"))


if __name__ == "__main__":
    main()
