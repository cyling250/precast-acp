# -*- coding: utf-8 -*-
"""SfM 稀疏点云三视图（原始 RGB 着色，只要点云）。

输入：out/sfm/prefab_scan/sfm_points.npz（scripts/sfm_prefab_scan.py 产出）。
产出：out/sfm/prefab_scan/views/
  panel_a_top.png / panel_b_front.png / panel_c_side.png（各 48×34.6mm @600dpi）
  sfm_views.png（排列二一行三张，14.4×3.46cm）。
z 向重力：俯视 x-y、正视 x-z、侧视 y-z。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
from _bootstrap import *   # noqa: F401  路径锚定 + sci_style 装载
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

WS = os.path.join(OUT_ROOT, "..", "sfm", "prefab_scan")
OUT = os.path.join(WS, "views")
os.makedirs(OUT, exist_ok=True)
data = np.load(os.path.join(WS, "sfm_points.npz"))
xyz, rgb = data["xyz"].astype(np.float64), data["rgb"].astype(np.float64) / 255.0
print(f"points={len(xyz)}")

# 稳健边界：1–99% 分位裁离群点（SfM 稀疏云的漂移点不少见）
lo = np.percentile(xyz, 1.0, axis=0)
hi = np.percentile(xyz, 99.0, axis=0)
keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
xyz, rgb = xyz[keep], rgb[keep]
print(f"after clip: {len(xyz)} (dropped {len(data['xyz']) - len(xyz)})")

W_CELL, H_CELL, DPI = 48.0, 34.6, 600


def new_panel():
    fig = plt.figure(figsize=(W_CELL / 25.4, H_CELL / 25.4))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.90])
    return fig, ax


def save_panel(fig, ax, name):
    ax.set_xticks([])
    ax.set_yticks([])
    ss.no_grid(ax)
    for sp in ax.spines.values():
        sp.set_visible(False)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


views = [("panel_a_top.png", 0, 1),      # 俯视 x-y
         ("panel_b_front.png", 0, 2),    # 正视 x-z
         ("panel_c_side.png", 1, 2)]     # 侧视 y-z
panels = []
for name, ci, cj in views:
    fig, ax = new_panel()
    ax.scatter(xyz[:, ci], xyz[:, cj], s=0.3, c=rgb, marker="o",
               linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xyz[:, ci].min(), xyz[:, ci].max())
    ax.set_ylim(xyz[:, cj].min(), xyz[:, cj].max())
    panels.append(save_panel(fig, ax, name))

stitched = ss.stitch_panels(panels, os.path.join(OUT, "sfm_views.png"), cols=3)
ss.report_size_mm(stitched)
for p in panels:
    im = np.asarray(Image.open(p).convert("L"))
    print(f"{os.path.basename(p)}: {im.shape[1]}x{im.shape[0]}px nonwhite={int((im < 245).sum())}")
print("stitched ->", stitched)
