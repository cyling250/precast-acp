# -*- coding: utf-8 -*-
"""Fig7 素材：真实堆场点云三视图 + 侧方俯视图（原始 RGB 着色，只要点云）。

输入：real_resource/scene_full_mm - Cloud.ply（ASCII，1,728,844 点，
x/y/z + RGB + 法线，毫米坐标，z 向重力）。
产出：out/figures/fig07_scene_views/
  panel_a_top.png / panel_b_front.png / panel_c_side.png / panel_d_oblique.png
  （各 72×52mm @600dpi，排列一单格）+ fig07_scene_views.png（2×2 拼接 14.4×10.4cm）。

要点：按 style-display——点尺寸小、无网格、边界紧凑、多面板必须 (a)(b)(c)(d)
编号；RGB 用点云原始颜色（3D 视图 depthshade=False 不做深度压暗）。
"""
from _bootstrap import *   # noqa: F401  路径锚定 + sci_style 装载
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PLY = os.path.join(ROOT, "real_resource", "scene_full_mm - Cloud.ply")
OUT = os.path.join(OUT_ROOT, "fig07_scene_views")
os.makedirs(OUT, exist_ok=True)

# ---- 读 PLY：解析头部行数与点数，pandas 秒级读入 ----
n_pts, hdr_lines = None, 0
with open(PLY, "r", errors="ignore") as f:
    for line in f:
        hdr_lines += 1
        if line.startswith("element vertex"):
            n_pts = int(line.split()[2])
        if line.startswith("end_header"):
            break
assert n_pts is not None, "PLY 头部无 element vertex"
df = pd.read_csv(PLY, sep=r"\s+", header=None, skiprows=hdr_lines,
                 nrows=n_pts, usecols=(0, 1, 2, 3, 4, 5),
                 names=["x", "y", "z", "r", "g", "b"])
xyz = df[["x", "y", "z"]].to_numpy()
rgb = df[["r", "g", "b"]].to_numpy(dtype=float) / 255.0
print(f"points={len(xyz)}")

# ---- 稳健边界：0.5–99.5% 分位裁掉离群点（噪声点会让画幅失控）----
lo = np.percentile(xyz, 0.5, axis=0)
hi = np.percentile(xyz, 99.5, axis=0)
keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
xyz, rgb = xyz[keep], rgb[keep]
print(f"after clip: {len(xyz)} (dropped {n_pts - len(xyz)})")
rng = np.random.default_rng(0)

# ---- 展示图面板渲染（单格 72×52mm @600dpi）----
W_CELL, H_CELL, DPI = 72.0, 52.0, 600


def new_panel():
    fig = plt.figure(figsize=(W_CELL / 25.4, H_CELL / 25.4))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.90])
    return fig, ax


def frame(fig, ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def save_panel(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def scatter2d(ax, px, py):
    ax.scatter(px, py, s=0.5, c=rgb, marker="o", linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(px.min(), px.max())
    ax.set_ylim(py.min(), py.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ss.no_grid(ax)
    frame(fig=ax.figure, ax=ax)


panels = {}

# (a) 俯视 x-y
fig, ax = new_panel()
scatter2d(ax, xyz[:, 0], xyz[:, 1])
ss.label_panels([ax], ["(a)"])
panels["a"] = save_panel(fig, "panel_a_top.png")

# (b) 正视 x-z
fig, ax = new_panel()
scatter2d(ax, xyz[:, 0], xyz[:, 2])
ss.label_panels([ax], ["(b)"])
panels["b"] = save_panel(fig, "panel_b_front.png")

# (c) 侧视 y-z
fig, ax = new_panel()
scatter2d(ax, xyz[:, 1], xyz[:, 2])
ss.label_panels([ax], ["(c)"])
panels["c"] = save_panel(fig, "panel_c_side.png")

# (d) 侧方俯视（3D 轴测，elev=30 azim=-60；下采样到 40 万点加速）
idx = rng.choice(len(xyz), size=min(400_000, len(xyz)), replace=False)
sub = xyz[idx]
sub_rgb = rgb[idx]
fig = plt.figure(figsize=(W_CELL / 25.4, H_CELL / 25.4))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.92], projection="3d")
ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.4, c=sub_rgb,
           depthshade=False, linewidths=0)
r = hi - lo
ax.set_box_aspect((r[0], r[1], r[2]))
ax.set_xlim(lo[0], hi[0])
ax.set_ylim(lo[1], hi[1])
ax.set_zlim(lo[2], hi[2])
ax.view_init(elev=30, azim=-60)
ax.set_axis_off()
fig.add_artist(Rectangle((0.02, 0.02), 0.96, 0.92, fill=False,
                         lw=0.6, transform=fig.transFigure,
                         edgecolor="black"))
bb = ax.get_position()
fig.text(bb.x0 - 0.01, bb.y1, "(d)", ha="left", va="top", fontsize=10.5)
panels["d"] = save_panel(fig, "panel_d_oblique.png")

# ---- 拼接 2×2（排列一 14.4×10.4cm）+ 程序化自检 ----
stitched = ss.stitch_panels(
    [panels["a"], panels["b"], panels["c"], panels["d"]],
    os.path.join(OUT, "fig07_scene_views.png"), cols=2)
ss.report_size_mm(stitched)

from PIL import Image  # noqa: E402
for k, p in panels.items():
    im = np.asarray(Image.open(p).convert("L"))
    nonwhite = int((im < 245).sum())
    print(f"panel {k}: {im.shape[1]}x{im.shape[0]}px nonwhite={nonwhite}")
print("stitched ->", stitched)
