"""
批量数据集生成器 v2。

规模（方案评估 §四）：8 类型 × 12 构型 × 2 密度 × 2 噪声 = 384 件，
按 (type, vidx) 组 6:2:2 拆分（同一构型不跨 split，防泄漏）。

输出结构：
mock/
  origin/   ← 干净版（无测量噪声，但含贴地缺失/半弧钢筋等可见性几何）
  inuse/{train,val,test}/ ← 加噪版
  qa/       ← 每构型一张标签着色渲染图

NPZ 内容：points(float32 N×3), labels(int32), meta_json(str)。
meta_json 含拟合层验证所需全部 GT（尺寸/筋位/套筒位/箍筋位）。
"""

import os, json
import numpy as np
from typing import Optional

from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn,
)
from rich.console import Console

from ._config import (
    DENSITY_MAP, NOISE_LEVELS,
    NUM_VARIANTS_PER_TYPE, TRAIN_VAL_TEST_SPLIT,
    MOCK_DIR, ORIGIN_DIR, QA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
)
from ._builder import build_member
from ._noise import apply_noise, apply_density_variation

_CONSOLE = Console()

# 七种构件类型（全部布局 C）
COMP_TYPES = [
    "rectangular_column",
    "l_shaped_column",
    "rectangular_beam",
    "t_beam",
    "rectangular_composite_beam",
    "t_composite_beam",
    "l_composite_beam",
    "shear_wall_panel",   # v2.3（2026-10-24 用户要求：板+窗洞+洞口周围钢筋）
]

LABEL_COLORS = {
    0: (0.62, 0.62, 0.62),   # 非预制构件点：灰
    1: (0.78, 0.72, 0.60),   # 混凝土：米灰
    2: (0.10, 0.25, 0.75),   # 纵筋：蓝
    3: (0.85, 0.15, 0.10),   # 箍筋：红
}


def render_qa(
    points: np.ndarray,
    labels: np.ndarray,
    fname: str,
    out_dir: str = QA_DIR,
    title: str = "",
    max_points: int = 250_000,
):
    """三视图标签着色渲染（QA 人工核对用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, labels = points[idx], labels[idx]
    colors = np.zeros((len(points), 3))
    for lb, c in LABEL_COLORS.items():
        colors[labels == lb] = c

    views = [
        ("XY top", 0, 1),
        ("XZ side", 0, 2),
        ("YZ end", 1, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=100)
    for ax, (name, i, j) in zip(axes, views):
        ax.scatter(points[:, i], points[:, j], s=0.15, c=colors, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("xyz"[i])
        ax.set_ylabel("xyz"[j])
    fig.suptitle(title or fname)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


def _generate_one(args: tuple) -> list[dict]:
    """
    生成一个构型（clean + 各噪声级）。
    args: (comp_type, vidx, dens_name, dens_val, noise_names, seed)
    """
    comp_type, vidx, dens_name, dens_val, noise_names, seed = args
    rng = np.random.default_rng(seed)
    result = build_member(comp_type, density=dens_val, rng=rng)
    pts, lbls, meta = result["points"], result["labels"], result["metadata"]

    # 非均匀密度（模拟扫描距离变化）——clean 版即含
    pts, lbls = apply_density_variation(pts, lbls, rng)

    meta["density"] = dens_name
    meta["seed"] = int(seed)

    outputs = [{
        "comp_type": comp_type, "vidx": vidx, "density": dens_name,
        "noise_level": "clean", "points": pts, "labels": lbls, "metadata": meta,
    }]
    for noise_name in noise_names:
        rng_noise = np.random.default_rng(seed + 7919)
        cfg = NOISE_LEVELS[noise_name]
        pts_n, lbls_n = apply_noise(pts, lbls, cfg, rng_noise)
        meta_n = dict(meta)
        meta_n["noise_level"] = noise_name
        outputs.append({
            "comp_type": comp_type, "vidx": vidx, "density": dens_name,
            "noise_level": noise_name, "points": pts_n, "labels": lbls_n,
            "metadata": meta_n,
        })
    return outputs


def _save_npz(output: dict, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    fname = (
        f"Mock_{output['comp_type']}_{output['vidx']:03d}"
        f"_{output['density']}_{output['noise_level']}.npz"
    )
    fpath = os.path.join(save_dir, fname)
    np.savez_compressed(
        fpath,
        points=output["points"].astype(np.float32),
        labels=output["labels"].astype(np.int32),
        meta_json=json.dumps(output["metadata"], ensure_ascii=False),
    )
    return fpath


def test_keys(test_dir: str = None) -> set:
    """从 mock/inuse/test 文件名解析 (comp_type, vidx) 集合（与训练划分严格一致）。"""
    if test_dir is None:
        test_dir = TEST_DIR
    keys = set()
    for fname in os.listdir(test_dir):
        if not fname.endswith(".npz"):
            continue
        p = os.path.splitext(fname)[0].split("_")
        # Mock_{comp(可含下划线)}_{vidx:03d}_{dens}_{noise}
        try:
            vidx = int(p[-3])
        except (IndexError, ValueError):
            continue
        keys.add(("_".join(p[1:-3]), vidx))
    return keys


class DatasetGenerator:
    """v2 批量数据集生成器（单进程——Windows spawn 下 open3d/shapely 序列化成本高）。"""

    def __init__(
        self,
        origin_dir: str = ORIGIN_DIR,
        train_dir: str = TRAIN_DIR,
        val_dir: str = VAL_DIR,
        test_dir: str = TEST_DIR,
        num_variants: int = NUM_VARIANTS_PER_TYPE,
        density_levels: Optional[list[str]] = None,
        noise_levels: Optional[list[str]] = None,
        split_ratio: tuple[int, int, int] = TRAIN_VAL_TEST_SPLIT,
        render_qa: bool = True,
    ):
        self.origin_dir = origin_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_variants = num_variants
        self.density_levels = density_levels or ["high", "low"]
        self.noise_levels = noise_levels or ["low", "medium"]
        self.split_ratio = split_ratio
        self.render_qa = render_qa

    def _ensure_dirs(self):
        # 生成前清空 inuse/origin/qa 的旧 npz/png：防止多次生成残留导致同一
        # 构型跨 split（数据泄露）与文件数不均（T-027 实测踩到）
        import glob as _glob
        for d in [self.origin_dir, self.train_dir, self.val_dir, self.test_dir, QA_DIR]:
            os.makedirs(d, exist_ok=True)
            for f in _glob.glob(os.path.join(d, "*.npz")) +                     _glob.glob(os.path.join(d, "*.png")):
                os.remove(f)

    def generate_all(self) -> dict:
        self._ensure_dirs()
        n_geom = len(COMP_TYPES) * self.num_variants
        n_files = n_geom * len(self.density_levels) * (1 + len(self.noise_levels))

        print(f"生成数据集: {len(COMP_TYPES)} 类型 × {self.num_variants} 构型 "
              f"× {self.density_levels} 密度 × (clean+{self.noise_levels}) = {n_files} 文件")

        # 任务列表
        tasks = []
        seed = 20261016
        for comp_type in COMP_TYPES:
            for vi in range(1, self.num_variants + 1):
                for dens_name in self.density_levels:
                    tasks.append((comp_type, vi, dens_name, DENSITY_MAP[dens_name],
                                  self.noise_levels, seed))
                    seed += 1

        # 分层拆分：每个 comp_type 内部按 6:2:2 分配构型组——
        # 保证 composite（唯一含箍筋）与每种截面在三个 split 都有代表
        # （曾用全局随机分组，val 恰好无 composite → 箍筋类无验证信号）
        by_type: dict[str, list] = {}
        for comp_type in COMP_TYPES:
            by_type[comp_type] = [(comp_type, vi) for vi in range(1, self.num_variants + 1)]
        rng = np.random.default_rng(42)
        train_keys, val_keys, test_keys = set(), set(), set()
        r0, r1, r2 = self.split_ratio
        tot = r0 + r1 + r2
        for comp_type, groups in by_type.items():
            groups = list(groups)
            rng.shuffle(groups)
            n = len(groups)
            if n >= 3:
                n_test = max(1, round(n * r2 / tot))
                n_val = max(1, round(n * r1 / tot))
                n_train = max(1, n - n_val - n_test)
            else:
                n_train, n_val, n_test = 1, max(0, n - 1), 0
            train_keys.update(groups[:n_train])
            val_keys.update(groups[n_train:n_train + n_val])
            test_keys.update(groups[n_train + n_val:])
        print(f"拆分(分层): {len(train_keys)}/{len(val_keys)}/{len(test_keys)} 构型组 → train/val/test")

        stats = {"origin": 0, "train": 0, "val": 0, "test": 0, "qa": 0}
        progress = Progress(
            SpinnerColumn(), TextColumn("[bold]生成[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}"), console=_CONSOLE,
        )
        with progress:
            task_id = progress.add_task("gen", total=len(tasks), info="")
            for task in tasks:
                comp_type, vidx = task[0], task[1]
                try:
                    outputs = _generate_one(task)
                except Exception as e:
                    progress.update(task_id, advance=1,
                                    info=f"[red]失败 {comp_type} v{vidx}: {e}[/]")
                    continue

                key = (comp_type, vidx)
                for out in outputs:
                    if out["noise_level"] == "clean":
                        _save_npz(out, self.origin_dir)
                        stats["origin"] += 1
                        if self.render_qa and vidx == 1:
                            render_qa(
                                out["points"], out["labels"],
                                f"qa_{comp_type}_{out['density']}_clean.png",
                                title=f"{comp_type} {out['density']} clean",
                            )
                            stats["qa"] += 1
                    else:
                        if key in train_keys:
                            _save_npz(out, self.train_dir); stats["train"] += 1
                        elif key in val_keys:
                            _save_npz(out, self.val_dir); stats["val"] += 1
                        elif key in test_keys:
                            _save_npz(out, self.test_dir); stats["test"] += 1
                progress.update(
                    task_id, advance=1,
                    info=f"{comp_type} v{vidx:03d} {task[2]}",
                )

        print(f"\n生成完成: origin={stats['origin']} train={stats['train']} "
              f"val={stats['val']} test={stats['test']} qa={stats['qa']}")
        return stats
