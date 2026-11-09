"""
数据集加载器: GPU缓存训练加载器、验证/测试数据集、工厂函数.

训练集: GpuCachedTrainLoader — 物理球查询采样 + GPU缓存 + 正样本平衡
验证/测试集: RandomValDataset — 固定种子球查询采样 (一次性生成不变)
消融用: RandomSampleDataset — 纯随机采样 (无物理邻域约束)
"""

import os, glob, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree

from .common import TRAIN_DIR, VAL_DIR, TEST_DIR

PATCH_RADIUS_MM = 150.0  # 球查询半径 (mm)
NUM_POINTS = 8192
# v2.1 密度均衡：patch 生成前体素细化（每体素保留一个原始点）。
# 阈值与生成器 _config.VOXEL_THIN_MM、真实推理 real_case_e2e 一致（3mm cap）。
try:
    from ..pointcloud_generator._config import VOXEL_THIN_MM
except Exception:  # 独立运行 utils 时兜底
    VOXEL_THIN_MM = 3.0


def voxel_thin(pts: np.ndarray, *arrays, voxel_mm: float = None):
    """体素细化：每体素保留一个原始点（标签/成员关系不破坏）。"""
    if voxel_mm is None:
        voxel_mm = VOXEL_THIN_MM
    keys = np.floor(pts / voxel_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return (pts[idx],) + tuple(a[idx] for a in arrays)


def _list_npz_files(directory: str) -> list[str]:
    """列出目录下所有 NPZ 文件."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise FileNotFoundError(f"未找到 NPZ 文件: {directory}")
    return files


def _load_geo_features(npz_path: str, num_points: int) -> np.ndarray:
    """加载几何特征, 不存在时返回全零 (3通道)."""
    geo_path = npz_path.replace(".npz", "_geo.npy")
    if os.path.exists(geo_path):
        return np.load(geo_path).astype(np.float32)
    return np.zeros((num_points, 3), dtype=np.float32)


# ═══════════════════════════════════════════════════════
#  训练集: GPU 缓存加载器
# ═══════════════════════════════════════════════════════


class GpuCachedTrainLoader:
    """训练数据一次性生成并缓存在 GPU, 每 epoch 在显存中打乱索引."""

    def __init__(
        self,
        file_paths: list,
        num_points: int = 8192,
        batch_size: int = 8,
        keep_ratio: float = 0.4,
        seed: int = 42,
        device: torch.device = None,
        pos_ratio: float = 0.8,
        samples_per_epoch: int = None,
    ):
        self.num_points = num_points
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._samples_per_epoch = samples_per_epoch

        # ── 检查磁盘缓存 ──
        cache_dir = os.path.dirname(file_paths[0]) if file_paths else TRAIN_DIR
        cache_name = (f".gpu_cache_s{seed}_n{num_points}_k{keep_ratio}_p{pos_ratio}"
                      f"_r{int(PATCH_RADIUS_MM)}_t{int(VOXEL_THIN_MM * 10)}.pt")
        cache_path = os.path.join(cache_dir, cache_name)
        _loaded_from_cache = False

        if os.path.exists(cache_path):
            _t0 = time.time()
            state = torch.load(cache_path, map_location=device, weights_only=False)
            self.pts = state["pts"].to(device, non_blocking=True)
            self.geo = state["geo"].to(device, non_blocking=True)
            self.lbl = state["lbl"].to(device, non_blocking=True)
            self.scale = state["scale"].to(device, non_blocking=True)
            self.N_per_epoch = len(self.pts)
            mb = (
                (self.pts.numel() + self.geo.numel() + self.lbl.numel())
                * 4
                / 1024
                / 1024
            )
            print(
                f"  [缓存] 从 {cache_path} 加载 {self.N_per_epoch} patches ({mb:.0f}MB, {time.time() - _t0:.0f}s)"
            )
            _loaded_from_cache = True

        if not _loaded_from_cache:
            self._generate_patches(
                file_paths, num_points, keep_ratio, seed, device, pos_ratio
            )

        self._loaded_from_cache = _loaded_from_cache

    def _generate_patches(
        self, file_paths, num_points, keep_ratio, seed, device, pos_ratio
    ):
        """生成 patches 并缓存到磁盘 + GPU."""
        rng = np.random.default_rng(seed)

        # ── 1. 加载全部文件 + 建 KDTree ──
        _t0 = time.time()
        cache = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            pts, lbl, geo = voxel_thin(pts, lbl, geo)  # v2.1 密度均衡
            tree = cKDTree(pts)
            cache.append(
                {
                    "points": pts,
                    "geo": geo,
                    "labels": lbl,
                    "tree": tree,
                    "basename": os.path.basename(f),
                    "N": len(pts),
                }
            )
        n_total = sum(c["N"] for c in cache)
        print(
            f"  [加载] {len(cache)} 文件已缓存 (≈{n_total / 1e6:.1f}M 点, {time.time() - _t0:.0f}s)"
        )

        # ── 2. 生成 patches ──
        _t0 = time.time()
        all_patches = []
        file_order = rng.permutation(len(cache))

        from rich.progress import Progress

        with Progress() as progress:
            file_task = progress.add_task("[cyan]生成 patches", total=len(file_order))

            for fi in file_order:
                c = cache[fi]
                pts, geo, lbl, tree = c["points"], c["geo"], c["labels"], c["tree"]
                available = np.ones(c["N"], dtype=bool)
                n_fail = 0
                n_from_this_file = 0

                while available.sum() >= num_points:
                    if n_fail >= 50:
                        break
                    avail_indices = np.where(available)[0]
                    seed_p = rng.choice(avail_indices)
                    ball_idx = tree.query_ball_point(pts[seed_p], r=PATCH_RADIUS_MM)
                    candidates = np.array(
                        [i for i in ball_idx if available[i]], dtype=np.int64
                    )
                    if len(candidates) >= num_points:
                        chosen = rng.choice(candidates, num_points, replace=False)
                    else:
                        chosen_set = set(candidates.tolist())
                        k_needed = min(num_points * 2, len(avail_indices))
                        _, knn_idx = tree.query(pts[seed_p].reshape(1, -1), k=k_needed)
                        for i in knn_idx[0]:
                            if available[i]:
                                chosen_set.add(int(i))
                                if len(chosen_set) >= num_points:
                                    break
                        chosen = np.array(list(chosen_set)[:num_points], dtype=np.int64)
                    if len(chosen) < num_points:
                        n_fail += 1
                        continue
                    n_fail = 0
                    available[chosen] = False

                    patch_pts = pts[chosen]
                    center = patch_pts.mean(axis=0)
                    patch_pts = patch_pts - center
                    scale_mm = np.max(np.linalg.norm(patch_pts, axis=1))
                    patch_pts = patch_pts / (scale_mm + 1e-8)

                    n_pos = (lbl[chosen] == 2).sum() + (lbl[chosen] == 3).sum()
                    all_patches.append(
                        (
                            patch_pts,
                            geo[chosen],
                            lbl[chosen],
                            scale_mm,
                            n_pos / num_points,
                        )
                    )
                    n_from_this_file += 1

                progress.update(
                    file_task,
                    advance=1,
                    description=f"[cyan]{c['basename'][:40]} ({n_from_this_file} patches)",
                )

        # ── 3. 正样本平衡筛选 ──
        rng.shuffle(all_patches)
        has_pos = [p for p in all_patches if p[4] > 0]
        no_pos = [p for p in all_patches if p[4] == 0]
        n_concrete_target = max(
            0, min(len(no_pos), int(len(has_pos) / pos_ratio - len(has_pos)))
        )
        filtered = has_pos + no_pos[:n_concrete_target]
        rng.shuffle(filtered)
        self.N_per_epoch = len(filtered)

        pos_ratios = np.array([p[4] for p in all_patches])
        sorted_ratios = np.sort(pos_ratios)[::-1]
        print(
            f"  [生成] {len(all_patches)} 总patches → {self.N_per_epoch} 训练patches "
            f"({len(has_pos)} 含钢筋/套筒 + {n_concrete_target} 纯混凝土, "
            f"ratio={len(has_pos) / max(1, self.N_per_epoch):.2f}, "
            f"{time.time() - _t0:.0f}s)"
        )
        print(
            f"  [分布] max={sorted_ratios[0]:.3f} "
            f"P90={sorted_ratios[max(0, int(len(sorted_ratios) * 0.1))]:.3f} "
            f"P75={sorted_ratios[max(0, int(len(sorted_ratios) * 0.25))]:.3f} "
            f"P50={sorted_ratios[max(0, int(len(sorted_ratios) * 0.5))]:.3f}"
        )

        # ── 4. 转换到 GPU tensor ──
        _t0 = time.time()
        self.pts = torch.stack([torch.from_numpy(p[0]) for p in filtered]).to(device)
        self.geo = torch.stack([torch.from_numpy(p[1]) for p in filtered]).to(device)
        self.lbl = torch.stack([torch.from_numpy(p[2]) for p in filtered]).to(device)
        self.scale = torch.tensor(
            [p[3] for p in filtered], dtype=torch.float32, device=device
        )
        mb = (self.pts.numel() + self.geo.numel() + self.lbl.numel()) * 4 / 1024 / 1024
        print(
            f"  [GPU] {self.N_per_epoch} patches {mb:.0f}MB → {device} ({time.time() - _t0:.0f}s)"
        )

        # ── 5. 保存磁盘缓存 ──
        _t0 = time.time()
        cache_dir = os.path.dirname(file_paths[0]) if file_paths else TRAIN_DIR
        cache_name = (f".gpu_cache_s{seed}_n{num_points}_k{keep_ratio}_p{pos_ratio}"
                      f"_r{int(PATCH_RADIUS_MM)}_t{int(VOXEL_THIN_MM * 10)}.pt")
        cache_path = os.path.join(cache_dir, cache_name)
        torch.save(
            {
                "pts": self.pts.cpu(),
                "geo": self.geo.cpu(),
                "lbl": self.lbl.cpu(),
                "scale": self.scale.cpu(),
            },
            cache_path,
        )
        print(f"  [缓存] 保存至 {cache_path} ({time.time() - _t0:.0f}s)")

    def __iter__(self):
        n_use = self._samples_per_epoch if self._samples_per_epoch else self.N_per_epoch
        n_use = min(n_use, self.N_per_epoch)
        indices = torch.randperm(self.N_per_epoch, device=self._device)[:n_use]
        for i in range(0, n_use, self.batch_size):
            idx = indices[i : i + self.batch_size]
            yield (self.pts[idx], self.geo[idx], self.lbl[idx], self.scale[idx])

    def __len__(self):
        n_use = self._samples_per_epoch if self._samples_per_epoch else self.N_per_epoch
        n_use = min(n_use, self.N_per_epoch)
        return (n_use + self.batch_size - 1) // self.batch_size


# ═══════════════════════════════════════════════════════
#  验证/测试集
# ═══════════════════════════════════════════════════════


class RandomValDataset(Dataset):
    """验证/测试集: 固定种子球查询采样 (一次性生成不变)."""

    def __init__(self, split: str = "val", num_points: int = 8192, seed: int = 42):
        assert split in ("val", "test"), "RandomValDataset 用于 val 或 test"
        data_dir = VAL_DIR if split == "val" else TEST_DIR
        file_paths = _list_npz_files(data_dir)
        self.num_points = num_points

        # ── 检查磁盘缓存 ──
        cache_name = (f".val_cache_{split}_s{seed}_n{num_points}"
                      f"_r{int(PATCH_RADIUS_MM)}_t{int(VOXEL_THIN_MM * 10)}.pt")
        cache_path = os.path.join(data_dir, cache_name)
        if os.path.exists(cache_path):
            _t0 = time.time()
            state = torch.load(cache_path, weights_only=False)
            self.pts = state["pts"]
            self.geo = state["geo"]
            self.lbl = state["lbl"]
            self.scale = state["scale"]
            n = len(self.pts)
            self.patches = list(range(n))
            print(f"  [缓存] {split}: {n} patches 已加载 ({time.time() - _t0:.0f}s)")
            return

        rng = np.random.default_rng(seed)
        self.patches = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            pts, lbl, geo = voxel_thin(pts, lbl, geo)  # v2.1 密度均衡
            tree = cKDTree(pts)
            N = len(pts)
            available = np.ones(N, dtype=bool)
            n_fail = 0

            while available.sum() >= num_points:
                if n_fail >= 50:
                    break
                avail_indices = np.where(available)[0]
                seed_p = rng.choice(avail_indices)
                ball_idx = tree.query_ball_point(pts[seed_p], r=PATCH_RADIUS_MM)
                candidates = np.array(
                    [i for i in ball_idx if available[i]], dtype=np.int64
                )
                if len(candidates) >= num_points:
                    chosen = rng.choice(candidates, num_points, replace=False)
                else:
                    chosen_set = set(candidates.tolist())
                    k_needed = min(num_points * 2, len(avail_indices))
                    _, knn_idx = tree.query(pts[seed_p].reshape(1, -1), k=k_needed)
                    for i in knn_idx[0]:
                        if available[i]:
                            chosen_set.add(int(i))
                            if len(chosen_set) >= num_points:
                                break
                    chosen = np.array(list(chosen_set)[:num_points], dtype=np.int64)
                if len(chosen) < num_points:
                    n_fail += 1
                    continue
                n_fail = 0
                available[chosen] = False

                patch_pts = pts[chosen]
                center = patch_pts.mean(axis=0)
                patch_pts = patch_pts - center
                scale = np.max(np.linalg.norm(patch_pts, axis=1))
                patch_pts = patch_pts / (scale + 1e-8)
                self.patches.append((patch_pts, geo[chosen], lbl[chosen], scale))

        print(
            f"[RandomValDataset] {split}: {len(self.patches)} patches "
            f"(来自 {len(file_paths)} 文件, 一次性生成固定不变)"
        )

        # ── 保存磁盘缓存 ──
        cache_path = os.path.join(data_dir, cache_name)
        pts_t = torch.stack([torch.from_numpy(p[0]) for p in self.patches])
        geo_t = torch.stack([torch.from_numpy(p[1]) for p in self.patches])
        lbl_t = torch.stack([torch.from_numpy(p[2]) for p in self.patches])
        scale_t = torch.tensor([p[3] for p in self.patches], dtype=torch.float32)
        torch.save(
            {"pts": pts_t, "geo": geo_t, "lbl": lbl_t, "scale": scale_t}, cache_path
        )
        print(f"  [缓存] 保存至 {cache_path}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if hasattr(self, "pts") and isinstance(self.patches[0], int):
            return (self.pts[idx], self.geo[idx], self.lbl[idx], self.scale[idx])
        pts, geo, lbl, scale = self.patches[idx]
        return (
            torch.from_numpy(pts),
            torch.from_numpy(geo),
            torch.from_numpy(lbl),
            torch.tensor(scale, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════
#  随机采样数据集 (消融 A1a)
# ═══════════════════════════════════════════════════════


class RandomSampleDataset(Dataset):
    """纯随机采样 (无球查询), 用于消融实验."""

    def __init__(
        self,
        file_paths: list,
        num_points: int = 8192,
        seed: int = 42,
        num_samples_per_file: int = 5,
    ):
        rng = np.random.default_rng(seed)
        self.patches = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            N = len(pts)

            for _ in range(num_samples_per_file):
                if N < num_points:
                    chosen = rng.choice(N, num_points, replace=True)
                else:
                    chosen = rng.choice(N, num_points, replace=False)

                patch_pts = pts[chosen]
                center = patch_pts.mean(axis=0)
                patch_pts = patch_pts - center
                scale = np.max(np.linalg.norm(patch_pts, axis=1))
                patch_pts = patch_pts / (scale + 1e-8)
                self.patches.append((patch_pts, geo[chosen], lbl[chosen], scale))

        print(
            f"[RandomSampleDataset] {len(self.patches)} samples "
            f"(来自 {len(file_paths)} 文件, 每文件 {num_samples_per_file} 个随机块)"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        pts, geo, lbl, scale = self.patches[idx]
        return (
            torch.from_numpy(pts),
            torch.from_numpy(geo),
            torch.from_numpy(lbl),
            torch.tensor(scale, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════


def create_dataloaders(
    num_points: int = 8192,
    batch_size: int = 8,
    keep_ratio: float = 0.4,
    seed: int = 42,
    file_filter: callable = None,
    only_test: bool = False,
    pos_ratio: float = 0.8,
    samples_per_epoch: int = None,
    random_sample: bool = False,
    val_batch_mult: int = 1,
):
    """返回 (make_train_loader, make_val_loader, make_test_loader)."""
    train_files = _list_npz_files(TRAIN_DIR) if not only_test else []
    val_files = _list_npz_files(VAL_DIR) if not only_test else []
    test_files = _list_npz_files(TEST_DIR)
    if file_filter is not None:
        train_files = sorted(f for f in train_files if file_filter(f))
        val_files = sorted(f for f in val_files if file_filter(f))
        test_files = sorted(f for f in test_files if file_filter(f))

    print(
        f"[DataLoader] 训练: {len(train_files)} 文件 | "
        f"验证: {len(val_files)} 文件 | 测试: {len(test_files)} 文件"
    )

    if only_test:
        if random_sample:
            test_dataset = RandomSampleDataset(test_files, num_points, seed=seed + 200)
        else:
            test_dataset = RandomValDataset(
                split="test", num_points=num_points, seed=seed + 200
            )

        def make_test_loader():
            return DataLoader(
                test_dataset,
                batch_size=batch_size * val_batch_mult,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        def make_dummy():
            raise RuntimeError("only_test 模式下未创建该 DataLoader")

        return make_dummy, make_dummy, make_test_loader

    if random_sample:
        train_dataset = RandomSampleDataset(train_files, num_points, seed=seed)
        val_dataset = RandomSampleDataset(val_files, num_points, seed=seed + 100)
        test_dataset = RandomSampleDataset(test_files, num_points, seed=seed + 200)

        def make_train_loader():
            return DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

        def make_val_loader():
            return DataLoader(
                val_dataset,
                batch_size=batch_size * val_batch_mult,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        def make_test_loader():
            return DataLoader(
                test_dataset,
                batch_size=batch_size * val_batch_mult,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = GpuCachedTrainLoader(
            train_files,
            num_points,
            batch_size,
            keep_ratio=keep_ratio,
            seed=seed,
            device=device,
            pos_ratio=pos_ratio,
            samples_per_epoch=samples_per_epoch,
        )

        val_dataset = RandomValDataset(
            split="val", num_points=num_points, seed=seed + 100
        )
        test_dataset = RandomValDataset(
            split="test", num_points=num_points, seed=seed + 200
        )

        def make_train_loader():
            return train_loader

        def make_val_loader():
            return DataLoader(
                val_dataset,
                batch_size=batch_size * val_batch_mult,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        def make_test_loader():
            return DataLoader(
                test_dataset,
                batch_size=batch_size * val_batch_mult,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

    return make_train_loader, make_val_loader, make_test_loader
