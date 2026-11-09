# -*- coding: utf-8 -*-
"""预制柱扫描照片集 SfM 重构（pycolmap 增量式）。

输入：G:\\图片\\预制柱扫描 的 80 张 JPG（复制到 ASCII 路径规避 COLMAP 中文路径坑）。
产出：out/sfm/prefab_scan/sparse/<best>/（COLMAP 稀疏模型）
      out/sfm/prefab_scan/sfm_points.npz（xyz float32 + rgb uint8，供三视图渲染）。
"""
import shutil
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]
IMG_SRC = Path(r"G:\图片\预制柱扫描")
WS = ROOT / "out" / "sfm" / "prefab_scan"
images_dir = WS / "images"
db = WS / "database.db"
sparse = WS / "sparse"


def main():
    WS.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    # Windows glob 大小写不敏感：*.JPG/*.jpg 会把同一批文件数两遍，改用 iterdir 去重
    imgs = sorted(p for p in IMG_SRC.iterdir() if p.suffix.lower() == ".jpg")
    print(f"source images: {len(imgs)}", flush=True)
    have = [p for p in images_dir.iterdir() if p.suffix.lower() == ".jpg"]
    if len(have) != len(imgs):
        for p in have:
            p.unlink()
        for f in imgs:
            shutil.copy2(f, images_dir / f.name)
        print("images copied to", images_dir, flush=True)

    if db.exists():
        db.unlink()
    # 本 wheel 为 CPU-only 构建（无 CUDA SIFT/GPU 匹配），限线程与降采样控制内存
    fe = pycolmap.FeatureExtractionOptions()
    fe.max_image_size = 2000
    fe.num_threads = 8
    print("== extract_features (cpu, max_image_size=2000, threads=8) ==", flush=True)
    pycolmap.extract_features(database_path=db, image_path=images_dir,
                              camera_mode=pycolmap.CameraMode.AUTO,
                              extraction_options=fe)
    print("== match_exhaustive (cpu) ==", flush=True)
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = 8
    mo.use_gpu = False
    pycolmap.match_exhaustive(database_path=db, matching_options=mo)
    print("== incremental_mapping ==", flush=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    maps = pycolmap.incremental_mapping(database_path=db, image_path=images_dir,
                                        output_path=sparse)
    stats = {k: (len(v.images), v.num_points3D) for k, v in maps.items()}
    print("models (reg_images, points3D):", stats, flush=True)
    best = max(maps, key=lambda k: maps[k].num_points3D)
    rec = pycolmap.Reconstruction(str(sparse / str(best)))
    print(f"best={best}: images={len(rec.images)} points3D={rec.num_points3D}",
          flush=True)
    xyz = np.empty((rec.num_points3D, 3), np.float32)
    rgb = np.empty((rec.num_points3D, 3), np.uint8)
    for i, p in enumerate(rec.points3D.values()):
        xyz[i] = p.xyz
        rgb[i] = p.color
    out = WS / "sfm_points.npz"
    np.savez_compressed(out, xyz=xyz, rgb=rgb)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
