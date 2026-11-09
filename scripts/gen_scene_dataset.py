"""
SDS v1 场景数据集生成驱动（T-037 M2 数据准备，docs/
t037_scene_face_pipeline.md §4.3）。

与 `lib.scene.scene_generator` CLI 的种子公式**逐字一致**（base =
seed + crc32(mode)%100000；第 i 场景 rng = abs(base + i*7919) % 2^31；
cfg_idx = i%12+1）——区别仅在本驱动支持 **index 分片**（--i0/--i1）以
多进程并行，且幂等续跑（已存在文件跳过）。改动任一处的种子公式必须
同步另一处（两个文件的 docstring 互指）。

索引维护：不写 meta_index.json（并行写会互相覆盖），落盘完成后用
`--merge-index` 扫描全部 mode 目录重建（从 npz 内 scene_meta_json 读
mode/config_id/split/member_types/n_points）。

CLI 示例（4 进程并行生成 s1×200）：
  python -m scripts.gen_scene_dataset --mode s1 --i0 0   --i1 50
  python -m scripts.gen_scene_dataset --mode s1 --i0 50  --i1 100
  ...
  python -m scripts.gen_scene_dataset --merge-index
"""

import argparse
import json
import os
import time
import zlib

import numpy as np

from lib.scene.scene_generator import _split_of, generate_scene


def gen_range(mode: str, i0: int, i1: int, root: str, seed: int = 20261026,
              types: list = None) -> None:
    os.makedirs(os.path.join(root, mode), exist_ok=True)
    base = seed + zlib.crc32(mode.encode("utf-8")) % 100000
    t0 = time.time()
    for i in range(i0, i1):
        cfg_idx = i % 12 + 1
        config_id = f"{mode}_cfg{cfg_idx:02d}"
        fname = f"scene_{mode}_cfg{cfg_idx:02d}_{i:04d}.npz"
        fpath = os.path.join(root, mode, fname)
        if os.path.exists(fpath):
            print(f"[{mode} {i}] skip（已存在）", flush=True)
            continue
        rng = np.random.default_rng(abs(base + i * 7919) % (2**31))
        t1 = time.time()
        sc = generate_scene(mode, rng, types=types)
        meta = json.loads(sc["scene_meta_json"])
        meta["config_id"] = config_id
        meta["split"] = _split_of(config_id)
        np.savez_compressed(
            fpath,
            points=sc["points"], class_id=sc["class_id"],
            member_id=sc["member_id"], face_id=sc["face_id"],
            visible=sc["visible"], face_table=sc["face_table"],
            member_table_json=sc["member_table_json"],
            scene_meta_json=json.dumps(meta, ensure_ascii=False),
            scan_poses=sc["scan_poses"],
        )
        vis_r = float(sc["visible"].mean())
        print(f"[{mode} {i}] {fname}: {meta['n_points']} pts, "
              f"{meta['n_members']} 件, {meta['n_faces']} 面, "
              f"可见率 {vis_r:.2f} ({time.time()-t1:.0f}s)", flush=True)
    print(f"[{mode}] {i0}-{i1} 完成 ({time.time()-t0:.0f}s)", flush=True)


def merge_index(root: str) -> None:
    """扫描全部 mode 目录的 scene_*.npz，重建 meta_index.json。"""
    scenes = []
    for mode in sorted(os.listdir(root)):
        mdir = os.path.join(root, mode)
        if not os.path.isdir(mdir):
            continue
        for f in sorted(os.listdir(mdir)):
            if not f.endswith(".npz"):
                continue
            rel = f"{mode}/{f}"
            with np.load(os.path.join(mdir, f),
                         allow_pickle=True) as z:
                meta = json.loads(str(z["scene_meta_json"]))
            scenes.append({
                "file": rel, "mode": mode,
                "config_id": meta.get("config_id"),
                "split": meta.get("split"),
                "member_types": meta.get("member_types"),
                "n_points": meta.get("n_points"),
            })
    scenes.sort(key=lambda s: s["file"])
    index = {"version": "sds_v1", "scenes": scenes}
    with open(os.path.join(root, "meta_index.json"), "w",
              encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    by = {}
    for s in scenes:
        by.setdefault((s["mode"], s["split"]), []).append(s)
    for (mode, split), ss in sorted(by.items()):
        print(f"  {mode}/{split}: {len(ss)}")
    print(f"meta_index.json 重建：共 {len(scenes)} 场景 → {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default=None, help="s1..s6|d1")
    ap.add_argument("--i0", type=int, default=0, help="起始 index（含）")
    ap.add_argument("--i1", type=int, default=1, help="结束 index（不含）")
    ap.add_argument("--root", default=None,
                    help="输出根目录（默认 mock/scene/v1）")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--merge-index", action="store_true")
    args = ap.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    root = args.root or os.path.join(proj, "mock", "scene", "v1")
    if args.merge_index:
        merge_index(root)
        return
    if not args.mode:
        ap.error("--mode 必填（或用 --merge-index）")
    gen_range(args.mode, args.i0, args.i1, root, seed=args.seed)


if __name__ == "__main__":
    main()
