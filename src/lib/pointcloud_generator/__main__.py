"""
命令行入口：python -m lib.pointcloud_generator [选项]

用法:
  python -m lib.pointcloud_generator --type rectangular_column   # 单构件预览（标签统计+QA图）
  python -m lib.pointcloud_generator --all                       # 每类型一件预览
  python -m lib.pointcloud_generator --batch                     # 批量生成完整数据集 (112 件)
"""

import argparse
import numpy as np
import os, sys


def main():
    parser = argparse.ArgumentParser(description="预制混凝土构件点云生成器 v2（布局 C）")
    parser.add_argument("--type", type=str, default="rectangular_column",
                        help="构件类型 (默认: rectangular_column)")
    parser.add_argument("--all", action="store_true", help="遍历所有构件类型并预览")
    parser.add_argument("--batch", action="store_true", help="批量生成完整数据集")
    parser.add_argument("--no-qa", action="store_true", help="批量模式不渲染 QA 图")
    parser.add_argument("--density", type=str, default="high", help="预览密度 (high/low)")
    args = parser.parse_args()

    if args.batch:
        from ._dataset import DatasetGenerator
        dg = DatasetGenerator(render_qa=not args.no_qa)
        stats = dg.generate_all()
        print("数据集生成完成!")
        return

    from ._config import DENSITY_MAP, QA_DIR
    from ._dataset import render_qa, COMP_TYPES
    from ._builder import build_member

    types = COMP_TYPES if args.all else [args.type]
    dens = DENSITY_MAP[args.density]

    for comp_type in types:
        print(f"\n=== {comp_type} ===")
        result = build_member(comp_type, density=dens)
        pts, lbls = result["points"], result["labels"]
        unique, counts = np.unique(lbls, return_counts=True)
        names = {0: "non_member", 1: "concrete", 2: "rebar", 3: "stirrup"}
        for u, c in zip(unique, counts):
            print(f"  {names.get(u, u)}: {c} 点 ({c/len(pts)*100:.1f}%)")
        print(f"  总计: {len(pts)} 点")
        meta = result["metadata"]
        print(f"  GT: L={meta['length_mm']:.0f}mm 截面={meta['section_bbox']}"
              f" 筋数={meta['n_rebars']} 筋径={meta['rebar_diam']}"
              f" 外露={meta['protrusion_mm']:.0f} 套筒Ø{meta['sleeve_inner_diam']:.0f}/{meta['sleeve_outer_diam']:.0f}")
        render_qa(pts, lbls, f"preview_{comp_type}_{args.density}.png",
                  title=f"{comp_type} {args.density}", out_dir=QA_DIR)
        print(f"  QA 图 → {os.path.join(QA_DIR, f'preview_{comp_type}_{args.density}.png')}")


if __name__ == "__main__":
    main()
