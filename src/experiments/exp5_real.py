# -*- coding: utf-8 -*-
"""exp5 —— 真实场景实验（real_resource 1.73M 点 + 手绘 GT，不可再生）。

canonical: results/real_fullchain.json（rb_thr=0.3 定稿工作点，钢筋
12/12 检出全匹配、套筒 9 检出/8 配准内点）。
分割标签缓存 out/cache/real_labels.npz 命中则分割零重跑。

用法（cwd=src）：
  python -m experiments.exp5_real --rb-thr 0.3 [--out results/real_fullchain.json]
  （--out 缺省=只打印，canonical 防覆盖；重跑对账时给 staging 路径）
"""
from lib.pipeline import real_chain


def main():
    real_chain.run_real(real_chain.build_parser().parse_args())


if __name__ == "__main__":
    main()
