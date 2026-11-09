# -*- coding: utf-8 -*-
"""exp2 —— 全场景最终指标验证（论文主结果表入口）。

子命令：
  single        单件部署口径（--face pointgrow-pred 定稿臂）
                canonical: results/e2e_fullchain/e2e_fullchain_pointgrow-pred.json
  scene         场景批量（s3/s4/s5/s6 每工况每 cfg 一件 ×3）+ 工况池化
                canonical: results/scene_pooled.json + results/scene_batch/
  pooled-only   只对已有 json 重新池化（不跑推理）

确定性（P-072）：分割标签缓存 out/cache/scene_batch/ 命中则分割零重跑；
检测头 GPU 推理有非确定性——复跑允许毫级漂移与个别检出翻转，
引用数字一律以 results/ 入库 json 为准。
选场规则=每 cfg 一件（scene_batch.pick_scenes，P-071：改动=换评测集）。

用法（cwd=src）：
  python -m experiments.exp2_validate single [--limit 20] [--out PATH]
  python -m experiments.exp2_validate scene [--modes s3,s4,s5,s6] [--per-mode 3]
  python -m experiments.exp2_validate pooled-only
"""
import sys


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    sub, rest = argv[0], argv[1:]
    if sub == "single":
        from lib.pipeline import single_chain
        args = single_chain.build_parser().parse_args(rest)
        if "--face" not in rest:
            args.face = "pointgrow-pred"      # 部署口径定稿臂
        single_chain.run_single(args)
    elif sub in ("scene", "pooled-only"):
        from lib.pipeline import scene_batch
        if sub == "pooled-only" and "--pooled-only" not in rest:
            rest = rest + ["--pooled-only"]
        scene_batch.main(rest)
    else:
        raise SystemExit(f"未知子命令 {sub!r}（可选 single/scene/pooled-only）")


if __name__ == "__main__":
    main()
