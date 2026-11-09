# -*- coding: utf-8 -*-
"""lib.pipeline —— 预制构件装配连接参数检测·部署管线（冻结胶水）。

三个链路 + 一个批量池化：
  single_chain  单件链：面来源×5 → 面视图 → 三检测头 → 五组装配参数 → 评估
  scene_chain   场景链：筋簇锚定裁剪 → 构件级联调（候选择优/端别守卫/认证视图）
  scene_batch   场景批量 + 工况池化（池化口径唯一权威实现）
  real_chain    真实链：手绘 GT 两轮刚体/成对 RANSAC 配准协议

⚠️ 冻结红线（不经全量对账不得改动）：视图 u 轴手性（P-057）、
sleeve_fusion.face_band 的 rel 系 pk（P-059/061/063）、top_view 面法向
参数化（P-064）、裁剪盒箍筋腿外扩（P-066）、端别翻转守卫（P-068）。
canonical 数字一律以 results/ 入库 json 为准（P-072）。
"""
from lib.pipeline.single_chain import run_single, run_one, eval_one, aggregate
from lib.pipeline.scene_chain import run_scene
from lib.pipeline.scene_batch import pool_evals, pool_scene_json, pick_scenes
from lib.pipeline.real_chain import run_real, load_heads

__all__ = ["run_single", "run_one", "eval_one", "aggregate", "run_scene",
           "pool_evals", "pool_scene_json", "pick_scenes", "run_real",
           "load_heads"]
