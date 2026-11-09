# ACP：预制混凝土构件装配连接参数的多实例检测与测量

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2013.0-red)](https://pytorch.org/)

论文《Point cloud-based multi-instance detection and measurement of assembly connection parameters of precast concrete components in stacked storage yards》（投稿于 *Journal of Building Engineering*）配套代码仓库。

**Languages / 语言：[English](README.md) | [中文](README.zh-CN.md)**

## 简介

预制混凝土（PC）构件进场验收需对三类装配连接参数——外露钢筋、灌浆套筒、箍筋——逐件检查，而既有点云方法只能输出单构件整体尺寸，无法支撑堆场堆叠场景下的实例级验收。本仓库实现 **ACP（Assembly Connection Parameter detection）**：PointMLP 网络读取逐点材质，钢筋簇锚定分组与连接面生长把场景点云转化为归一化端面视图，点级与柱式检测头按各自几何先验完成目标检测，级联套筒融合与无参数几何测量层输出全部装配参数。训练完全使用合成数据，无需人工标注。

## 仓库内容

1. **ACP 部署管线**——分割推理、分组裁剪、连接面生长、双检测头、级联融合、几何测量，含单构件 / 堆叠场景 / 真实场景三条链路。
2. **参数化合成数据生成器**（依据 JGJ 355-2015 与 GB 50010-2010）——384 个带标注单构件（8 类 × 12 配置 × 2 密度 × 2 噪声）与 SDS v1 场景集（410 个场景、6 种堆叠工况）。
3. **论文实验套件**——训练与划分（exp1）、全链路验证（exp2）、消融（exp3）、真实场景验证（exp5）、效率（exp6），规范结果 JSON 入库于 `results/`。

## 核心结果

- 单构件基准上钢筋中心位置误差 **p50 = 3.65 mm**，低于 GB 50204-2015 的 **5 mm** 公差（召回 0.986）。
- 套筒召回 **0.962**，位置误差 p50 = 0.35 mm，内径读数毫米级；箍筋肢位置 / 高度误差 p50 = 5.14 / 3.27 mm。
- 堆叠场景：立柱堆叠与平放带滚转接近单构件水平（钢筋召回 0.910 / 1.00）；立墙板堆叠与梁柱混排仍最难（0.765 / 0.717）。
- 真实堆场扫描：**12 根外露钢筋全部检出并匹配**；套筒内径误差 p50 = 2.04 mm（12 环中 3 环点云本身无数据）。
- 效率：全管线可学习参数 **< 1 M**，单 RTX 4060 Ti（8 GB）上 **31.5 s/构件**。

## 仓库结构

```
precast-acp/
├── src/
│   ├── lib/
│   │   ├── pipeline/               # 部署管线（单构件 / 场景 / 真实三链路）
│   │   ├── training/               # 训练器（点级头 / BEV头 / 语义面 / 场景分割）
│   │   ├── e2e/                    # 点级检测头、连接面生长、视图数据集
│   │   ├── measure/                # 套筒级联融合 + 几何测量
│   │   ├── segmentation/           # PointMLP 材质分割 + 推理
│   │   ├── scene/                  # 场景合成 / 堆叠工况
│   │   ├── pointcloud_generator/   # 参数化单构件生成器
│   │   └── utils/                  # 公共工具
│   ├── experiments/                # 论文实验 exp1–exp6
│   └── scripts/                    # （已并入顶层 scripts/）
├── scripts/                        # 数据生成、缓存、GT标注工具、诊断
├── results/                        # 规范结果 JSON —— 论文数字以此为准
├── mock/                           # 生成数据集（gitignore；可由生成器复现）
├── real_resource/                  # 真实场景点云 + 人工 GT（gitignore；不可再生）
├── out/                            # 权重与缓存（gitignore）
└── manuscript/                     # 投稿稿件（gitignore）
```

## 快速开始

环境要求：Python 3.10、NVIDIA GPU（参考运行使用 RTX 4060 Ti 8 GB）。

```bash
# 1. 安装依赖（torch 走 cu130 索引）
uv sync

# 2. 复现论文数字（在 src/ 下运行；规范 JSON 已入库 results/）
cd src
../.venv/Scripts/python.exe -m experiments.exp2_validate single    # 单构件部署协议
../.venv/Scripts/python.exe -m experiments.exp2_validate scene     # 堆叠场景 + 池化
../.venv/Scripts/python.exe -m experiments.exp5_real --rb-thr 0.3  # 真实堆场扫描

# 3. 本地再生成合成数据集（确定性；不入库 git）
../.venv/Scripts/python.exe -m lib.pointcloud_generator --batch                      # 单构件
cd .. && PYTHONPATH=src .venv/Scripts/python.exe -m scripts.gen_scene_dataset --mode s3 --i0 0 --i1 60   # 场景
```

注：GPU 检测头推理非确定（cuDNN 归约顺序），重跑可能有毫米级漂移或个别检出翻转。

## 数据可用性

源代码在本仓库开放获取。带标注数据集由确定性参数化生成器产出，可按上述命令本地复现，故不入库 git。

## 引用

若本仓库对您的研究有帮助，请引用对应论文（发表后补充引用信息）。
