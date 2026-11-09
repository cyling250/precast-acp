# ACP: Multi-Instance Detection and Measurement of Assembly Connection Parameters of Precast Concrete Components

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2013.0-red)](https://pytorch.org/)

Companion repository for the manuscript *"Point cloud-based multi-instance detection and measurement of assembly connection parameters of precast concrete components in stacked storage yards"* (submitted to the *Journal of Building Engineering*).

**Languages / 语言：[English](README.md) | [中文](README.zh-CN.md)**

## Overview

Incoming acceptance of precast concrete (PC) components requires item-by-item inspection of three classes of assembly connection parameters — projecting rebars, grouted sleeves, and stirrups — yet existing point cloud methods output only the overall dimensions of a single upright component and cannot handle instance-level acceptance in stacked storage yards. This repository implements **ACP (Assembly Connection Parameter detection)**: a PointMLP network reads per-point material, rebar-cluster-anchored grouping and connection-face growing convert the scene into normalized end-face views, point-level and pillar-based detection heads handle the targets according to their geometric priors, and cascaded sleeve fusion plus a parameter-free geometric measurement layer output all assembly parameters. Training is fully synthetic and annotation-free.

## What the repository provides

1. **The ACP pipeline** — segmentation inference, grouping & cropping, face growing, dual detection heads, cascaded fusion, geometric measurement, with single-piece / stacking-scene / real-scene deployment chains.
2. **A parametric synthetic data generator** grounded in JGJ 355-2015 and GB 50010-2010: 384 labeled single components (8 types × 12 configurations × 2 density × 2 noise levels) and the SDS v1 scene set (410 scenes, 6 stacking conditions).
3. **The paper's experiment suite** — training & splits (exp1), full-chain validation (exp2), ablations (exp3), real-scene validation (exp5), efficiency (exp6), with canonical result JSONs tracked under `results/`.

## Headline results

- Rebar center position error **p50 = 3.65 mm** on the single-component benchmark, below the **5 mm** tolerance of GB 50204-2015 (recall 0.986).
- Sleeve recall **0.962**, position error p50 = 0.35 mm, millimeter-level inner-diameter readings; stirrup leg position / height errors p50 = 5.14 / 3.27 mm.
- Stacking scenes: vertical column stacking and flat laying with roll approach single-component level (rebar recall 0.910 / 1.00); vertical wall panel stacking and mixed beam-column arrangements remain the hardest (0.765 / 0.717).
- Real storage-yard scan: **all 12 projecting rebars detected and matched**; sleeve inner-diameter error p50 = 2.04 mm (3 of 12 rings are physically absent from the scan).
- Efficiency: **< 1 M learnable parameters**, **31.5 s per component** on a single RTX 4060 Ti (8 GB).

## Repository layout

```
precast-acp/
├── src/
│   ├── lib/
│   │   ├── pipeline/               # deployment pipeline (single-piece / scene / real chains)
│   │   ├── training/               # trainers (point head / BEV head / semantic faces / scene seg)
│   │   ├── e2e/                    # point-level heads, face growing, view datasets
│   │   ├── measure/                # sleeve cascaded fusion + geometric measurement
│   │   ├── segmentation/           # PointMLP material segmentation + inference
│   │   ├── scene/                  # scene synthesis / stacking conditions
│   │   ├── pointcloud_generator/   # parametric single-piece generator
│   │   └── utils/                  # shared tools
│   ├── experiments/                # paper experiments exp1–exp6
│   └── scripts/                    # (merged into top-level scripts/)
├── scripts/                        # data generation, caching, GT annotation tools, diagnostics
├── results/                        # canonical result JSONs — cite these numbers
├── mock/                           # generated datasets (gitignored; reproducible via the generator)
├── real_resource/                  # real-scene point cloud + manual GT (gitignored; irreplaceable)
├── out/                            # weights & caches (gitignored)
└── manuscript/                     # submission manuscript (gitignored)
```

## Quick start

Requirements: Python 3.10 and an NVIDIA GPU (reference runs used an RTX 4060 Ti 8 GB).

```bash
# 1. Install dependencies (torch is pulled from the cu130 index)
uv sync

# 2. Reproduce paper numbers (from src/; canonical JSONs are tracked under results/)
cd src
../.venv/Scripts/python.exe -m experiments.exp2_validate single    # single-piece deployment protocol
../.venv/Scripts/python.exe -m experiments.exp2_validate scene     # stacking scenes + pooling
../.venv/Scripts/python.exe -m experiments.exp5_real --rb-thr 0.3  # real storage-yard scan

# 3. Regenerate the synthetic dataset (deterministic; not stored in git)
../.venv/Scripts/python.exe -m lib.pointcloud_generator --batch                      # single components
cd .. && PYTHONPATH=src .venv/Scripts/python.exe -m scripts.gen_scene_dataset --mode s3 --i0 0 --i1 60   # scenes
```

Note: GPU detection-head inference is non-deterministic (cuDNN reduction order); re-runs may drift by millimeters or flip individual detections.

## Data availability

The source code is openly available in this repository. The labeled dataset is produced by the deterministic parametric generator and can be reproduced locally with the commands above instead of being stored in git.

## Citation

If this repository helps your research, please cite the manuscript (citation to be added upon publication).
