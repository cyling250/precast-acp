# scripts/ — 可复用脚本

统一用项目根 `.venv` 解释器运行。

## 脚本清单

| 脚本 | 用途 | 典型调用 |
|---|---|---|
| `check_capture_quality.py` | 预制柱现场拍摄质量验证（Stage A 单张质量 + Stage B pycolmap SfM 配准/覆盖，输出 PASS/WARN/FAIL 与补拍建议；自包含可在任意笔记本运行） | `python scripts/check_capture_quality.py <照片根目录>`（`--init-zones` 建分区骨架；`--selftest --fast` 出发前环境演练） |
| `sleeve_gt_tool.py` | 真实套筒端面人工标注 HTML 工具（端面密度图 + 圆形画笔，导出 `sleeve_gt_manual.json`；产物见 `real_resource/`，不可再生） | `.venv/Scripts/python.exe scripts/sleeve_gt_tool.py [--cache temp_scripts/out/real_labels_cache.npz]`，浏览器开 `out/sleeve_gt/sleeve_gt_tool.html` |
| `rebar_gt_tool.py` | 真实钢筋插入位置人工标注 HTML 工具（双模式：端面 2D 画圆 + WebGL 三维旋转核对[端带点云/识别骨架线/圆环]，导出 `rebar_gt_manual.json`；产物见 `real_resource/`，不可再生） | `.venv/Scripts/python.exe scripts/rebar_gt_tool.py [--cache temp_scripts/out/real_labels_cache.npz]`，浏览器开 `out/rebar_gt/rebar_gt_tool.html` |
| `queue_runner.py` | 训练/实验队列执行器（幂等可续跑；队列配置需按实验入口编写） | `.venv/Scripts/python.exe scripts/queue_runner.py` |
