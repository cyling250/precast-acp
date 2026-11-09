# -*- coding: utf-8 -*-
"""论文插图出图脚本包（每图一脚本，可复用可复现）。

运行约定（项目根）：
  .venv/Scripts/python.exe -I scripts/figures/<脚本>.py
产物统一落 out/figures/<图号>/（out/ 不入库）。

规范：scientific-plotting 技能（sci_style 短划网格 0.5pt；数据图必带
主网格；点云展示图内部一律 no_grid；全英文；图内不加大标题、不打
(a)(b)(c) 编号——编号由 Visio 拼稿添加；CS 黑话不得出现）。
