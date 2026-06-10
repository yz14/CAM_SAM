# SAM3 结肠息肉分割效果"一般般"的原因分析与改进方案（TODO 1）

> 对应脚本：`endoscope_sam3.py`（现状） → `endoscope_sam3_v2.py`（改进版）。
> 公共逻辑抽到 `seg_common.py`。

## 1. 现状链路回顾

```
ConvNeXt 分类模型 → CAM 热图 (adapt_pku1_convnext.py)
   → cam_to_boxes() 阈值化出 bbox → boxes/<stem>.json
   → endoscope_sam3.py：纯 box prompt 喂 SAM3 → mask
```

`endoscope_sam3.py` 的做法是：每个框 `set_image` → `add_geometric_prompt(box)` →
`pick_best_mask`（取全图最高 score 的实例）→ 多框并集。

## 2. 问题诊断（按影响从大到小）

### 2.1 没有用文本概念，SAM3 退化成"哑"分割器（影响最大）

SAM3 的定位是 **Promptable Concept Segmentation**：文本概念是一等公民，
几何提示（框/点）是概念的 *exemplar*（正/负样例），二者联合才是完整玩法。
纯 box prompt 时，`Sam3Processor.add_geometric_prompt` 内部会把文本设成
占位词 `"visual"`（见 sam3_image_processor.py 源码），模型只能从框内
像素"猜"你要什么 —— 息肉与周围黏膜颜色纹理连续、边界模糊，猜错或
只抠出高对比子区域（高光、血管）非常常见。

**改进**：先 `set_text_prompt("polyp")` 注入语义，再把 CAM 框作为
exemplar。框从"唯一信息源"降级为"空间锚点"，边界由概念决定。

### 2.2 CAM 框天然偏紧 / 偏移，框边即 mask 边

- CAM 高响应通常只覆盖判别性最强的中心区域（`energy_keep=0.7` 还会进一步收缩框）；
  分类模型只需看到"最像息肉的那一块"即可分类，不需要看到整个息肉。
- box prompt 对 SAM 系模型是强空间先验：框太紧 → mask 被截断（欠分割）；
  框偏移 → 抠到旁边的皱襞。

**改进**：
1. `expand_box()` 默认外扩 12%（`--expand-ratio` 可调）；
2. 上游也可以在 `adapt_pku1_convnext.py` 用 `--box-pad-ratio 0.1`、
   `--box-energy-keep 0.85` 出更松的框（两端调一头即可，别都调）。

### 2.3 候选选择策略错误：盲取全图最高分

`pick_best_mask` 直接 `argmax(scores)`。SAM3 输出的是**全图所有**过阈值
实例，最高分实例完全可能在提示框之外（另一个息肉、反光点）。

**改进**：`select_candidate()` 用 `score × IoU(候选框, 提示框)` 排序，
并用 `--min-iou` 硬过滤框外候选。

### 2.4 二值化阈值不可调，丢掉了概率信息

旧版用 `masks > 0`（processor 内部已按 0.5 二值化）。息肉边缘是
渐变的，0.5 往往保守。

**改进**：直接用 `masks_logits`（sigmoid 概率图），`--mask-threshold`
可调：欠分割调低（0.35），过分割调高（0.6）。不用重跑模型。

### 2.5 缺少 mask 后处理

qa_sam3.py 里定义的坏例模式（碎成多块 / 框外飞溅 / 高光被抠洞 /
贴边抓黑环）大多可以用便宜的后处理消掉。

**改进**：`postprocess_mask()` = 闭运算平滑 + 只保留与提示框相交的
连通域 + floodfill 补洞。

### 2.6 工程问题：每框重复编码图像 + 不落 meta

- 每个框 `set_image` 一次 = 重复跑最贵的 image backbone。改用
  `set_image` 一次 + 框间 `reset_all_prompts()`（只清提示，复用
  image embedding）。
- 旧版不存 score，qa_sam3.py 的 `--meta` 通道用不上。v2 每图写
  `meta/<stem>.json`（score / iou / 候选数），低置信样本可自动 flag。

## 3. 内镜域特有的干扰（按需启用）

这些不在 v2 默认链路里，结果仍不理想时再加：

1. **镜面高光**：息肉表面强反光会割裂 mask。可先
   `cv2.inpaint` 高光区（阈值 V>230 的小连通域）再喂 SAM3。
2. **黑边/镜头圆环**：mask 贴边时基本是抓到了视野外黑环。qa_sam3.py 的
   `border_frac` 已能 flag；也可预先把圆形视野外区域置零。
3. **多形态息肉**（扁平 vs 带蒂）：扁平息肉边界弱，靠通用 SAM3 很难，
   这正是 TODO 2 用 MedSAM3（医学数据微调）的动机。

## 4. 调参决策表（v2）

| 症状 | 旋钮 | 方向 |
|---|---|---|
| mask 偏小 / 截断 | `--expand-ratio` / `--mask-threshold` | 0.12→0.2 / 0.5→0.35 |
| mask 吞背景 | `--expand-ratio` / `--mask-threshold` | →0.05 / →0.6 |
| 整图无输出（漏检） | `--conf-threshold` | 0.4→0.25 |
| 选错目标（框外实例） | `--min-iou` | 0.1→0.5 |
| 想复现旧版对照 | `--no-text` | 关闭概念提示 |

## 5. 验证方法（无 GT 时）

1. 同一批图分别跑旧版与 v2，输出到不同目录；
2. `python qa_sam3.py --mask <out>/mask --boxes <boxes> --image <imgs> --meta <out>/meta`
   对比两份 `qa_report.csv` 的 flag 数量与分布；
3. 重点目检被 flag 的样本的 `overlay/`。

## 6. 进一步路线（本轮未实现，可拆下一轮）

- **box jitter ensemble**：对每个框做 ±5% 抖动出 3~5 个提示，mask 投票，
  对 CAM 框不稳的样本更鲁棒（代价：推理 ×N）。
- **负样例提示**：把明显的反光点/活检钳作为 `label=False` 的负框。
- **专用息肉分割模型对比基线**：Polyp-PVT、SAM 2 微调版（如
  Polyp-SAM）在 Kvasir-SEG/CVC-ClinicDB 上是 SOTA，可作为效果上限参照。
- **MedSAM3**：见 TODO 2，`endoscope_medsam3.py` + `docs/medsam3_usage.md`。
