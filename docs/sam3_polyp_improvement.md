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

**改进**：`select_candidate()` 不再盲取最高分（见 §2.3.2 的进一步修正），
并用 `--min-iou` 硬过滤框外候选。

### 2.3.1 ⚠️ 回归修复：内部置信度阈值把「未见概念」的目标整批删空

> 这是 v2 第一版「连目标都分割不出来」的**直接根因**，务必理解。

`Sam3Processor._forward_grounding()` 在出 mask 前有一步硬过滤：

```python
out_probs = pred_logits.sigmoid() * presence_logit.sigmoid()  # 概念匹配分 × 存在分
keep = out_probs > self.confidence_threshold                  # 默认 0.4
out_masks = out_masks[keep]   # ← 不过阈值的实例在这里就被丢掉了
```

通用 SAM3 的文本编码器是 CLIP，对 `"polyp"` 这类**分布外医学术语**对齐很差，
`out_probs` 普遍很低。v2 第一版把 `--conf-threshold`（默认 0.4）直接喂给
`confidence_threshold`，于是**与 CAM 框重叠的目标实例也被一起删空** → 候选数 0 →
`select_candidate` 拿到空列表 → 整图 mask 全黑。

（v1 纯 box prompt 没踩这个坑：文本退化成占位词 `"visual"`，box exemplar 让
`presence`/匹配分都很高，`out_probs > 0.5` 仍能留下候选——代价是没有概念约束、假阳多。）

**改进**：把「置信度过滤」从模型内部挪到候选选择之后。
- `confidence_threshold` 固定置 **0**（`--model-conf-threshold`，保留全部候选）；
- `--conf-threshold` 改为 `select_candidate` 里**选中候选**的后置得分下限，
  默认 **0**（有框必出目标），误检多时再调高到 0.3~0.5 抑制假阳。

这样「有 CAM 框 → 必能选出框内最匹配的实例」，概念分低也不会把目标删空。

### 2.3.2 ⚠️ 真正根因：最高分候选「框准但 mask 是空的」，按 score 选必然选空

> §2.3.1 把内部阈值置 0 后，整图仍分不出目标。`debug_sam3_v2.py` 把候选摊开看，
> 才暴露真正的根因——**检测分(score) 和 mask 质量并不挂钩**。

服务器诊断输出（图 `C_AHX..M66_002`，text 模式，n=200 候选）：

```
 idx  score    IoU   score*IoU  maskArea  pred_box(px)            mask_bbox(px)
  44   0.930   0.93     0.863   0.0001  [   16,  70, 789, 803]  [  24, 430, 409, 568]  ← 选中
  41   0.104   0.34     0.035   0.1173  [  174,260, 670, 652]  [ 174, 254, 679, 659]  ← 真·息肉
  98   0.080   0.46     0.037   0.1431  [  172,260, 683, 839]  [ 175, 256, 679, 836]
```

最高分的 #44：score 0.93、pred_box 与提示框 IoU 0.93（框几乎完美贴合），
但 **maskArea≈0.0001（mask 基本是空的）**，且它的 mask_bbox 还跑到左下角、
跟自己的 pred_box 都对不上。真正的息肉 mask 在 **score 仅 0.1** 的 #41/#98 里。
旧的 `score × IoU` 选法（IoU 用的是 pred_box）必然选中 #44 → 阈值化后整图空白。

这解释了为什么 §2.3.1 之后仍分不出、且调 `--conf-threshold` 毫无变化
（被选中的就是空 mask，提不提分都一样）；也解释了为什么 v1 反而能分出来——
v1 用模型自带的二值 `output["masks"]`（logit>0）+ 纯框 `argmax(score)`，
而 v2 用 `masks_logits>0.5` + `score×IoU`，两处叠加把目标弄丢。

**改进**：`select_candidate()` 改成**按「mask 与提示框的契合度」选**，不再看 score：

```
key = coverage × precision
  coverage  = |mask ∩ 提示框| / |提示框|     # mask 填满了多少 CAM 框
  precision = |mask ∩ 提示框| / |mask|       # mask 有多少落在框内（罚框外飞溅）
```

- **空/退化 mask 直接丢弃**（`min_mask_frac`，默认 1e-4）——#44 在这步就被踢掉；
- 选「填满 CAM 框且不往框外飞溅」的候选，天然同时压掉框外假阳；
- score 仅作极小权重的同分裂项（目标 score 可能只有 0.1，不能让它主导）。

配合 `--min-coverage`（默认 0）可进一步压框外飞溅；`--min-iou` 仍做一道廉价空间初筛。

### 2.4 二值化阈值：masks_logits 是 logit，旧版当概率用导致弱边界被侵蚀

旧版直接 `masks_logits > 0.5`，但 `masks_logits` 是**原始 logit**（不是概率），
>0.5 等价于 logit>0.5（≈ sigmoid>0.62），比模型自带的 `masks`（logit>0）更严，
弱边界/低置信目标会被侵蚀到接近空。

**改进**：先 `sigmoid(masks_logits)` 转概率，再用 `--mask-threshold`（默认 0.5 =
logit>0，等价模型自带 `masks`）二值化；候选打分也优先用模型自带二值 `masks`。
欠分割调低（0.35），过分割调高（0.6），不用重跑模型。

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
| 假阳太多（框外飞溅） | `--min-coverage` / `--min-iou` | 0→0.1~0.3 / 0.1→0.3 |
| 选错目标（框外实例） | `--min-iou` / `--min-coverage` | 0.1→0.5 / 0→0.2 |
| mask 偏空/分不出 | `--mask-threshold` | 0.5→0.3 |
| 想复现旧版对照 | `--no-text` | 关闭概念提示 |

> 注意：选候选现在按 **mask∩框 的 coverage×precision**（见 §2.3.2），不看 score。
> **`--conf-threshold`（选中候选的 score 下限）默认 0，且不建议调高**——目标实例
> 的 score 可能只有 ~0.1，调高会把目标一起丢掉；要压假阳请用 `--min-coverage` /
> `--min-iou`。模型内部阈值 `--model-conf-threshold` 默认 0（保留全部候选），勿动。

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
