# CAM + SAM3 内镜病灶弱监督分割

把**分类模型的 CAM 热图**当作弱监督定位线索，转成 bbox 提示，再用 **SAM3**
（Promptable Concept Segmentation）把内镜图像里的病灶（如结肠息肉）分割出来。
仓库同时内置一套**可教学、可迁移**的 CAM 工具集（封装自
[jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)）。

> 本文档面向**没有参与过本项目的工程师**：先讲清楚整体背景与数据流，再给出
> 可照着一步步跑通的命令。

---

## 1. 背景与整体流程

本项目其实是两块，既能独立使用，也能串成一条端到端管线：

```
A. CAM 工具集（教学 / 通用解释）
   cam_runner.py + configs/*.yaml      # 统一入口，按 yaml 切换模型/算法
   examples/demo_*.py                  # 每个 CAM 算法一个独立 demo

B. 内镜「CAM → bbox → SAM3 分割」管线（本仓库主线）
   ┌─────────────────────────────────────────────────────────────┐
   │ ① 分类模型 (ConvNeXt-Base, 22 类内镜)                          │
   │      └─ adapt_pku1_convnext.py                                │
   │           ├─ 出 CAM 热图（可多算法对比）                       │
   │           └─ cam_to_boxes() 阈值化出 bbox → boxes/<stem>.json  │
   │ ②（可选）人工绿框标注 → extract_green_bbox.py → 同格式 JSON    │
   │ ③ SAM3 分割：boxes/<stem>.json + 原图 → mask + overlay        │
   │      ├─ endoscope_sam3.py     (v1, 纯 box prompt, 基线)        │
   │      └─ endoscope_sam3_v2.py  (v2, 文本概念 + box exemplar, 推荐)│
   │ ④ 无 GT 质检：qa_sam3.py 输出 qa_report.csv，flag 可疑样本      │
   └─────────────────────────────────────────────────────────────┘
```

bbox JSON 是 ① 和 ③ 之间的**唯一契约**，格式固定：

```json
{
  "image": "/path/to/xxx.jpg",
  "image_size_wh": [W, H],
  "pred_class": 3,
  "boxes_xyxy": [[x0, y0, x1, y1]]   // 原图像素坐标 xyxy
}
```

为什么需要 v2：SAM3 是**概念分割**模型，纯 box prompt（v1）会让文本退化成占位词，
模型只能从框内像素「猜」目标，息肉边界模糊时效果一般。v2 先注入文本概念
（如 `"polyp"`），把 CAM 框降级为「空间锚点」，并改进候选选择与后处理。
完整原因分析见 `docs/sam3_polyp_improvement.md`。

### 目录结构

```
.
├── core/                       # CAM 复用模块（model/image/layers/factory/visualize/logger）
├── configs/                    # CAM 统一入口的 YAML 配置（字段说明见 configs/README.md）
├── custom_models/              # 自写模型示例（供 yaml entrypoint 引用）
├── examples/                   # 每个 CAM 算法的独立 demo（教学用，少依赖）
├── docs/                       # 分析文档（sam3_polyp_improvement.md 等）
│
├── cam_runner.py               # CAM 统一入口：python cam_runner.py --config ...
├── adapt_pku1_convnext.py      # ① 自训练 ConvNeXt 分类模型的 CAM + 出 bbox
├── extract_green_bbox.py       # ② 从绿色涂抹标注图提取 bbox（替代 CAM 出框）
├── endoscope_sam3.py           # ③ SAM3 分割 v1（纯 box prompt，基线）
├── endoscope_sam3_v2.py        # ③ SAM3 分割 v2（文本概念 + box，推荐）
├── seg_common.py               # SAM3 管线公共逻辑（框外扩/后处理/坐标转换等）
├── seg_concept_predict.py      # SAM3 概念分割推理逻辑（与具体权重解耦）
├── qa_sam3.py                  # ④ 分割结果无 GT 质检 triage
└── requirements.txt
```

---

## 2. 环境准备（Step by Step）

> 服务器需有 NVIDIA GPU + CUDA（SAM3 推理默认 `--device cuda`）。

1. **创建并激活 Python 环境**（建议 Python 3.10）：

   ```bash
   conda create -n camsam3 python=3.10 -y
   conda activate camsam3
   ```

2. **安装 pip 依赖**：

   ```bash
   pip install -r requirements.txt
   ```

3. **安装 SAM3（必需，不在 PyPI，需从源码装）**。本仓库脚本依赖
   `from sam3.model_builder import build_sam3_image_model`，所以 `sam3` 必须能被 import：

   ```bash
   # 在本项目根目录下，把 SAM3 仓库克隆为可导入包 sam3/
   git clone <SAM3 仓库地址> sam3
   pip install -e sam3
   ```

   > 仓库里不附带 SAM3 源码（`sam3/` 为空目录）；请按官方说明在服务器上自行安装。

4. **准备 SAM3 权重**：把 `sam3.pt` 放到服务器某目录，并确保
   **同目录**下有 `bpe_simple_vocab_16e6.txt.gz`（脚本会自动取
   `Path(checkpoint).parent / "bpe_simple_vocab_16e6.txt.gz"`）。

   ```
   /your/ckpt/dir/
   ├── sam3.pt
   └── bpe_simple_vocab_16e6.txt.gz
   ```

5. **准备分类模型权重**（仅在用 `adapt_pku1_convnext.py` 出框时需要）：
   自训练的 ConvNeXt-Base 内镜分类权重 `best_model.pth`（22 类，MLP head，
   结构说明见 `adapt_pku1_convnext.py` 顶部 docstring）。

验证 SAM3 是否装好：

```bash
python -c "from sam3.model_builder import build_sam3_image_model; print('sam3 ok')"
```

---

## 3. 端到端跑通内镜分割（Step by Step）

下面以一个图片文件夹为例。把路径换成你自己的即可。

### Step 1 — 用 CAM 出 bbox

```bash
python adapt_pku1_convnext.py \
    --image   /data/endoscope/imgs \                     # 单张或图片文件夹
    --weights /data/ckpt/pku1_convnext_base/best_model.pth \
    --method  gradcam \
    --output  /data/out/pku1_compare                     # 输出根目录
```

产物（`--output` 下）：
- `boxes/<stem>.json` ← **下一步要用的 bbox**
- `<box_method>/<stem>.png` ← 出框用的热图
- `vis`/各算法热图（可选对比）

常用旋钮（框太紧/太松时调）：`--box-energy-keep`（越大框越大）、
`--box-thresh-value`（越小框越大）、`--box-pad-ratio`。一次跑多算法对比加
`--all-methods`（慢算法再加 `--include-slow`）。

> **替代方案**：若你有人工绿色涂抹标注图，可跳过分类模型，直接出框：
> ```bash
> python extract_green_bbox.py --image /data/annotated/x.jpg --output /data/out/green_boxes
> ```
> 产物同样是 `boxes/<stem>.json`，可直接喂给 Step 2。

### Step 2 — SAM3 分割（推荐 v2）

```bash
python endoscope_sam3_v2.py \
    --image      /data/endoscope/imgs \                  # 与 Step 1 同一批图
    --boxes      /data/out/pku1_compare/boxes \          # Step 1 产出的 boxes 目录
    --output     /data/out/sam3_v2 \
    --checkpoint /your/ckpt/dir/sam3.pt \
    --text-prompt "polyp"
```

产物（`--output` 下）：
- `mask/<stem>.png` — 二值 mask（0/255，原图分辨率）
- `overlay/<stem>.png` — 原图 + 半透明 mask + 绿色提示框（肉眼检查）
- `meta/<stem>.json` — score / iou / 候选数（供质检）

单张图：`--image` 和 `--boxes` 都传单个文件路径即可。

调参速查（详见脚本顶部 docstring 与 `docs/sam3_polyp_improvement.md`）：

| 症状 | 旋钮 | 方向 |
|---|---|---|
| 漏分割 / mask 偏小 | `--expand-ratio` | 0.12 → 0.2 |
| 过分割 / 吞背景 | `--expand-ratio` | → 0.05 |
| 假阳太多（框外飞溅） | `--min-precision` / `--min-iou` | 0.5→0.7 / 0.1→0.3 |
| 整图 / 大块被选中 | `--min-precision` / `--max-mask-frac` | →0.7 / 0.6→0.4 |
| 选错目标 | `--min-iou` | 0.1 → 0.5 |
| 想对比旧版（纯 box） | `--no-text` | 关闭文本概念 |
| 看不懂为何这样选 | `--debug` | 打印候选表与选中原因 |

> **基线 v1**：`endoscope_sam3.py`，参数 `--image/--boxes/--output/--checkpoint`
> 同名，纯 box prompt，无文本概念。主要用于和 v2 做对照。

### Step 3 — 无 GT 质检

```bash
python qa_sam3.py \
    --mask   /data/out/sam3_v2/mask \
    --boxes  /data/out/pku1_compare/boxes \
    --image  /data/endoscope/imgs \    # 可选，给了才算 mask 内亮度
    --meta   /data/out/sam3_v2/meta \  # 可选，读 SAM3 score
    --output /data/out/sam3_v2/qa_report.csv
```

输出 `qa_report.csv` 并按可疑度排序打印需重点人工复核的样本（碎块 / 框外飞溅 /
近乎空 / 吞整片视野等）。指标是**代理质量**，不能替代真实 GT 的 Dice。

---

## 4. CAM 工具集（可独立使用）

与上面的内镜管线解耦，用于学习 / 对通用模型做可解释性分析。

### 方式 A：统一入口（推荐生产/批量）

```bash
python cam_runner.py --config configs/default.yaml
# 覆盖 yaml 字段：
python cam_runner.py --config configs/default.yaml --method gradcam++ --image path/to/x.jpg
```

输出三联图 `[原图 | 热力图 | 叠加图]` 到 `outputs/`。配置字段速查见
`configs/README.md`；迁移到自定义模型（含 ViT/Swin 的 `reshape_transform`）见
`configs/{vit,swin,custom_model}.yaml` 与 `custom_models/`。

### 方式 B：单算法教学脚本

```bash
python examples/demo_gradcam.py
```

每个 `examples/demo_*.py` 自包含、不依赖 `core/`，方便复制走人。算法选择速查：

| 方法 | 需要梯度 | 需要类别 | 速度 | 一句话 |
|---|---|---|---|---|
| GradCAM | √ | √ | 快 | 经典基线 |
| GradCAM++ | √ | √ | 快 | 多目标更稳 |
| XGradCAM | √ | √ | 快 | 类别一致性更好 |
| HiResCAM | √ | √ | 快 | 元素级，定位更精细 |
| GradCAMElementWise | √ | √ | 快 | 仅保留正贡献 |
| LayerCAM | √ | √ | 快 | 浅层也可用 |
| EigenCAM | × | × | 快 | 类别无关，找显著区域 |
| EigenGradCAM | √ | √ | 中 | EigenCAM 的类别敏感版 |
| AblationCAM | × | √ | 慢 | 通道置零，前向重计算 |
| ScoreCAM | × | √ | 慢 | 激活当 mask，重新前向 |
| FullGrad | √ | √ | 中 | 自动聚合所有偏置层 |

---

## 5. 常见问题

- **`ModuleNotFoundError: No module named 'sam3'`**：SAM3 未安装或未在当前环境，
  见 §2 Step 3。
- **找不到 `bpe_simple_vocab_16e6.txt.gz`**：该文件必须与 `sam3.pt` 同目录。
- **mask 全黑 / 分不出目标**：先用 v2 + `--debug` 看候选表；常见是框太紧
  （调 `--expand-ratio`）或被 precision 门槛误删（调 `--min-precision`）。
  原理见 `docs/sam3_polyp_improvement.md` §2.3。
- **bbox 尺寸不一致告警**：JSON 里 `image_size_wh` 与实际图分辨率不符，说明出框
  和分割读的不是同一张/同分辨率图。

---

## 参考

- [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- `docs/sam3_polyp_improvement.md`：SAM3 息肉分割效果分析与 v2 改进
