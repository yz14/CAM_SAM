# Medical-SAM3 内镜分割使用指南

> 对应脚本：`endoscope_medical_sam3.py`。
> 文本 / 文本+框 推理逻辑在 `seg_concept_predict.py`（与 `endoscope_medsam3.py` 共用），
> IO/后处理在 `seg_common.py`，候选选择复用 `endoscope_sam3_v2.py:select_candidate()`。

## 1. Medical-SAM3 是什么

[Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3)（AIM Research Lab，
arXiv 2601.10880）= SAM3 基座 + 大规模异构医学数据**全量微调**。

- **核心理念**：和 SAM3 / MedSAM3 一样是概念驱动分割 —— 给一个医学概念词
  （如 `"colon polyp"`），模型输出匹配实例的 mask；无需框/点提示即可工作（也可加框做锚定）。
- **训练数据**：33 个数据集、10 个医学影像模态（CT/MRI/内镜/超声/X-ray/病理/皮肤镜/OCT 等）
  的 2D + 3D 图像 + 配对 mask + 文本提示。
- **与 MedSAM3 的关键区别**：
  - MedSAM3（Joey-S-Liu）= 通用 SAM3 基座 + **LoRA 增量**（约 74MB），加载时要 inject LoRA。
  - Medical-SAM3（AIM）= **完整微调权重**（约 10GB），加载方式与 `endoscope_sam3.py` 完全一样，
    `build_sam3_image_model` 指向它的 checkpoint 即可，**没有 LoRA**。

结论：**可以用，且能直接和 SAM3、MedSAM3 做 A/B 对比**，看哪个对结肠息肉更好。

## 2. 安装（服务器端）

### 2.1 前置条件

- Python ≥ 3.10，PyTorch ≥ 2.0，CUDA ≥ 11.7
- 已有本 CAM 仓库（提供 `endoscope_medical_sam3.py` 等脚本）

### 2.2 克隆 Medical-SAM3 仓库

Medical-SAM3 **自带一份 sam3 包**（含 checkpoint state_dict 的转换逻辑）。本脚本启动时会把
`--medical-sam3-repo` 指向的目录插到 `sys.path` 最前，让所有 `import sam3` 解析到这份，
保证权重能被正确加载。**不需要修改 Medical-SAM3 仓库的任何文件。**

```bash
cd /data0/yzhen/projects
git clone https://github.com/AIM-Research-Lab/Medical-SAM3.git
cd Medical-SAM3
pip install -r requirements.txt    # 装它的依赖（与官方 SAM3 一致）
```

> 若服务器上已安装官方 facebookresearch/sam3（SAM3、MedSAM3 用的那个），无需卸载：
> 本脚本只在自己进程内通过 `sys.path` 优先用 Medical-SAM3 自带的 sam3 包，
> 不影响 `endoscope_sam3.py` / `endoscope_medsam3.py`。

### 2.3 下载全量微调权重

权重托管在 HuggingFace：[ChongCong/Medical-SAM3](https://huggingface.co/ChongCong/Medical-SAM3)，
文件为 `checkpoint.pt`（约 **10GB**，是完整 SAM3 权重，不是增量）。

```bash
# 方法 A：huggingface-cli（推荐，支持断点续传）
pip install -U "huggingface_hub[cli]"
huggingface-cli download ChongCong/Medical-SAM3 checkpoint.pt \
    --local-dir /data0/yzhen/data/medical_sam3

# 方法 B：wget
wget -O /data0/yzhen/data/medical_sam3/checkpoint.pt \
    "https://huggingface.co/ChongCong/Medical-SAM3/resolve/main/checkpoint.pt"
```

### 2.4 验证安装

```bash
# 注意 PYTHONPATH 把 Medical-SAM3 仓库放最前，模拟脚本的 sys.path 行为
PYTHONPATH=/data0/yzhen/projects/Medical-SAM3 \
    python -c "from sam3.model_builder import build_sam3_image_model; print('Medical-SAM3 sam3 OK')"
```

## 3. 运行 endoscope_medical_sam3.py

### 3.1 纯文本模式（最简，无需 CAM bbox）

```bash
python endoscope_medical_sam3.py \
    --image  /data0/yzhen/data/endoscope_pku1/1 \
    --output ./medical_sam3_out/1 \
    --medical-sam3-repo /data0/yzhen/projects/Medical-SAM3 \
    --checkpoint /data0/yzhen/data/medical_sam3/checkpoint.pt \
    --text-prompt "colon polyp"
```

模型看到整张内镜图 + `"colon polyp"` 概念，自动输出所有过阈值实例 mask 并取并集。

### 3.2 文本 + CAM 框联合模式（推荐，与 v2 / MedSAM3 对齐）

```bash
python endoscope_medical_sam3.py \
    --image  /data0/yzhen/data/endoscope_pku1/1 \
    --boxes  /data0/yzhen/projects/CAM/pku1/1/boxes \
    --output ./medical_sam3_out/1 \
    --medical-sam3-repo /data0/yzhen/projects/Medical-SAM3 \
    --checkpoint /data0/yzhen/data/medical_sam3/checkpoint.pt
```

> `--text-prompt` 默认 `"colon polyp"`；`--boxes` 给了就走文本+框联合路径。

### 3.3 质检

输出格式与 SAM3 v2 / MedSAM3 完全对齐，直接用现有 `qa_sam3.py`：

```bash
python qa_sam3.py \
    --mask  ./medical_sam3_out/1/mask \
    --boxes /data0/yzhen/projects/CAM/pku1/1/boxes \
    --image /data0/yzhen/data/endoscope_pku1/1 \
    --meta  ./medical_sam3_out/1/meta \
    --output ./medical_sam3_out/1/qa_report.csv
```

## 4. 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--medical-sam3-repo` | 必填 | Medical-SAM3 仓库克隆目录（提供 sam3 包） |
| `--checkpoint` | 必填 | Medical-SAM3 全量微调权重 `checkpoint.pt` 路径 |
| `--text-prompt` | `"colon polyp"` | 医学概念词；可试 `"polyp"` / `"colorectal polyp"` |
| `--conf-threshold` | 0.5 | 实例置信度阈值；按任务在 0.5~0.8 调 |
| `--mask-threshold` | 0.5 | mask 概率二值化阈值 |
| `--expand-ratio` | 0.12 | （文本+框模式）bbox 外扩比例 |
| `--min-iou` | 0.1 | （文本+框模式）候选框与提示框最小 IoU |
| `--min-precision` | 0.5 | （文本+框模式）候选 mask 落在框内的最小比例 |
| `--keep-components` | `overlap` | 后处理保留策略：`all` / `largest` / `overlap` |
| `--bpe-path` | None | BPE 词表；不给按 checkpoint 同目录 / 仓库 assets / 包内默认 依次找 |

### 4.1 调参策略

| 症状 | 旋钮 | 方向 |
|---|---|---|
| 什么都检不到 | `--conf-threshold` | 0.5→0.3→0.2 |
| 漏分割 / mask 偏小 | `--mask-threshold` / `--expand-ratio` | →0.35 / →0.2 |
| 过分割 / 吞背景 | `--conf-threshold` / `--mask-threshold` | →0.7 / →0.6 |
| 检到错误区域 | `--text-prompt` | 换更具体词："colon polyp" vs "polyp" |

### 4.2 三模型 A/B/C 对比

```bash
# A: SAM3 v2（通用 SAM3 + 文本 + 框）
python endoscope_sam3_v2.py --image ... --boxes ... --output sam3_v2_out/1 \
    --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt

# B: MedSAM3（通用 SAM3 + LoRA 增量 + 文本 + 框）
python endoscope_medsam3.py --image ... --boxes ... --output medsam3_out/1 \
    --medsam3-repo ... --lora-weights ...

# C: Medical-SAM3（全量微调 + 文本 + 框）
python endoscope_medical_sam3.py --image ... --boxes ... --output medical_sam3_out/1 \
    --medical-sam3-repo ... --checkpoint /data0/yzhen/data/medical_sam3/checkpoint.pt

# 三份结果分别质检后对比 qa_report.csv 的 n_flag / mask_area_frac
python qa_sam3.py --mask sam3_v2_out/1/mask      --boxes ... --output sam3_v2_out/1/qa.csv
python qa_sam3.py --mask medsam3_out/1/mask      --boxes ... --output medsam3_out/1/qa.csv
python qa_sam3.py --mask medical_sam3_out/1/mask --boxes ... --output medical_sam3_out/1/qa.csv
```

对比 `n_flag` 列和 `mask_area_frac`，目检被 flag 样本的 `overlay/`，即可定量+定性评估三种方案优劣。

## 5. 架构说明

```
endoscope_medical_sam3.py
   ├── _bootstrap_medical_sam3_repo():  import sam3 之前把仓库插到 sys.path 最前
   ├── build_medical_sam3():  build_sam3_image_model(checkpoint=全量微调权重)  # 无 LoRA
   ├── seg_concept_predict.py:  predict_text_only / predict_with_boxes（与 MedSAM3 共用）
   ├── seg_common.py:  IO/bbox/mask 后处理（多脚本共用）
   └── sam3 包:  来自 Medical-SAM3 仓库（通过 --medical-sam3-repo 注入）
```

与 MedSAM3 的唯一实现差异是 `build_*`：Medical-SAM3 直接加载完整权重，不需要 inject LoRA。

## 6. FAQ

### Q: 为什么要 `--medical-sam3-repo`，而 SAM3 v2 不用？
A: SAM3 v2 用服务器上已安装的官方 sam3 包加载官方 `sam3.pt`。Medical-SAM3 的权重来自它自己的
视频训练管线，state_dict 的键布局/转换逻辑在它**自带的 sam3 包**里最稳妥；插到 `sys.path` 最前
即可保证正确加载，且不影响其它脚本。

### Q: 概念词应该填什么？
A: 先试 `"colon polyp"`，效果不好退到 `"polyp"` / `"colorectal polyp"` / `"lesion"`。
Medical-SAM3 训练覆盖内镜模态，息肉类概念应在分布内。

### Q: 显存/权重很大跑不动？
A: `checkpoint.pt` 约 10GB（完整 SAM3 权重）。推理显存与 SAM3 v2 同量级；如 OOM，
确认没有同时加载多个模型，必要时 `--device cpu` 验证流程（慢）。
