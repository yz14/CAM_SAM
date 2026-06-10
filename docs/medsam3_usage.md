# MedSAM3 内镜分割使用指南（TODO 2）

> 对应脚本：`endoscope_medsam3.py`。
> 公共逻辑在 `seg_common.py`，候选选择复用 `endoscope_sam3_v2.py:select_candidate()`。

## 1. MedSAM3 是什么

[MedSAM3](https://github.com/Joey-S-Liu/MedSAM3)（HKUST-GZ，arXiv 2511.19046）=
SAM3 基座 + 医学数据大规模 LoRA 微调。

- **核心理念**：纯文本概念驱动分割 —— 给一个医学概念词（如 `"colon polyp"`），
  模型直接输出所有匹配实例的 mask；**无需框/点提示**即可工作（但可以加框做锚定）。
- **训练数据**：658K 图、2.86M 实例标注、330 个医学文本 ID，覆盖
  内镜（Endoscopy）、CT、MRI、PET、X-ray、超声、病理、皮肤镜、OCT 等 11 个模态。
- **参数效率**：只训练 LoRA 增量权重（约 74MB），SAM3 基座参数冻结。
- **为什么比通用 SAM3 更适合内镜息肉**：SAM3 的文本编码器是 CLIP，
  对医学术语的对齐度有限；MedSAM3 用医学数据重新对齐了 vision-language 映射，
  `"colon polyp"` 这类概念落在模型的分布内而非分布外。

结论：**可以用，且理论上比通用 SAM3 更适合内镜分割。**

## 2. 安装（服务器端）

### 2.1 前置条件

- 已安装 SAM3（`pip install -e .` 过 facebookresearch/sam3）
- 已有 SAM3 基座权重（`sam3.pt`，与 `endoscope_sam3.py` 同一个）
- Python ≥ 3.10，PyTorch ≥ 2.0，CUDA ≥ 11.7

### 2.2 克隆 MedSAM3 仓库

```bash
cd /data0/yzhen/projects
git clone https://github.com/Joey-S-Liu/MedSAM3.git
cd MedSAM3
pip install -e .       # 安装依赖（不会覆盖 SAM3，仅加 lora_layers 等模块）
```

### 2.3 下载 LoRA 权重

MedSAM3 v1 LoRA 权重托管在 HuggingFace：
[lal-Joey/MedSAM3_v1](https://huggingface.co/lal-Joey/MedSAM3_v1)

```bash
# 方法 A：huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download lal-Joey/MedSAM3_v1 best_lora_weights.pt \
    --local-dir /data0/yzhen/data/medsam3

# 方法 B：wget
wget -O /data0/yzhen/data/medsam3/best_lora_weights.pt \
    "https://huggingface.co/lal-Joey/MedSAM3_v1/resolve/main/best_lora_weights.pt"
```

> 文件约 74MB，含 LoRA 增量参数；SAM3 基座权重不需要重新下载。

### 2.4 验证安装

```bash
python -c "from lora_layers import LoRAConfig; print('MedSAM3 lora_layers OK')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

## 3. 运行 endoscope_medsam3.py

### 3.1 纯文本模式（最简，无需 CAM bbox）

```bash
python endoscope_medsam3.py \
    --image  /data0/yzhen/data/endoscope_pku1/1 \
    --output ./medsam3_out/1 \
    --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt \
    --medsam3-repo /data0/yzhen/projects/MedSAM3 \
    --lora-weights /data0/yzhen/data/medsam3/best_lora_weights.pt \
    --text-prompt "colon polyp"
```

这种模式下：模型看到整张内镜图 + `"colon polyp"` 概念，自动输出所有
过阈值的息肉实例 mask 并取并集。

### 3.2 文本 + CAM 框联合模式（推荐，与 v2 对齐）

```bash
python endoscope_medsam3.py \
    --image  /data0/yzhen/data/endoscope_pku1/1 \
    --boxes  /data0/yzhen/projects/CAM/pku1/1/boxes \
    --output ./medsam3_out/1 \
    --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt \
    --medsam3-repo /data0/yzhen/projects/MedSAM3 \
    --lora-weights /data0/yzhen/data/medsam3/best_lora_weights.pt
```

> `--text-prompt` 默认为 `"colon polyp"`；`--boxes` 给了就走文本+框联合路径。

### 3.3 质检

与 SAM3 v2 输出格式完全对齐，可直接用现有 `qa_sam3.py`：

```bash
python qa_sam3.py \
    --mask  ./medsam3_out/1/mask \
    --boxes /data0/yzhen/projects/CAM/pku1/1/boxes \
    --image /data0/yzhen/data/endoscope_pku1/1 \
    --meta  ./medsam3_out/1/meta \
    --output ./medsam3_out/1/qa_report.csv
```

## 4. 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--text-prompt` | `"colon polyp"` | 医学概念词；可试 `"polyp"` / `"colorectal polyp"` |
| `--conf-threshold` | 0.5 | 实例置信度阈值；MedSAM3 官方建议按任务在 0.5~0.8 调 |
| `--mask-threshold` | 0.5 | mask 概率二值化阈值 |
| `--expand-ratio` | 0.12 | （文本+框模式）bbox 外扩比例 |
| `--min-iou` | 0.1 | （文本+框模式）候选框与提示框最小 IoU |
| `--keep-components` | `overlap` | 后处理保留策略：`all` / `largest` / `overlap` |
| `--lora-config` | None | MedSAM3 训练 yaml（不给用内置 v1 默认值） |

### 4.1 调参策略

| 症状 | 旋钮 | 方向 |
|---|---|---|
| 什么都检不到 | `--conf-threshold` | 0.5→0.3→0.2 |
| 漏分割 / mask 偏小 | `--mask-threshold` / `--expand-ratio` | →0.35 / →0.2 |
| 过分割 / 吞背景 | `--conf-threshold` / `--mask-threshold` | →0.7 / →0.6 |
| 检到错误区域 | `--text-prompt` | 换更具体词："colon polyp" vs "polyp" |

### 4.2 与 SAM3 v2 的 A/B 对比

```bash
# A: SAM3 v2（通用 SAM3 + 文本 + 框）
python endoscope_sam3_v2.py --image ... --boxes ... --output sam3_v2_out/1

# B: MedSAM3（医学微调 + 文本 + 框）
python endoscope_medsam3.py --image ... --boxes ... --output medsam3_out/1 \
    --medsam3-repo ... --lora-weights ...

# 两份结果分别质检
python qa_sam3.py --mask sam3_v2_out/1/mask --boxes ... --output sam3_v2_out/1/qa.csv
python qa_sam3.py --mask medsam3_out/1/mask --boxes ... --output medsam3_out/1/qa.csv
```

对比 `qa_report.csv` 的 `n_flag` 列和 `mask_area_frac`，目检被 flag 样本的
`overlay/` 即可定量+定性评估两种方案的优劣。

## 5. 架构说明

```
endoscope_medsam3.py
   ├── build_medsam3():  SAM3 基座 + inject LoRA + load MedSAM3 LoRA weights
   ├── predict_text_only():  纯文本模式 —— set_text_prompt → 全实例并集
   ├── predict_with_boxes(): 文本+框模式 —— 复用 endoscope_sam3_v2.select_candidate()
   ├── seg_common.py:  IO/bbox/mask 后处理（三脚本共用）
   └── lora_layers.py:  来自 MedSAM3 仓库（通过 --medsam3-repo 导入）
```

**不需要修改 MedSAM3 仓库的任何文件**；只需要它提供 `lora_layers.py` 和 LoRA 权重。

## 6. FAQ

### Q: MedSAM3 的概念词列表有哪些？
A: v1 覆盖 330 个医学文本 ID。MedSAM3 作者尚未公开完整列表
（README 提到"几天内发布"）。根据训练数据描述，内镜模态包含
`polyp`、`lesion`、`ulcer` 等常见标注；如果 `"colon polyp"` 不在
训练集概念中，可退化到 `"polyp"` 或 `"lesion"`。

### Q: 只用 MedSAM3 还是同时跑 SAM3 v2？
A: 建议先两个都跑，用 qa_sam3.py 对比。MedSAM3 在医学数据上的
vision-language 对齐更好，但 v1 是第一版（1 epoch 数据），
通用 SAM3 + 框在某些高对比息肉上可能反而更稳。两个管线可以
**互为备选**：哪个 flag 少就用哪个，或 ensemble（mask 投票）。
