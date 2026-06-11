"""MedSAM3 内镜分割预测（医学概念文本提示版）
============================================

MedSAM3（https://github.com/Joey-S-Liu/MedSAM3）= SAM3 基座 + 医学数据
LoRA 微调权重。它是**纯文本概念驱动**的医学分割模型，训练数据覆盖内镜
（Endoscopy）模态，对"结肠息肉"这类病灶可以直接用概念词分割，
理论上比通用 SAM3 更懂医学影像 —— 见 docs/medsam3_usage.md。

本脚本与 endoscope_sam3.py 的 IO 完全对齐：

输入
----
1) 内镜原图：单张图片或图片文件夹
2) （可选）bbox JSON：CAM 产出的 boxes/<stem>.json；给了就用框筛选/锚定
   文本检出的实例，不给则纯文本模式输出全部检出实例

输出
----
<out>/mask/<stem>.png      二值 mask（0/255，原图分辨率）
<out>/overlay/<stem>.png   原图 + 半透明 mask + 绿色提示框
<out>/meta/<stem>.json     score / 候选数（qa_sam3.py --meta 可读）

两种工作模式
------------
1. 纯文本模式（不传 --boxes）::

       python endoscope_medsam3.py \\
           --image /data0/yzhen/data/endoscope_pku1/1 \\
           --output ./medsam3_out/1 \\
           --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt \\
           --medsam3-repo /data0/yzhen/projects/MedSAM3 \\
           --lora-weights /data0/yzhen/data/medsam3/best_lora_weights.pt \\
           --text-prompt "colon polyp"

2. 文本 + CAM 框联合模式（传 --boxes，行为对齐 endoscope_sam3_v2.py）::

       python endoscope_medsam3.py \\
           --image  /data0/yzhen/data/endoscope_pku1/1 \\
           --boxes  /data0/yzhen/projects/CAM/pku1/1/boxes \\
           --output ./medsam3_out/1 \\
           --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt \\
           --medsam3-repo /data0/yzhen/projects/MedSAM3 \\
           --lora-weights /data0/yzhen/data/medsam3/best_lora_weights.pt

安装与权重下载见 docs/medsam3_usage.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from endoscope_sam3_v2 import _sigmoid, select_candidate
from seg_common import (
    check_size_consistency,
    collect_pairs,
    expand_box,
    load_boxes_json,
    postprocess_mask,
    save_results,
    squeeze_masks,
    to_numpy,
    xyxy_to_norm_cxcywh,
)

# MedSAM3 v1 发布时使用的 full LoRA 配置（configs/full_lora_config.yaml）；
# 加载 LoRA 权重时结构必须与训练时一致，故内置一份兜底，--lora-config 可覆盖。
_DEFAULT_LORA_CFG = dict(
    rank=16,
    alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj",
        "qkv", "proj", "fc1", "fc2",
        "c_fc", "c_proj",
        "linear1", "linear2",
    ],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=True,
    apply_to_geometry_encoder=True,
    apply_to_detr_encoder=True,
    apply_to_detr_decoder=True,
    apply_to_mask_decoder=True,
)


def load_lora_cfg(config_path: str | None) -> dict:
    """读取 MedSAM3 训练 yaml 中的 lora 段；没给就用内置 v1 默认值。"""
    if not config_path:
        return dict(_DEFAULT_LORA_CFG)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    lora = cfg.get("lora", {})
    merged = dict(_DEFAULT_LORA_CFG)
    merged.update({k: lora[k] for k in merged if k in lora})
    return merged


def build_medsam3(checkpoint: str, medsam3_repo: str, lora_weights: str,
                  lora_config: str | None, device: str):
    """构建 SAM3 基座 -> 注入 LoRA 结构 -> 加载 MedSAM3 LoRA 权重。"""
    repo = Path(medsam3_repo)
    if not (repo / "lora_layers.py").exists():
        raise FileNotFoundError(
            f"{repo} 下找不到 lora_layers.py，请确认 --medsam3-repo 指向 "
            f"克隆好的 https://github.com/Joey-S-Liu/MedSAM3 目录")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

    print(f"[1/4] 构建 SAM3 基座: {checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        bpe_path=str(Path(checkpoint).parent / "bpe_simple_vocab_16e6.txt.gz"),
        device=device,
        load_from_HF=False,
        eval_mode=True,
    )

    cfg = load_lora_cfg(lora_config)
    print(f"[2/4] 注入 LoRA 结构 rank={cfg['rank']} alpha={cfg['alpha']}")
    lora_cfg = LoRAConfig(
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=0.0,
        target_modules=cfg["target_modules"],
        apply_to_vision_encoder=cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_cfg)

    print(f"[3/4] 加载 MedSAM3 LoRA 权重: {lora_weights}")
    load_lora_weights(model, lora_weights)

    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# 单图预测
# ──────────────────────────────────────────────

def predict_text_only(processor: Sam3Processor, image: Image.Image,
                      text_prompt: str, mask_threshold: float,
                      keep_components: str):
    """纯文本模式：所有过阈值实例取并集。返回 (union mask, meta)。"""
    W, H = image.size
    state = processor.set_image(image)
    output = processor.set_text_prompt(text_prompt, state)

    masks_logits = to_numpy(output.get("masks_logits"))
    scores = to_numpy(output.get("scores"))

    union = np.zeros((H, W), dtype=np.uint8)
    inst_scores = []
    if masks_logits is not None and len(masks_logits) > 0:
        masks_logits = squeeze_masks(masks_logits)
        probs = _sigmoid(masks_logits)   # masks_logits 是 logit，先转概率再阈值
        for i in range(len(masks_logits)):
            m = (probs[i] > mask_threshold).astype(np.uint8)
            m = postprocess_mask(m, ref_box=None,
                                 keep_components=("largest" if keep_components == "overlap"
                                                  else keep_components))
            union = np.logical_or(union, m).astype(np.uint8)
            if scores is not None and len(scores) > i:
                inst_scores.append(round(float(scores[i]), 4))

    meta = dict(mode="text_only", text_prompt=text_prompt,
                n_candidates=int(0 if masks_logits is None else len(masks_logits)),
                score=(max(inst_scores) if inst_scores else None),
                instance_scores=inst_scores)
    return union, meta


def predict_with_boxes(processor: Sam3Processor, image: Image.Image,
                       boxes_xyxy, text_prompt: str, expand_ratio: float,
                       mask_threshold: float, min_iou: float,
                       keep_components: str):
    """文本 + CAM 框模式：框作为概念的正样例锚点，逐框选候选取并集。"""
    W, H = image.size
    state = processor.set_image(image)
    union = np.zeros((H, W), dtype=np.uint8)
    meta_boxes = []

    for box in boxes_xyxy:
        processor.reset_all_prompts(state)
        ebox = expand_box(box, W, H, expand_ratio)
        processor.set_text_prompt(text_prompt, state)
        output = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(ebox, W, H), label=True, state=state)

        mask_prob, info = select_candidate(
            output.get("masks_logits"), output.get("boxes"),
            output.get("scores"), ebox, min_iou=min_iou,
            masks_bin=output.get("masks"))
        info["prompt_box_xyxy"] = [round(v, 1) for v in ebox]
        meta_boxes.append(info)
        if mask_prob is None:
            continue
        mask = (mask_prob > mask_threshold).astype(np.uint8)
        mask = postprocess_mask(mask, ref_box=ebox, keep_components=keep_components)
        union = np.logical_or(union, mask).astype(np.uint8)

    scores = [b["score"] for b in meta_boxes if b.get("score") is not None]
    meta = dict(mode="text+box", text_prompt=text_prompt,
                score=(max(scores) if scores else None), boxes=meta_boxes)
    return union, meta


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(args) -> None:
    model = build_medsam3(args.checkpoint, args.medsam3_repo,
                          args.lora_weights, args.lora_config, args.device)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.conf_threshold)

    pairs = collect_pairs(args.image, args.boxes)
    mode = "text+box" if args.boxes else "text_only"
    print(f"[4/4] 推理  mode={mode}  prompt={args.text_prompt!r}  "
          f"conf>{args.conf_threshold}  mask>{args.mask_threshold}  共 {len(pairs)} 张")

    out_root = Path(args.output)
    for image_path, json_path in tqdm(pairs, desc="MedSAM3"):
        boxes = []
        if json_path is not None:
            boxes, size = load_boxes_json(json_path)
            check_size_consistency(image_path, size, warn_fn=tqdm.write)

        try:
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode():
                if boxes:
                    mask, meta = predict_with_boxes(
                        processor, image, boxes, args.text_prompt,
                        args.expand_ratio, args.mask_threshold,
                        args.min_iou, args.keep_components)
                else:
                    mask, meta = predict_text_only(
                        processor, image, args.text_prompt,
                        args.mask_threshold, args.keep_components)
        except Exception as e:
            tqdm.write(f"  {image_path.name} 推理异常: {e}")
            continue

        meta["image"] = str(image_path)
        save_results(out_root, image_path.stem, image, mask, boxes,
                     meta=meta, save_overlay_flag=not args.no_overlay)

    print(f"完成 -> {out_root}")
    print(f"建议质检: python qa_sam3.py --mask {out_root}/mask "
          + (f"--boxes {args.boxes} " if args.boxes else "")
          + f"--image {args.image} --meta {out_root}/meta --output {out_root}/qa_report.csv")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("MedSAM3 内镜分割（医学概念文本提示）")
    ap.add_argument("--image", required=True, help="内镜原图：单张图片或图片文件夹")
    ap.add_argument("--boxes", default=None,
                    help="（可选）CAM bbox JSON：文件或 boxes/ 目录；给了走 文本+框 模式")
    ap.add_argument("--output", default="medsam3_out", help="输出根目录")
    ap.add_argument("--checkpoint", default="/data0/yzhen/data/sam3_service/ckpt/sam3.pt",
                    help="SAM3 基座权重（与 endoscope_sam3.py 同一个）")
    ap.add_argument("--medsam3-repo", required=True,
                    help="MedSAM3 仓库克隆目录（提供 lora_layers.py）")
    ap.add_argument("--lora-weights", required=True,
                    help="MedSAM3 LoRA 权重 best_lora_weights.pt 路径")
    ap.add_argument("--lora-config", default=None,
                    help="（可选）MedSAM3 configs/full_lora_config.yaml；不给用内置 v1 默认")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-prompt", default="colon polyp",
                    help='医学概念词（英文），如 "colon polyp" / "polyp" / "lesion"')
    ap.add_argument("--expand-ratio", type=float, default=0.12,
                    help="（文本+框模式）bbox 外扩比例")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="实例置信度阈值；MedSAM3 官方建议按任务在 0.5~0.8 之间调")
    ap.add_argument("--mask-threshold", type=float, default=0.5,
                    help="mask 概率二值化阈值")
    ap.add_argument("--min-iou", type=float, default=0.1,
                    help="（文本+框模式）候选框与提示框最小 IoU")
    ap.add_argument("--keep-components", default="overlap",
                    choices=["all", "largest", "overlap"])
    ap.add_argument("--no-overlay", action="store_true", help="不存 overlay 检查图")
    run(ap.parse_args())
