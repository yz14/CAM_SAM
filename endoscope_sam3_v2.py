"""SAM3 内镜分割预测 v2（概念提示 + box exemplar 联合版）
======================================================

相比 endoscope_sam3.py（纯 box prompt）的改进点（详见
docs/sam3_polyp_improvement.md 的分析）：

1. **文本概念 + 框 exemplar 联合提示**：SAM3 是「概念分割」模型，
   纯 box prompt 时文本会退化为占位词 "visual"，模型只能猜框内是什么。
   先 ``set_text_prompt("polyp")`` 注入语义，再把 CAM 框作为正样例
   （exemplar）喂入，框从「唯一信息源」变成「概念的空间锚点」。
2. **bbox 自适应外扩**：CAM 框普遍偏紧（高响应只覆盖病灶中心），
   默认外扩 12%，把病灶边缘留在提示框内。
3. **候选选择策略**：不再盲取全图最高分实例（概念模式下可能选到
   框外的其它息肉），改为 score × IoU(候选框, 提示框) 混合排序。
   **置信度过滤放在选完之后**：SAM3 内部 confidence_threshold 置 0、
   保留全部候选，避免「polyp」这类未见医学概念的低匹配分把目标实例
   在出 mask 前就整批删空（这正是 v2 之前「连目标都分割不出来」的根因）。
4. **mask 二值化阈值可调**：直接用 ``masks_logits``（概率图），
   欠分割时调低 --mask-threshold 即可外扩 mask，无需重跑模型。
5. **mask 后处理**：只保留与提示框相交的连通域 + 补洞 + 闭运算，
   消掉「碎成多块 / 框外飞溅 / 高光被抠洞」三类常见坏例。
6. **置信度后置过滤 + meta 落盘**：--conf-threshold 是选中候选的后置得分
   下限（默认 0 不过滤），每图写 meta/<stem>.json（score/iou/候选数），
   qa_sam3.py --meta 直接可用。
7. **每图只编码一次图像**：多框时复用 image embedding（reset 提示而非
   重跑 backbone），推理显著提速。

用法
----
单张::

    python endoscope_sam3_v2.py \\
        --image  /path/img.jpg \\
        --boxes  /path/pku1_compare/boxes/img.json \\
        --output /path/sam3_v2_out \\
        --checkpoint /data0/.../ckpt/sam3.pt \\
        --text-prompt "polyp"

批量（图片文件夹 + boxes 目录，按 stem 配对）::

    python endoscope_sam3_v2.py \\
        --image  /data0/yzhen/data/endoscope_pku1/1 \\
        --boxes  /data0/yzhen/projects/CAM/pku1/1/boxes \\
        --output ./sam3_v2_out/1 \\
        --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt

调参速查（结果不理想时按顺序试）
--------------------------------
- 漏分割/mask 偏小：--expand-ratio 0.2、--mask-threshold 0.35
- 过分割/吞背景：--expand-ratio 0.05、--mask-threshold 0.6
- 假阳太多：--conf-threshold 0.3~0.5（后置过滤掉低分选中候选）、--min-iou 0.3
- 选错目标：--min-iou 0.5（强制候选框与提示框重叠）
- 想对比旧行为：--no-text（退回纯 box prompt，仅保留外扩与后处理）

注意：--conf-threshold 现在是「选中候选」的后置得分下限（默认 0 = 有框必出
目标），不再直接喂给 SAM3 内部阈值；内部阈值由 --model-conf-threshold 控制，
默认 0（保留全部候选）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from seg_common import (
    box_iou_xyxy,
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


# ──────────────────────────────────────────────
# 候选选择
# ──────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def select_candidate(masks_logits, pred_boxes, scores, prompt_box_xyxy,
                     min_iou: float = 0.0, min_score: float = 0.0,
                     min_coverage: float = 0.0,
                     min_mask_frac: float = 1e-4, masks_bin=None):
    """从 SAM3 输出的候选实例中选最匹配提示框的一个。

    返回 (mask_prob (H,W) float ∈ [0,1], info dict)；无合格候选返回 (None, info)。

    **按「mask 与提示框的契合度」选，而不是按检测分 score。**
    诊断发现：概念模式下 SAM3 会返回上百个候选，**检测分和 mask 质量并不挂钩**
    —— 分最高的候选常常「框很准但 mask 是空的」（score 0.93 / box-IoU 0.93 但
    maskArea≈0），真正的息肉 mask 反而在低分候选里（score~0.1 但 maskArea 0.12）。
    旧的 score×IoU 选法必然选中那个空 mask 检测，导致整图分不出目标。

    现在的排序键 = coverage × precision：
      - coverage  = |mask ∩ 提示框| / |提示框|     —— 这个 mask 填满了多少提示框
      - precision = |mask ∩ 提示框| / |mask|       —— mask 有多少落在框内（惩罚框外飞溅）
    既选中「填满 CAM 框」的目标，又自动压掉跑到框外的假阳。空/退化 mask 由
    min_mask_frac 直接丢弃。min_iou 用候选框与提示框做一道廉价空间初筛。

    mask 概率 = sigmoid(masks_logits)：SAM3 的 ``masks_logits`` 是**原始 logit**，
    旧版当成概率直接 >0.5 会把弱边界侵蚀掉（v1 用的是模型自带二值 ``masks``，即
    logit>0）。这里统一转成概率，配 --mask-threshold（默认 0.5 = logit 0）二值化。
    若 SAM3 输出里带二值 ``masks``，优先用它来做候选打分（与模型自身判定一致）。

    min_score 是对「选中候选」检测分的下限（默认 0 不过滤）；注意目标实例的
    score 可能很低（~0.1），**调高 min_score 有丢目标风险**，抑制假阳优先用
    --min-iou / --min-coverage（见调参表）。
    """
    masks_logits = to_numpy(masks_logits)
    pred_boxes = to_numpy(pred_boxes)
    scores = to_numpy(scores)
    masks_bin = to_numpy(masks_bin)
    info = dict(n_candidates=0, score=None, iou=None, coverage=None, precision=None)
    if masks_logits is None or len(masks_logits) == 0:
        return None, info

    masks_logits = squeeze_masks(masks_logits)
    probs = _sigmoid(masks_logits)
    if masks_bin is not None:
        masks_bin = squeeze_masks(masks_bin)
    n, H, W = probs.shape
    info["n_candidates"] = int(n)

    # 提示框像素范围（裁剪到图内）
    x0, y0, x1, y1 = prompt_box_xyxy
    ix0, iy0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    ix1, iy1 = min(int(round(x1)), W), min(int(round(y1)), H)
    box_area = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)

    best, best_key = None, -1.0
    for i in range(n):
        # 候选二值 mask：优先模型自带 masks（logit>0），否则 prob>0.5
        if masks_bin is not None and len(masks_bin) > i:
            bm = (masks_bin[i] > 0).astype(np.uint8)
        else:
            bm = (probs[i] > 0.5).astype(np.uint8)
        area = int(bm.sum())
        if area < min_mask_frac * bm.size:   # 丢弃空/退化 mask（旧版被它坑了）
            continue
        iou = box_iou_xyxy(pred_boxes[i].tolist(), prompt_box_xyxy) \
            if pred_boxes is not None and len(pred_boxes) > i else 0.0
        if iou < min_iou:
            continue
        inter = int(bm[iy0:iy1, ix0:ix1].sum()) if box_area > 0 else 0
        coverage = inter / box_area if box_area > 0 else 0.0
        precision = inter / area if area > 0 else 0.0
        if coverage < min_coverage:          # 框被覆盖太少，多半是框外飞溅，丢
            continue
        s = float(scores[i]) if scores is not None and len(scores) > i else 0.0
        key = coverage * precision
        # score 仅作极小权重的同分裂项，避免它主导（目标分可能很低）
        key_tb = key + 1e-4 * s
        if key_tb > best_key:
            best_key = key_tb
            best = (i, s, iou, coverage, precision)

    if best is None:
        return None, info
    i, s, iou, coverage, precision = best
    info.update(score=round(s, 4), iou=round(iou, 4),
                coverage=round(coverage, 4), precision=round(precision, 4))
    if s < min_score:
        info["dropped_by_min_score"] = True
        return None, info
    return probs[i].astype(np.float32), info


# ──────────────────────────────────────────────
# 单图预测
# ──────────────────────────────────────────────

def predict_one(
    processor: Sam3Processor,
    image_path: Path,
    boxes_xyxy,
    text_prompt: str | None,
    expand_ratio: float,
    mask_threshold: float,
    min_iou: float,
    min_score: float,
    min_coverage: float,
    keep_components: str,
):
    """一张图、若干 CAM 框 -> (PIL 图, union mask, meta)。

    图像只编码一次（set_image），多框间用 reset_all_prompts 复用
    image embedding，只重算提示与解码头。
    """
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    state = processor.set_image(image)
    union = np.zeros((H, W), dtype=np.uint8)
    meta_boxes = []

    for box in boxes_xyxy:
        processor.reset_all_prompts(state)
        ebox = expand_box(box, W, H, expand_ratio)

        if text_prompt:
            processor.set_text_prompt(text_prompt, state)
        output = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(ebox, W, H), label=True, state=state)

        mask_prob, info = select_candidate(
            output.get("masks_logits"), output.get("boxes"),
            output.get("scores"), ebox, min_iou=min_iou, min_score=min_score,
            min_coverage=min_coverage, masks_bin=output.get("masks"))
        info["prompt_box_xyxy"] = [round(v, 1) for v in ebox]
        meta_boxes.append(info)
        if mask_prob is None:
            continue

        mask = (mask_prob > mask_threshold).astype(np.uint8)
        mask = postprocess_mask(mask, ref_box=ebox, keep_components=keep_components)
        union = np.logical_or(union, mask).astype(np.uint8)

    scores = [b["score"] for b in meta_boxes if b.get("score") is not None]
    meta = dict(
        image=str(image_path),
        text_prompt=text_prompt,
        score=(max(scores) if scores else None),  # qa_sam3.py 读这个字段
        boxes=meta_boxes,
    )
    return image, union, meta


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(args) -> None:
    print(f"[1/3] 加载 SAM3: {args.checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        bpe_path=str(Path(args.checkpoint).parent / "bpe_simple_vocab_16e6.txt.gz"),
        device=args.device,
        load_from_HF=False,
        eval_mode=True,
    )
    model.eval()
    # 关键：内部 confidence_threshold 置 0，让 SAM3 保留所有候选实例（包括与
    # CAM 框重叠的目标），把置信度/空间过滤交给 select_candidate。否则「polyp」
    # 这类未见医学概念的匹配分偏低，内部阈值会在出 mask 前就把目标删空 -> 整图无输出。
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.model_conf_threshold)

    print("[2/3] 收集 (图片, boxes) 配对")
    pairs = collect_pairs(args.image, args.boxes)
    print(f"      配对成功 {len(pairs)} 张")

    out_root = Path(args.output)
    text_prompt = None if args.no_text else args.text_prompt

    print(f"[3/3] 推理  text_prompt={text_prompt!r}  expand={args.expand_ratio}  "
          f"选法=mask∩框coverage×precision  min_iou>{args.min_iou}  "
          f"min_cov>{args.min_coverage}  mask>{args.mask_threshold}")
    for image_path, json_path in tqdm(pairs, desc="SAM3-v2"):
        boxes, size = load_boxes_json(json_path)
        if not boxes:
            tqdm.write(f"  {image_path.name}: boxes 为空，跳过")
            continue
        check_size_consistency(image_path, size, warn_fn=tqdm.write)

        try:
            with torch.inference_mode():
                image, mask, meta = predict_one(
                    processor, image_path, boxes,
                    text_prompt=text_prompt,
                    expand_ratio=args.expand_ratio,
                    mask_threshold=args.mask_threshold,
                    min_iou=args.min_iou,
                    min_score=args.conf_threshold,
                    min_coverage=args.min_coverage,
                    keep_components=args.keep_components,
                )
        except Exception as e:
            tqdm.write(f"  {image_path.name} 推理异常: {e}")
            continue

        save_results(out_root, image_path.stem, image, mask, boxes,
                     meta=meta, save_overlay_flag=not args.no_overlay)

    print(f"完成 -> {out_root}")
    print(f"建议质检: python qa_sam3.py --mask {out_root}/mask --boxes {args.boxes} "
          f"--image {args.image} --meta {out_root}/meta --output {out_root}/qa_report.csv")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("SAM3 内镜分割 v2（概念 + box exemplar）")
    ap.add_argument("--image", required=True, help="内镜原图：单张图片或图片文件夹")
    ap.add_argument("--boxes", required=True, help="bbox JSON：单个文件或 boxes/ 目录")
    ap.add_argument("--output", default="sam3_v2_out", help="输出根目录")
    ap.add_argument("--checkpoint", default="/data0/yzhen/data/sam3_service/ckpt/sam3.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-prompt", default="polyp",
                    help='概念词（英文，CLIP 文本编码器），结肠息肉可试 "polyp" / "colon polyp"')
    ap.add_argument("--no-text", action="store_true",
                    help="关闭文本提示，退回纯 box prompt（对照旧版行为）")
    ap.add_argument("--expand-ratio", type=float, default=0.12,
                    help="bbox 按宽高比例外扩（0 关闭）；CAM 框偏紧时调大")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="选中候选的最低分（后置过滤，默认 0 = 有框必出目标）；"
                         "误检多时调高 0.3~0.5 抑制假阳。注意这是后置过滤，"
                         "不再喂给 SAM3 内部 —— 内部阈值过高会把未见概念的目标删空")
    ap.add_argument("--model-conf-threshold", type=float, default=0.0,
                    help="SAM3 内部 confidence_threshold；默认 0 保留全部候选，"
                         "由 select_candidate 做空间/置信过滤。一般不用改")
    ap.add_argument("--mask-threshold", type=float, default=0.5,
                    help="mask 二值化阈值（对 sigmoid(masks_logits) 的概率）；"
                         "默认 0.5 = logit>0（等价模型自带 masks）；mask 偏小调低 0.3")
    ap.add_argument("--min-iou", type=float, default=0.1,
                    help="候选框与提示框最小 IoU（廉价空间初筛）")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="选中 mask 对提示框的最小覆盖率（|mask∩框|/|框|）；"
                         "默认 0；假阳多时调 0.1~0.3 压框外飞溅")
    ap.add_argument("--keep-components", default="overlap",
                    choices=["all", "largest", "overlap"],
                    help="后处理保留哪些连通域")
    ap.add_argument("--no-overlay", action="store_true", help="不存 overlay 检查图")
    run(ap.parse_args())

# 服务器示例：
# python endoscope_sam3_v2.py \
#     --image /data0/yzhen/data/endoscope_pku1/1 \
#     --boxes /data0/yzhen/projects/CAM/pku1/1/boxes \
#     --output ./sam3_v2_out/1 \
#     --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt
