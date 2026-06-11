"""SAM3 v2 诊断脚本：把"为什么选错/分不出目标"摊开看。
================================================================

背景：endoscope_sam3.py（纯 box）能分出目标但假阳多；endoscope_sam3_v2.py
（text "polyp" + box）反而分不出目标、假阳还跑到框外。本脚本对**单张/少量**图，
逐个 CAM 框打印所有候选实例的几何与分数，定位 select_candidate 选错的原因：

每个框会打印：
  - 提示框 ebox（像素 xyxy）与图像尺寸（核对坐标系是否一致）
  - 候选总数 n
  - 候选列表（按 score 排序）：pred_box(像素 xyxy 原值)、score、
    IoU(候选框, 提示框)、mask 面积占比、mask 外接框
  - 四种"选法"各自会选中谁，便于对比：
      * argmax(score)              —— v1 的选法
      * argmax(score × IoU)        —— v2 旧 select_candidate 的选法
      * argmax(IoU)                —— 纯空间最近
      * argmax(cov×prec)           —— v2 新 select_candidate（mask∩框契合度）★
overlay 用**新选法**（与修复后的 endoscope_sam3_v2.py 一致）画出，方便直接目检
目标是否被正确分出。分别在 text 模式与 no-text 模式各跑一遍。

用法::

    python debug_sam3_v2.py \\
        --image  /data0/yzhen/projects/CAM/test_data/imgs \\
        --boxes  /data0/yzhen/projects/CAM/test_data/bbox \\
        --output /data0/yzhen/projects/CAM/sam3_v2_debug \\
        --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt \\
        --text-prompt "polyp"

把打印输出 + output/ 里的 overlay 图发回来即可。
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

from endoscope_sam3_v2 import select_candidate
from seg_common import (
    box_iou_xyxy,
    collect_pairs,
    expand_box,
    load_boxes_json,
    mask_to_xyxy,
    postprocess_mask,
    save_overlay,
    squeeze_masks,
    to_numpy,
    xyxy_to_norm_cxcywh,
)


def _cov_prec(mask_bin, ebox, W, H):
    """mask 与提示框的 coverage(|m∩box|/|box|) 与 precision(|m∩box|/|m|)。"""
    x0, y0, x1, y1 = ebox
    ix0, iy0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    ix1, iy1 = min(int(round(x1)), W), min(int(round(y1)), H)
    box_area = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area = int(mask_bin.sum())
    if area == 0 or box_area == 0:
        return 0.0, 0.0
    inter = int(mask_bin[iy0:iy1, ix0:ix1].sum())
    return inter / box_area, inter / area


def _dump_candidates(output, ebox, W, H, top_k=12):
    """打印一次推理的全部候选，并返回 (masks_prob, boxes, scores) numpy。"""
    masks = to_numpy(output.get("masks_logits"))
    boxes = to_numpy(output.get("boxes"))
    scores = to_numpy(output.get("scores"))

    n = 0 if masks is None else len(masks)
    print(f"      候选总数 n={n}")
    if n == 0:
        print("      [!] 模型没有返回任何候选实例")
        return masks, boxes, scores

    masks = squeeze_masks(masks)
    # 坐标系自检：pred_box 的数值量级 vs 图像尺寸
    bmax = float(np.max(boxes)) if boxes is not None and len(boxes) else 0.0
    print(f"      坐标自检: 图像 WxH={W}x{H}, pred_box 最大值={bmax:.3f} "
          f"({'像素量级√' if bmax > 2.0 else '疑似归一化[0,1]！IoU 会全=0'})")

    rows = []
    for i in range(n):
        pb = boxes[i].tolist() if boxes is not None else [0, 0, 0, 0]
        s = float(scores[i]) if scores is not None and len(scores) > i else 0.0
        iou = box_iou_xyxy(pb, ebox)
        bm = (masks[i] > 0.5).astype(np.uint8)
        area = float(bm.mean())  # mask 面积占全图比例
        cov, prec = _cov_prec(bm, ebox, W, H)
        mbox = mask_to_xyxy(bm)
        rows.append((i, s, iou, cov * prec, area, pb, mbox, cov, prec))

    # 按 score 降序打印
    print("        idx  score    IoU   cov*prec  maskArea  pred_box(px)            mask_bbox(px)")
    for (i, s, iou, key, area, pb, mbox, cov, prec) in sorted(rows, key=lambda r: -r[1])[:top_k]:
        pb_s = "[" + ",".join(f"{v:6.0f}" for v in pb) + "]"
        mb_s = ("[" + ",".join(f"{v:6.0f}" for v in mbox) + "]") if mbox else "None"
        print(f"        {i:3d}  {s:6.3f}  {iou:5.2f}  {key:8.3f}  {area:7.4f}  {pb_s}  {mb_s}")

    def _argmax(keyfn):
        best, bi = -1.0, None
        for r in rows:
            v = keyfn(*r)
            if v > best:
                best, bi = v, r[0]
        return bi
    i_score = _argmax(lambda i, s, iou, key, *a: s)
    i_mix = _argmax(lambda i, s, iou, key, *a: s * max(iou, 1e-3))
    i_iou = _argmax(lambda i, s, iou, key, *a: iou)
    i_cp = _argmax(lambda i, s, iou, key, *a: key)
    print(f"      选法对比: argmax(score)=#{i_score}  "
          f"argmax(score*IoU)=#{i_mix}  argmax(IoU)=#{i_iou}  "
          f"argmax(cov*prec)=#{i_cp} ★新选法")
    return masks, boxes, scores


def _run_mode(processor, image, boxes_xyxy, text_prompt, expand_ratio, W, H,
              mask_threshold, out_overlay):
    """跑一种模式（text_prompt=None 表示纯 box），逐框 dump，存一张 overlay。"""
    state = processor.set_image(image)
    union = np.zeros((H, W), dtype=np.uint8)
    for bi, box in enumerate(boxes_xyxy):
        processor.reset_all_prompts(state)
        ebox = expand_box(box, W, H, expand_ratio)
        print(f"    [框 {bi}] 原始box={[round(v,1) for v in box]} "
              f"-> 外扩ebox={[round(v,1) for v in ebox]}")
        if text_prompt:
            processor.set_text_prompt(text_prompt, state)
        output = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(ebox, W, H), label=True, state=state)
        masks, pboxes, scores = _dump_candidates(output, ebox, W, H)
        if masks is None or len(masks) == 0:
            continue
        # overlay 用**新** select_candidate（mask∩框契合度），与修复后 v2 一致
        mask_prob, info = select_candidate(
            output.get("masks_logits"), output.get("boxes"),
            output.get("scores"), ebox, masks_bin=output.get("masks"))
        print(f"      新选法选中: {info}")
        if mask_prob is not None:
            m = (mask_prob > mask_threshold).astype(np.uint8)
            m = postprocess_mask(m, ref_box=ebox, keep_components="overlap")
            union = np.logical_or(union, m).astype(np.uint8)
    save_overlay(image, union, boxes_xyxy, out_overlay)
    print(f"    overlay -> {out_overlay}")


def run(args):
    print(f"加载 SAM3: {args.checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        bpe_path=str(Path(args.checkpoint).parent / "bpe_simple_vocab_16e6.txt.gz"),
        device=args.device, load_from_HF=False, eval_mode=True)
    model.eval()
    # 内部阈值置 0：保留全部候选，诊断时不让模型提前删实例
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.0)

    pairs = collect_pairs(args.image, args.boxes)[: args.limit]
    print(f"诊断 {len(pairs)} 张\n")
    out_root = Path(args.output)

    for image_path, json_path in pairs:
        boxes, size = load_boxes_json(json_path)
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        print(f"==== {image_path.name}  (W={W},H={H}, json_size={size}, n_box={len(boxes)}) ====")
        if not boxes:
            print("  boxes 为空，跳过\n")
            continue
        with torch.inference_mode():
            print("  --- 模式 A: text+box  (text_prompt=%r) ---" % args.text_prompt)
            _run_mode(processor, image, boxes, args.text_prompt, args.expand_ratio,
                      W, H, args.mask_threshold,
                      out_root / "overlay_text" / f"{image_path.stem}.png")
            print("  --- 模式 B: no-text (纯 box exemplar) ---")
            _run_mode(processor, image, boxes, None, args.expand_ratio,
                      W, H, args.mask_threshold,
                      out_root / "overlay_notext" / f"{image_path.stem}.png")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("SAM3 v2 候选诊断")
    ap.add_argument("--image", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--output", default="sam3_v2_debug")
    ap.add_argument("--checkpoint", default="/data0/yzhen/data/sam3_service/ckpt/sam3.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-prompt", default="polyp")
    ap.add_argument("--expand-ratio", type=float, default=0.12)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=2, help="最多诊断几张（默认 2）")
    run(ap.parse_args())
