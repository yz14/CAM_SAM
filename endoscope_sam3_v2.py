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
                     min_iou: float = 0.1, min_score: float = 0.0,
                     min_coverage: float = 0.0, min_precision: float = 0.5,
                     min_mask_frac: float = 5e-4, max_mask_frac: float = 0.6,
                     masks_bin=None, debug: bool = False):
    """从 SAM3 输出的候选实例中选一个，返回 (mask (H,W) uint8∈{0,1}, info)。

    **以 v1（endoscope_sam3.py，确认能分出目标）为基准**：用模型自带的二值
    ``output["masks"]``（logit>0，下同），在「框内」候选里取 ``argmax(score)``。
    v1 只是 argmax(score)、假阳多；这里在它前面加一层**空间过滤**，把跑到框外、
    覆盖整图、或空的候选先剔掉，只保留落在 CAM 框里的候选，再按 score 选——
    既保留 v1 的找目标能力，又压掉框外/整图假阳。

    每个候选先算：
      - area      = |mask| / 图像面积
      - precision = |mask ∩ 提示框| / |mask|     —— mask 有多少落在框内
      - coverage  = |mask ∩ 提示框| / |提示框|   —— mask 填满了多少 CAM 框
    过滤条件（任一不满足即丢弃）：
      - min_mask_frac ≤ area ≤ max_mask_frac     —— 丢空 mask、丢覆盖整图的背景块
      - precision ≥ min_precision                —— 丢框外飞溅/整图（它们 precision 很低）
      - coverage  ≥ min_coverage（默认 0）       —— 可选，进一步要求填满框
      - IoU(候选框, 提示框) ≥ min_iou            —— 廉价空间初筛
    幸存候选（已过 precision/area 门槛 = 框内、非空、不覆盖整图）里取
    **argmax(coverage)**（填满 CAM 框最多 = 完整目标），score 作同分裂项——
    因为检测分(score)与 mask 大小不挂钩，纯 argmax(score) 会选中分高但极小的碎块。
    若没有候选通过 precision 门槛，则放宽到「非空 + IoU + 不覆盖整图」再
    argmax(score)（与 v1 一致）兜底，保证「有框尽量出目标」，不至于整图空白。

    ⚠️ 关键：用 ``masks_bin``(=output["masks"]) 作权威 mask，**不要**用
    ``masks_logits > 阈值``——诊断显示二者差异极大（masks_logits>0.5 常把真实
    mask 侵蚀到近乎空），v1 用的就是 ``masks``。masks_bin 缺失时才退回 logit>0。
    """
    masks_logits = to_numpy(masks_logits)
    pred_boxes = to_numpy(pred_boxes)
    scores = to_numpy(scores)
    masks_bin = to_numpy(masks_bin)
    info = dict(n_candidates=0, score=None, iou=None,
                coverage=None, precision=None, area=None, picked=None)

    # 权威 mask：优先模型自带二值 masks；否则退回 masks_logits>0（logit 阈值 0）
    if masks_bin is not None and len(masks_bin) > 0:
        M = squeeze_masks(masks_bin)
    elif masks_logits is not None and len(masks_logits) > 0:
        M = (squeeze_masks(masks_logits) > 0).astype(np.uint8)
    else:
        return None, info

    n, H, W = M.shape
    info["n_candidates"] = int(n)

    # 提示框像素范围（裁剪到图内）
    x0, y0, x1, y1 = prompt_box_xyxy
    ix0, iy0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    ix1, iy1 = min(int(round(x1)), W), min(int(round(y1)), H)
    box_area = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    img_area = float(H * W)

    rows = []   # (i, score, iou, area_frac, coverage, precision, passed)
    survivors, fallback = [], []
    for i in range(n):
        bm = (M[i] > 0).astype(np.uint8)
        area = int(bm.sum())
        area_frac = area / img_area
        s = float(scores[i]) if scores is not None and len(scores) > i else 0.0
        iou = box_iou_xyxy(pred_boxes[i].tolist(), prompt_box_xyxy) \
            if pred_boxes is not None and len(pred_boxes) > i else 0.0
        inter = int(bm[iy0:iy1, ix0:ix1].sum()) if (box_area > 0 and area > 0) else 0
        coverage = inter / box_area if box_area > 0 else 0.0
        precision = inter / area if area > 0 else 0.0

        non_empty = area_frac >= min_mask_frac
        passed = (non_empty and area_frac <= max_mask_frac
                  and precision >= min_precision and coverage >= min_coverage
                  and iou >= min_iou)
        if passed:
            survivors.append((i, s, iou, area_frac, coverage, precision))
        elif non_empty and iou >= min_iou and area_frac <= max_mask_frac:
            # 兜底池：放宽 precision/coverage，但仍排除空 mask 与覆盖整图的背景块
            fallback.append((i, s, iou, area_frac, coverage, precision))
        if debug:
            rows.append((i, round(s, 3), round(iou, 2), round(area_frac, 4),
                         round(coverage, 3), round(precision, 3), passed))

    pool = survivors if survivors else fallback
    info["used_fallback"] = (not survivors) and bool(fallback)
    if debug:
        info["debug_rows"] = sorted(rows, key=lambda r: -r[1])[:12]
    if not pool:
        return None, info

    # 排序键：
    #   幸存池（已过 precision≥min_precision + area≤max_mask_frac，即「框内、非空、
    #   不覆盖整图」）按 coverage（填满 CAM 框最多）选——检测分(score)与 mask 大小
    #   不挂钩，纯 argmax(score) 会选中分高但极小的碎块（debug 实测 #92），故改用
    #   coverage 选「填满框的完整目标」；因 precision 门槛已剔掉覆盖整图/框外的候选，
    #   这里用 coverage 不会再整图涂满。score 作同分裂项。
    #   兜底池（precision 没过门槛、mask 大半在框外）改回 argmax(score)（与 v1 一致），
    #   此时按 coverage 反而可能放大框外飞溅。
    if survivors:
        i, s, iou, area_frac, coverage, precision = max(
            pool, key=lambda r: (r[4], r[1]))
    else:
        i, s, iou, area_frac, coverage, precision = max(
            pool, key=lambda r: (r[1], r[4]))
    info.update(score=round(s, 4), iou=round(iou, 4),
                coverage=round(coverage, 4), precision=round(precision, 4),
                area=round(area_frac, 4), picked=int(i))
    if s < min_score:
        info["dropped_by_min_score"] = True
        return None, info
    return (M[i] > 0).astype(np.uint8), info


# ──────────────────────────────────────────────
# 单图预测
# ──────────────────────────────────────────────

def _print_debug(name, bi, box, ebox, info):
    """--debug：打印一个框的候选表与选中结果（用模型自带 masks 统计）。"""
    print(f"  [{name} 框{bi}] box={[round(v,1) for v in box]} "
          f"-> ebox={[round(v,1) for v in ebox]}  候选n={info.get('n_candidates')}")
    rows = info.get("debug_rows") or []
    if rows:
        print("      idx  score   IoU   area  coverage  precision  通过")
        for (i, s, iou, area, cov, prec, passed) in rows:
            print(f"      {i:3d}  {s:6.3f}  {iou:4.2f}  {area:6.4f}  "
                  f"{cov:7.3f}  {prec:8.3f}   {'√' if passed else '×'}")
    fb = "（兜底池）" if info.get("used_fallback") else ""
    print(f"      选中 -> #{info.get('picked')}{fb}  score={info.get('score')} "
          f"area={info.get('area')} coverage={info.get('coverage')} "
          f"precision={info.get('precision')}")


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
    min_precision: float,
    max_mask_frac: float,
    keep_components: str,
    debug: bool = False,
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

    for bi, box in enumerate(boxes_xyxy):
        processor.reset_all_prompts(state)
        ebox = expand_box(box, W, H, expand_ratio)

        if text_prompt:
            processor.set_text_prompt(text_prompt, state)
        output = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(ebox, W, H), label=True, state=state)

        mask_sel, info = select_candidate(
            output.get("masks_logits"), output.get("boxes"),
            output.get("scores"), ebox, min_iou=min_iou, min_score=min_score,
            min_coverage=min_coverage, min_precision=min_precision,
            max_mask_frac=max_mask_frac,
            masks_bin=output.get("masks"), debug=debug)
        info["prompt_box_xyxy"] = [round(v, 1) for v in ebox]
        if debug:
            _print_debug(image_path.name, bi, box, ebox, info)
        meta_boxes.append({k: v for k, v in info.items() if k != "debug_rows"})
        if mask_sel is None:
            continue

        mask = postprocess_mask(mask_sel, ref_box=ebox, keep_components=keep_components)
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
          f"选法=框内候选 argmax(score)  min_iou>{args.min_iou}  "
          f"min_prec>{args.min_precision}  min_cov>{args.min_coverage}"
          f"{'  [debug]' if args.debug else ''}")
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
                    min_precision=args.min_precision,
                    max_mask_frac=args.max_mask_frac,
                    keep_components=args.keep_components,
                    debug=args.debug,
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
                    help="（仅在模型未返回二值 masks、退回 logit 时生效）mask 二值化阈值；"
                         "默认下直接用模型自带的 output['masks']（与 v1 一致）")
    ap.add_argument("--min-iou", type=float, default=0.1,
                    help="候选框与提示框最小 IoU（廉价空间初筛）")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="选中 mask 对提示框的最小覆盖率（|mask∩框|/|框|）；"
                         "默认 0；要求填满 CAM 框可调 0.1~0.3")
    ap.add_argument("--min-precision", type=float, default=0.5,
                    help="候选 mask 落在提示框内的最小比例（|mask∩框|/|mask|）；"
                         "默认 0.5，主要用来踢掉框外飞溅/覆盖整图的候选；"
                         "目标被误丢时调低（0.3），假阳多时调高（0.7）")
    ap.add_argument("--max-mask-frac", type=float, default=0.6,
                    help="候选 mask 面积占全图上限（超过视为覆盖整图的背景块，丢弃）")
    ap.add_argument("--debug", action="store_true",
                    help="打印每个框的候选表（score/IoU/area/coverage/precision/是否通过）与选中结果")
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
