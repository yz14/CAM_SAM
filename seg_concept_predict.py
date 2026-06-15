"""SAM3 概念分割的「文本 / 文本+框」推理逻辑（与具体权重无关）。

抽出这一层是因为 SAM3 家族里只要 ``Sam3Processor`` 接口一致（set_image /
set_text_prompt / add_geometric_prompt / reset_all_prompts，输出 dict 带
``masks_logits/masks/boxes/scores``），推理流程就完全相同，差异只在「怎么把
权重 build 出来」：

- ``endoscope_medsam3.py``     —— 通用 SAM3 基座 + Joey-S-Liu/MedSAM3 的 LoRA 增量
- ``endoscope_medical_sam3.py`` —— AIM-Research-Lab/Medical-SAM3 的全量微调权重

两者都把构建好的 processor 交给这里的 ``predict_text_only`` /
``predict_with_boxes``，避免两份脚本各抄一遍推理代码（见 TODO.md「代码复用」）。
候选选择沿用 ``endoscope_sam3_v2.select_candidate()``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from endoscope_sam3_v2 import _print_debug, _sigmoid, select_candidate
from seg_common import (
    expand_box,
    postprocess_mask,
    squeeze_masks,
    to_numpy,
    xyxy_to_norm_cxcywh,
)

if TYPE_CHECKING:  # 仅类型标注用，避免在没装好 sam3 时强制导入
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor


def predict_text_only(processor: "Sam3Processor", image: "Image.Image",
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


def predict_with_boxes(processor: "Sam3Processor", image: "Image.Image",
                       boxes_xyxy, text_prompt: str, expand_ratio: float,
                       min_iou: float, min_precision: float,
                       max_mask_frac: float, keep_components: str,
                       name: str = "", debug: bool = False):
    """文本 + CAM 框模式：框作为概念的正样例锚点，逐框选候选取并集。

    选候选与 endoscope_sam3_v2.py 一致：用模型自带二值 ``output["masks"]``，
    在「框内、非空、不覆盖整图」的候选里 ``argmax(coverage × precision)``。
    """
    W, H = image.size
    state = processor.set_image(image)
    union = np.zeros((H, W), dtype=np.uint8)
    meta_boxes = []

    for bi, box in enumerate(boxes_xyxy):
        processor.reset_all_prompts(state)
        ebox = expand_box(box, W, H, expand_ratio)
        processor.set_text_prompt(text_prompt, state)
        output = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(ebox, W, H), label=True, state=state)

        mask_sel, info = select_candidate(
            output.get("masks_logits"), output.get("boxes"),
            output.get("scores"), ebox, min_iou=min_iou,
            min_precision=min_precision, max_mask_frac=max_mask_frac,
            masks_bin=output.get("masks"), debug=debug)
        info["prompt_box_xyxy"] = [round(v, 1) for v in ebox]
        if debug:
            _print_debug(name, bi, box, ebox, info)
        meta_boxes.append({k: v for k, v in info.items() if k != "debug_rows"})
        if mask_sel is None:
            continue
        mask = postprocess_mask(mask_sel, ref_box=ebox, keep_components=keep_components)
        union = np.logical_or(union, mask).astype(np.uint8)

    scores = [b["score"] for b in meta_boxes if b.get("score") is not None]
    meta = dict(mode="text+box", text_prompt=text_prompt,
                score=(max(scores) if scores else None), boxes=meta_boxes)
    return union, meta
