"""内镜分割脚本的公共模块（SAM3 / MedSAM3 共用）。

职责
----
1. 图片 <-> bbox JSON 的配对收集（单文件 / 目录批量，按 stem 匹配）
2. bbox 几何工具：xyxy(px) <-> 归一化 cxcywh、外扩、IoU
3. mask 后处理：连通域筛选、补洞
4. 结果落盘：mask / overlay / meta（meta 可直接喂给 qa_sam3.py 的 --meta）

被 endoscope_sam3.py / endoscope_sam3_v2.py / endoscope_medsam3.py 复用，
避免三份脚本各写一套 IO 与几何代码。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ──────────────────────────────────────────────
# bbox 几何
# ──────────────────────────────────────────────

def xyxy_to_norm_cxcywh(box, W: int, H: int) -> list[float]:
    """像素 [x0,y0,x1,y1] -> 归一化 [cx,cy,w,h]（SAM3 add_geometric_prompt 的格式）。"""
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    return [float(cx), float(cy), float(w), float(h)]


def expand_box(box, W: int, H: int, ratio: float) -> list[float]:
    """按宽高比例向四周外扩 bbox（ratio 可为 0 表示不外扩），并裁剪到图像边界。

    CAM 出的框往往偏紧、且可能只盖住病灶的高响应中心；适度外扩能把病灶
    边缘留在 prompt 框内，是 box-prompt 分割最便宜有效的改进之一。
    """
    x0, y0, x1, y1 = map(float, box)
    px, py = (x1 - x0) * ratio, (y1 - y0) * ratio
    return [max(0.0, x0 - px), max(0.0, y0 - py),
            min(float(W), x1 + px), min(float(H), y1 + py)]


def box_iou_xyxy(a, b) -> float:
    """两个像素 xyxy 框的 IoU。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, area_a + area_b - inter)


# ──────────────────────────────────────────────
# tensor / mask 工具
# ──────────────────────────────────────────────

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def squeeze_masks(masks: np.ndarray) -> np.ndarray:
    """(N,1,H,W) -> (N,H,W)；其它形状原样返回。"""
    if masks.ndim == 4 and masks.shape[1] == 1:
        return masks[:, 0]
    return masks


def mask_to_xyxy(mask: np.ndarray) -> list[float] | None:
    """二值 mask 的外接框（像素 xyxy）；空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def postprocess_mask(
    mask: np.ndarray,
    ref_box=None,
    keep_components: str = "overlap",  # "all" | "largest" | "overlap"
    fill_holes: bool = True,
    morph_close: int = 5,
) -> np.ndarray:
    """对二值 mask 做轻量后处理，抑制常见错误模式。

    - keep_components="largest"：只留最大连通域（病灶通常是单块）
    - keep_components="overlap"：只留与 ref_box 相交的连通域（去掉框外飞溅碎块）
    - fill_holes：补内部空洞（息肉表面高光常被抠成洞）
    - morph_close：闭运算平滑边缘锯齿；<=1 关闭
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    if morph_close and morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if keep_components != "all":
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        if keep_components == "largest" or ref_box is None:
            if n > 1:
                idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                keep[labels == idx] = 1
        else:  # overlap
            for i in range(1, n):
                x, y, w, h = stats[i, :4]
                if box_iou_xyxy((x, y, x + w, y + h), ref_box) > 0 or _box_contains_any(
                        ref_box, (x, y, x + w, y + h)):
                    keep[labels == i] = 1
            if keep.sum() == 0 and n > 1:  # 全部不相交时退化为最大连通域
                idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                keep[labels == idx] = 1
        m = keep

    if fill_holes:
        # 从边界 floodfill 背景，未被填到的 0 像素即内部空洞
        h, w = m.shape
        ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
        inv = (1 - m).astype(np.uint8)
        cv2.floodFill(inv, ff, (0, 0), 0)
        m = ((m + inv) > 0).astype(np.uint8)
    return m


def _box_contains_any(a, b) -> bool:
    """框 a 与 b 是否有任意重叠（含包含关系，IoU 可能为 0 的退化情形兜底）。"""
    return not (b[2] <= a[0] or b[0] >= a[2] or b[3] <= a[1] or b[1] >= a[3])


# ──────────────────────────────────────────────
# IO：配对收集 / JSON / 落盘
# ──────────────────────────────────────────────

def collect_pairs(image_arg: str, boxes_arg: str | None) -> list[tuple[Path, Path | None]]:
    """收集 (图片, bbox JSON or None) 配对。

    - 图片为目录：遍历目录；boxes 为目录则按 stem 配对，为文件则全部共用
    - 图片为单文件：boxes 为文件直接用，为目录则找 <stem>.json
    - boxes_arg 为 None：所有配对的 JSON 为 None（纯文本提示模式用）
    """
    img_path = Path(image_arg)
    box_path = Path(boxes_arg) if boxes_arg else None
    pairs: list[tuple[Path, Path | None]] = []

    if img_path.is_dir():
        for p in sorted(img_path.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if box_path is None:
                pairs.append((p, None))
            else:
                j = (box_path / f"{p.stem}.json") if box_path.is_dir() else box_path
                if j.exists():
                    pairs.append((p, j))
                else:
                    print(f"  [skip] 找不到对应 bbox JSON: {p.name}")
    else:
        if box_path is None:
            pairs.append((img_path, None))
        else:
            j = box_path if box_path.is_file() else (box_path / f"{img_path.stem}.json")
            pairs.append((img_path, j))
    return pairs


def load_boxes_json(json_path: Path):
    """返回 (boxes_xyxy, size_wh_or_None)。"""
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return rec.get("boxes_xyxy", []), rec.get("image_size_wh", None)


def check_size_consistency(image_path: Path, size_wh, warn_fn=print) -> None:
    """JSON 记录的尺寸与实际图片尺寸不一致时告警（框坐标可能错位）。"""
    actual = Image.open(image_path).size
    if size_wh is not None and tuple(size_wh) != tuple(actual):
        warn_fn(f"  [warn] {image_path.name} 尺寸不一致 json={size_wh} "
                f"actual={list(actual)}，框坐标可能错位")


def save_overlay(image_pil: Image.Image, mask: np.ndarray,
                 boxes_xyxy, out_path: Path, alpha: float = 0.45) -> None:
    """原图 + 半透明红色 mask + 绿色提示框，肉眼检查用。"""
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    m = mask.astype(bool)
    if m.any():
        overlay = bgr.copy()
        overlay[m] = (0, 0, 255)
        bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    for (x0, y0, x1, y1) in boxes_xyxy:
        cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def save_results(out_root: Path, stem: str, image_pil: Image.Image,
                 mask: np.ndarray, boxes_xyxy, meta: dict | None = None,
                 save_overlay_flag: bool = True) -> None:
    """统一落盘：mask/<stem>.png + overlay/<stem>.png + meta/<stem>.json。"""
    mask_path = out_root / "mask" / f"{stem}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    if save_overlay_flag:
        save_overlay(image_pil, mask, boxes_xyxy, out_root / "overlay" / f"{stem}.png")

    if meta is not None:
        meta_path = out_root / "meta" / f"{stem}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
