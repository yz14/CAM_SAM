"""可视化与保存：热力图叠加、原图/热力图/叠加图横向拼接保存。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from .logger import get_logger

log = get_logger("vis")


def overlay(rgb_float: np.ndarray, grayscale_cam: np.ndarray, use_rgb: bool = True) -> np.ndarray:
    """生成叠加图。``grayscale_cam`` 应为 (H,W) float in [0,1]。"""
    if grayscale_cam.ndim == 3:
        grayscale_cam = grayscale_cam[0]
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=use_rgb)


def heatmap_only(grayscale_cam: np.ndarray) -> np.ndarray:
    """单独的热力图（灰度->伪彩），用于对比。"""
    import cv2
    if grayscale_cam.ndim == 3:
        grayscale_cam = grayscale_cam[0]
    hm = np.uint8(255 * grayscale_cam)
    color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return color[..., ::-1]  # BGR->RGB


def save_grid(
    rgb_float: np.ndarray,
    grayscale_cam: np.ndarray,
    out_path: str | Path,
    titles: Sequence[str] | None = None,
) -> Path:
    """保存 [原图 | 热力图 | 叠加图] 横向拼接。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig = np.uint8(rgb_float * 255)
    hm = heatmap_only(grayscale_cam)
    over = overlay(rgb_float, grayscale_cam, use_rgb=True)

    panels = [orig, hm, over]
    h = max(p.shape[0] for p in panels)
    panels = [_pad_to_height(p, h) for p in panels]
    grid = np.concatenate(panels, axis=1)

    Image.fromarray(grid).save(out_path)
    log.info("saved: %s", out_path)
    return out_path


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
    return np.concatenate([img, pad], axis=0)
