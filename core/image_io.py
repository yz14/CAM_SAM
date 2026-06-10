"""图像 I/O 与预处理工具。

设计要点：
- 输出三件套：归一化后的 ``input_tensor`` (1,C,H,W)，
  以及 [0,1] 范围的 ``rgb_float`` (H,W,3) 用于 ``show_cam_on_image`` 叠加。
- 预处理参数（resize / mean / std）走配置，方便迁移到任意模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ImageNet 默认均值方差，迁移到自训练模型时按需覆盖。
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class PreparedImage:
    """统一封装预处理后的两份数据，避免上层多次零散传参。"""

    input_tensor: torch.Tensor  # (1,C,H,W) 已归一化
    rgb_float: np.ndarray       # (H,W,3) float32 in [0,1]
    pil: Image.Image            # 原图(已 resize)，便于保存/调试


def build_transform(
    image_size: int | Sequence[int] = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """构建标准的 Resize -> ToTensor -> Normalize 流水线。"""
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = tuple(image_size)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def load_image(
    path: str | Path,
    image_size: int | Sequence[int] = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    device: str | torch.device = "cpu",
) -> PreparedImage:
    """读取图像并返回 ``PreparedImage``。

    注意：``rgb_float`` 来自 *未归一化* 的 PIL，仅做 resize+ToFloat，
    这样叠加热力图时颜色才正确。
    """
    pil = Image.open(str(path)).convert("RGB")
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = tuple(image_size)
    pil_resized = pil.resize(size[::-1] if len(size) == 2 else size, Image.BILINEAR)

    rgb_float = np.asarray(pil_resized, dtype=np.float32) / 255.0  # (H,W,3)

    tfm = build_transform(image_size=image_size, mean=mean, std=std)
    input_tensor = tfm(pil).unsqueeze(0).to(device)
    return PreparedImage(input_tensor=input_tensor, rgb_float=rgb_float, pil=pil_resized)
