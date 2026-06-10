"""Swin Transformer 上跑 CAM 的完整教学 demo（torchvision ``swin_t``，独立可运行）。

Swin vs ViT 的 reshape 差异（重点）
-----------------------------------
ViT 全程 (B, N, C) 一条序列、且有 CLS token。
Swin 分 4 个 stage，**block 内部本来就是 (B, H, W, C)**（NHWC 而非 NCHW）。
torchvision ``swin_t`` 的 backbone 通过 ``model.features`` 串起所有 stage，最后
两个元素分别是：

    model.features[-2]  : 最终 LayerNorm，输出形状  (B, 7, 7, 768)   ← NHWC
    model.features[-1]  : Permute([0,3,1,2])，输出  (B, 768, 7, 7)   ← 已是 NCHW

因此对 Swin 有 **两种 hook 方式**：

A) hook ``features[-2]`` (LayerNorm)，输出 NHWC，**需要 reshape_transform**：
       lambda x: x.permute(0, 3, 1, 2).contiguous()

B) hook ``features[-1]`` (Permute)，输出已经是 NCHW，**不需要 reshape_transform**。

本 demo 采用 (A) 更具教学价值：展示 token-序列模型 / NHWC 模型如何被统一处理。
也用注释列出 (B) 的写法供你切换。

torchvision Swin-T 结构对照
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model.features = Sequential(
        PatchEmbed,                # [0]
        SwinBlock x 2,             # [1]   stage 1
        PatchMerging,              # [2]
        SwinBlock x 2,             # [3]   stage 2
        PatchMerging,              # [4]
        SwinBlock x 6,             # [5]   stage 3
        PatchMerging,              # [6]
        SwinBlock x 2,             # [7]   stage 4
        LayerNorm,                 # [-2]  ← 最后归一化（NHWC，7x7）
        Permute([0,3,1,2]),        # [-1]  ← NHWC -> NCHW
    )
    model.avgpool / flatten / head

权重说明
~~~~~~~~
默认随机初始化。改 ``USE_PRETRAINED = True`` 即可让 torchvision 下载 ImageNet 权重，
或自行 ``load_state_dict``。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_swin.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_swin.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_PRETRAINED = False  # True 时 torchvision 会下载 ImageNet 预训练权重


def build_model() -> torch.nn.Module:
    if USE_PRETRAINED:
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_t(weights=None)
        print("[demo_swin] 警告：使用随机初始化，热力图仅展示 pipeline 是否通畅。")
    return model.eval().to(DEVICE)


def load_image(path: str, size: int = 224):
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb_float = np.asarray(pil, dtype=np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil).unsqueeze(0).to(DEVICE), rgb_float


def swin_nhwc_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Swin 专用 reshape：(B, H, W, C) -> (B, C, H, W)。

    注意：与 ViT 不同，**没有 CLS token 需要丢弃**，输入张量本身就是 4D NHWC。
    """
    return tensor.permute(0, 3, 1, 2).contiguous()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    input_tensor, rgb_float = load_image(IMAGE)

    with torch.no_grad():
        pred = int(model(input_tensor).argmax(1).item())
    print(f"[demo_swin] predicted class = {pred}")

    # ★ 写法 (A)：hook 最终 LayerNorm，输出 NHWC，需要 reshape_transform
    target_layers = [model.features[-2]]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=swin_nhwc_to_nchw,
    )

    # 写法 (B) 等价做法（不需要 reshape_transform）：
    #   target_layers = [model.features[-1]]   # Permute, 输出已是 (B,C,H,W)
    #   cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_swin] saved -> {OUT}")


if __name__ == "__main__":
    main()
