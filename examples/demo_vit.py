"""ViT 上跑 CAM 的完整教学 demo（torchvision ``vit_b_16``，独立可运行）。

CNN vs Transformer 的关键不同
------------------------------
CNN 中卷积层激活天然是 (B, C, H, W)，CAM 直接在 H×W 上聚合即可。
ViT 把图像切成 patch 序列：

    image (B, 3, 224, 224)
      └─ conv_proj (patch_size=16) ─▶ (B, 768, 14, 14) ─reshape▶ (B, 196, 768)
      └─ 拼接 CLS token            ─▶ (B, 197, 768)
      └─ encoder.layers[i]         ─▶ (B, 197, 768)  ← CAM 要 hook 这个形状
      └─ encoder.ln                ─▶ (B, 197, 768)
      └─ heads(Linear)             ─▶ (B, num_classes)

因此 hook 到的激活是 **token 序列** (B, N, C)，不是空间图。我们必须告诉
``pytorch_grad_cam`` 如何"把 token 序列变回 H×W 空间图"——这就是
``reshape_transform`` 的唯一职责：

    (B, 1+H*W, C)  -- 丢掉 CLS token -->  (B, H*W, C)
                   -- reshape -->         (B, H, W, C)
                   -- permute -->         (B, C, H, W)

只要这一步对了，下游所有 GradCAM/GradCAM++/EigenCAM... 都能像 CNN 一样工作。

目标层的选取
~~~~~~~~~~~~
ViT 的"最有信息量"的层通常是 **最后一个 Encoder Block 的第一个 LayerNorm**
（``ln_1``，在自注意力之前）。这是社区惯例（jacobgil/pytorch-grad-cam 仓库也这样
推荐）。也可以试 ``ln_2``（MLP 之前）或最后的 ``encoder.ln``。

torchvision ViT-B/16 结构对照
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model.conv_proj                      Conv2d(patch_embed)
    model.encoder.layers[0..11]          EncoderBlock
        .ln_1   .self_attention   .ln_2   .mlp
    model.encoder.ln                     final LayerNorm
    model.heads                          Linear

权重说明
~~~~~~~~
本 demo 默认 **随机初始化**——重点是教学"如何接管 token 序列"。把
``USE_PRETRAINED = True`` 改为有网络环境，torchvision 会自动下载 ImageNet 权重；
也可以把你自己的 ``state_dict`` 通过 ``model.load_state_dict(...)`` 传入。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_vit.py
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
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_vit.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 224 / patch_size(16) = 14；ViT-B/16 的空间网格固定 14x14
GRID_H = GRID_W = 14
USE_PRETRAINED = False  # 改为 True 时 torchvision 会下载 ImageNet 权重


def build_model() -> torch.nn.Module:
    if USE_PRETRAINED:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = models.vit_b_16(weights=None)
        print("[demo_vit] 警告：使用随机初始化，热力图仅展示 pipeline 是否通畅。"
              " 把 USE_PRETRAINED=True 或自行 load_state_dict 加载真实权重。")
    return model.eval().to(DEVICE)


def load_image(path: str, size: int = 224):
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb_float = np.asarray(pil, dtype=np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil).unsqueeze(0).to(DEVICE), rgb_float


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """ViT 专用：(B, 1+H*W, C) -> (B, C, H, W)。

    步骤：
        1) tensor[:, 1:, :]  丢掉位置 0 的 CLS token，得到 (B, H*W, C)
        2) reshape           -> (B, H, W, C)
        3) permute           -> (B, C, H, W)
    """
    x = tensor[:, 1:, :]
    x = x.reshape(x.size(0), GRID_H, GRID_W, x.size(-1))
    return x.permute(0, 3, 1, 2).contiguous()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    input_tensor, rgb_float = load_image(IMAGE)

    with torch.no_grad():
        pred = int(model(input_tensor).argmax(1).item())
    print(f"[demo_vit] predicted class = {pred}")

    # ★ 关键 2 行：选最后一个 EncoderBlock 的 ln_1 + 传入 vit_reshape_transform
    target_layers = [model.encoder.layers[-1].ln_1]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform,
    )
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_vit] saved -> {OUT}")


if __name__ == "__main__":
    main()
