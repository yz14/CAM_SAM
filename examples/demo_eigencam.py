"""EigenCAM 教学 demo（独立可运行）。

与 GradCAM 的差异（重点）
-------------------------
EigenCAM **不需要反向传播、不需要类别标签**。它对目标层的激活张量 A∈(C,H,W) 沿
通道方向做主成分分析（SVD），取 **第一主成分** 作为热力图：
    flat = A.reshape(C, H*W)        # (C, HW)
    U, S, V = svd(flat)
    CAM = V[0].reshape(H, W)        # 第一主成分对应的空间方向
直观：找该层"最显著的共激活模式"，因此结果是 **类别无关的显著区域**。

何时用它
~~~~~~~~
- 模型不可微 / 梯度有问题（量化、ONNX 推理后处理等）
- 想看模型整体"在关注哪里"而非具体类别
- 作为 sanity check：和 GradCAM 差距太大说明类别信号弱

注意：API 上 ``targets`` 可传 None，也可仍传以保持接口一致（库内部不使用）。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_eigencam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_eigencam.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    model = models.resnet34(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
    return model.eval().to(DEVICE)


def load_image(path: str, size: int = 224):
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb_float = np.asarray(pil, dtype=np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil).unsqueeze(0).to(DEVICE), rgb_float


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    input_tensor, rgb_float = load_image(IMAGE)

    target_layers = [model.layer4[-1]]
    cam = EigenCAM(model=model, target_layers=target_layers)
    # ★ targets=None：类别无关
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_eigencam] saved -> {OUT}")


if __name__ == "__main__":
    main()
