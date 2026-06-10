"""EigenGradCAM 教学 demo（独立可运行）。

与 EigenCAM / GradCAM 的差异（重点）
------------------------------------
EigenCAM 是 **类别无关** 的（直接 SVD 激活）。
EigenGradCAM 把"激活"换成"激活 × 梯度"再做 SVD：
    M = A * dY/dA
    CAM = first_principal_component(M)
于是结果对 **目标类别敏感**，同时保留 EigenCAM "主成分降噪"的好处 ——
比朴素 GradCAM 更稳，但比 EigenCAM 更有类别区分能力。

成本：比 GradCAM 多一次 SVD，单图基本无感；大 batch 时略慢。

API 与 GradCAM 完全一致。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_eigengradcam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_eigengradcam.png"
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

    with torch.no_grad():
        pred = int(model(input_tensor).argmax(1).item())
    print(f"[demo_eigengradcam] predicted class = {pred}")

    target_layers = [model.layer4[-1]]
    cam = EigenGradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_eigengradcam] saved -> {OUT}")


if __name__ == "__main__":
    main()
