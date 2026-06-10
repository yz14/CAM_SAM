"""LayerCAM 教学 demo（独立可运行）。

与 GradCAM 的差异（重点）
-------------------------
LayerCAM（Jiang et al. 2021）的核心创新：**让浅层也能可视化**。
GradCAM 在浅层（如 layer1/layer2）效果差，因为浅层通道语义弱、平均池化把
位置信息抹掉。LayerCAM 改成：
    CAM = sum_c ReLU(grad) * A
即用"梯度的正部分"作为 **空间逐位置的权重**，再与激活相乘 —— 因此浅层也能
得到合理的、细粒度的定位。

实战建议
~~~~~~~~
- 浅层（layer2, layer3）单独看 LayerCAM 通常优于 GradCAM。
- 也可把多个层的结果取平均做"多尺度融合"。

本 demo 展示 **同时传入多个 target_layer**（库会按层分别算 CAM 再融合，输出形状仍为 (B,H,W)）。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_layercam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_layercam.png"
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
    print(f"[demo_layercam] predicted class = {pred}")

    # ★ LayerCAM 鼓励多层融合：浅 + 深，得到的热力图既保留细节又有语义
    target_layers = [model.layer2[-1], model.layer3[-1], model.layer4[-1]]
    cam = LayerCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_layercam] saved -> {OUT}")


if __name__ == "__main__":
    main()
