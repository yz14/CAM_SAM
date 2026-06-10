"""HiResCAM 教学 demo（独立可运行）。

与 GradCAM 的差异（重点）
-------------------------
GradCAM 把梯度先在 H×W 上 **全局平均池化**再与激活相乘，因此热力图低分辨率、定位
偏粗。HiResCAM（Draelos & Carin, 2020）改为：
    CAM = ReLU( sum_c ( A * dY/dA ) )   # 元素级乘积，再按通道求和
不做空间平均，保留每个空间位置的真实贡献，**定位更精细**，且可以证明它"忠实地反映
了 (在某类 CNN 结构下) 模型用于决策的特征"。

注意：HiResCAM 推荐在 **最后一个卷积层**用；如果你的网络在 conv 之后还有大量
非线性变换（全局池化只算一种线性算子，relu/bn 不影响），仍然合理；若中间还有
attention 等复杂模块，则忠实性保证不再严格成立。

API 与 GradCAM 完全一致。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_hirescam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_hirescam.png"
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
    print(f"[demo_hirescam] predicted class = {pred}")

    target_layers = [model.layer4[-1]]
    cam = HiResCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_hirescam] saved -> {OUT}")


if __name__ == "__main__":
    main()
