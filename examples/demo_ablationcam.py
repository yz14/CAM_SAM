"""AblationCAM 教学 demo（独立可运行，**慢方法**）。

与 GradCAM 的差异（重点）
-------------------------
AblationCAM **不使用反向传播**，思路是 "扰动法"：把目标层第 k 个通道的激活置零，
观察类别得分下降多少 —— 下降越多说明该通道越重要：
    w_k = (Y_orig - Y_{ablate_k}) / Y_orig
    CAM = ReLU( sum_k w_k * A_k )
因此它非常 **直观**、解释力强；缺点是要做 C 次前向（C 是通道数），**很慢**。

工程参数（``extra``）
~~~~~~~~~~~~~~~~~~~~
- ``batch_size``: 一次前向同时消融多少通道（取决于显存）。在 cam_runner 的 yaml 里
  通过 ``cam.extra.batch_size`` 配置；这里直接传给 AblationCAM 构造函数。
- ``ratio_channels_to_ablate``: 仅消融重要性预估靠前的 r% 通道（默认 1.0=全部）；
  设 0.25 等价于"采样近似"，在大模型上提速 4×。

何时用
~~~~~~
- 需要 **强可解释性** 的医疗 / 高风险场景（解释逻辑直白）
- 调试模型是否依赖某些通道
- 不太适合实时或大批量评估

运行（耗时数秒到分钟级）：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_ablationcam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_ablationcam.png"
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
    print(f"[demo_ablationcam] predicted class = {pred} (慢方法，请耐心等待)")

    target_layers = [model.layer4[-1]]
    # ★ AblationCAM 特有参数：batch_size 控制一次前向消融多少通道
    cam = AblationCAM(
        model=model,
        target_layers=target_layers,
        batch_size=32,
        ratio_channels_to_ablate=1.0,  # 完整版；调小可加速近似
    )
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_ablationcam] saved -> {OUT}")


if __name__ == "__main__":
    main()
