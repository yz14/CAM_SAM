"""ScoreCAM 教学 demo（独立可运行，**慢方法**）。

与 GradCAM 的差异（重点）
-------------------------
ScoreCAM 同样 **无梯度**：把目标层每个通道的激活归一化后当作 **mask** 叠加到原图，
重新前向得到该类的得分作为该通道的权重：
    M_k = upsample( normalize(A_k) ) * x         # 用激活当掩码
    w_k = softmax( f(M_k) )_target_class         # 重新前向算分
    CAM = ReLU( sum_k w_k * A_k )
直观：每个通道的 mask 让原图保留"该通道认为重要"的区域，得分越高的通道权重越大。
优点是 **结果非常干净、稳定**（无梯度噪声）；缺点是 **每通道一次前向**，比 AblationCAM
还略慢，但实现更简单。

工程参数
~~~~~~~~
- ``batch_size``: 同 AblationCAM；显存大就开大。
- 没有 ``ratio_channels_to_ablate``。

何时用
~~~~~~
- 需要论文质量、视觉非常干净的图
- 模型梯度不可用（量化推理 / 蒸馏后 stop_grad 等）

运行（数秒到分钟级）：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_scorecam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_scorecam.png"
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
    print(f"[demo_scorecam] predicted class = {pred} (慢方法，请耐心等待)")

    target_layers = [model.layer4[-1]]
    # ★ ScoreCAM 特有参数：batch_size（控制每次前向多少通道掩码）
    cam = ScoreCAM(model=model, target_layers=target_layers, batch_size=32)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_scorecam] saved -> {OUT}")


if __name__ == "__main__":
    main()
