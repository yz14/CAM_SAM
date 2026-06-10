"""FullGrad 教学 demo（独立可运行）。

与 GradCAM 的差异（重点）
-------------------------
FullGrad（Srinivas & Fleuret, NeurIPS 2019）认为只看"最后一层激活 × 梯度"会丢掉
偏置项（bias）通过非线性传播带来的贡献。它把 **全网络每个含偏置层的偏置敏感度**
都聚合到输入空间：
    Full = |dY/dx * x| + sum_layer  upsample( |dY/dbias_layer * bias_layer| )
等价于"完整反向归因 + 所有 bias 项的影响"。结果通常 **覆盖更广、对全局上下文更敏感**。

注意（这是 FullGrad 与所有 CAM 的本质不同）：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- FullGrad **不挑某一层**；它自动扫描 model 里所有有 ``bias`` 属性的层。
- 但 ``pytorch_grad_cam`` 的统一构造接口仍然要求传 ``target_layers``。这里传一个
  占位（如 ``model.layer4[-1]``）即可，库内部并不会真正用它做最终聚合。
- ``targets`` 仍然要传（指明对哪个类做归因）。
- 模型必须含 BN / Linear / Conv 之类的 bias 层；纯无 bias 的网络上效果有限。

API 与 GradCAM 一致。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_fullgrad.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_fullgrad.png"
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
    print(f"[demo_fullgrad] predicted class = {pred}")

    # ★ FullGrad 并不真正使用 target_layers（它扫描所有 bias 层），
    #   但接口要求传一个占位 module；这里用最后一个 layer 占位。
    target_layers = [model.layer4[-1]]
    cam = FullGrad(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_fullgrad] saved -> {OUT}")


if __name__ == "__main__":
    main()
