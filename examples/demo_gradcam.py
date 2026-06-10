"""GradCAM 最小可运行示例（教学版）。

阅读顺序：从上到下即可理解一次 CAM 调用的全部要素：
    1) 加载并 eval() 模型
    2) 选定 target_layers（CNN 通常是最后一个卷积块的输出）
    3) 预处理图像 -> input_tensor & 归一化前 RGB（用于叠加可视化）
    4) 用 ClassifierOutputTarget(class_idx) 指定要解释哪一类
    5) 调用 cam(input_tensor, targets) 拿到 (B,H,W) 灰度热力图
    6) show_cam_on_image 叠加保存

运行：
    python examples/demo_gradcam.py

说明：
- 此脚本独立、不依赖 core/，便于作为最简模板复制到任意项目。
- 想换模型？只改 `build_model` 和 `target_layers` 即可（注释里给出了常见架构对照）。
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

# -------------------- 1. 配置（按需修改） --------------------
WEIGHTS = r"D:\codes\work-projects\Gastrovision_results\pretrained\resnet34.pth"
IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_gradcam.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    # 不同架构推荐的 target_layers：
    #   ResNet/ResNeXt:           [model.layer4[-1]]
    #   VGG/MobileNetV2/EffNet:   [model.features[-1]]
    #   DenseNet:                 [model.features.norm5]
    #   ConvNeXt:                 [model.features[-1]]
    #   ViT (timm):               [model.blocks[-1].norm1]   + reshape_transform
    #   Swin (timm):              [model.layers[-1].blocks[-1].norm1] + reshape_transform
    model = models.resnet34(weights=None)
    sd = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(DEVICE)
    return model


def load_image(path: str, size: int = 224):
    """返回 (input_tensor, rgb_float[0,1])。"""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb_float = np.asarray(pil, dtype=np.float32) / 255.0  # (H,W,3)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = tfm(pil).unsqueeze(0).to(DEVICE)
    return input_tensor, rgb_float


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    input_tensor, rgb_float = load_image(IMAGE)

    # 预测最大类作为解释目标
    with torch.no_grad():
        pred = int(model(input_tensor).argmax(1).item())
    print(f"[demo_gradcam] predicted class = {pred}")

    # 关键三步：选层 -> 构 CAM -> 调用
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(pred)])[0]  # (H,W)

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    Image.fromarray(overlay).save(OUT)
    print(f"[demo_gradcam] saved -> {OUT}")


if __name__ == "__main__":
    main()
