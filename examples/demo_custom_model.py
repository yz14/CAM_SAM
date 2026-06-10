"""自写 ``nn.Module`` 接入 CAM 的完整教学 demo（独立可运行）。

场景
----
你不是在 torchvision/timm 现成模型上做 CAM，而是有一个自己写的网络（比如医疗 / 工业
定制结构）。本 demo 用 ``custom_models/my_simple_cnn.py`` 中的 ``SimpleCNN`` 走完整流程。

接入 CAM 的三件事
~~~~~~~~~~~~~~~~~
1. **构造模型 + eval()**：保证 BN/Dropout 处于推理模式。
2. **识别一个"最后一个卷积块的输出"作为 target_layer**：CAM 在此层抽激活与梯度。
3. **保留梯度通路**：不要在 forward 里 ``torch.no_grad()`` 包裹整段；CAM 依赖 hook
   能拿到反向梯度（GradCAM 系列）或重新前向（ScoreCAM/AblationCAM）。

如何挑 target_layer
~~~~~~~~~~~~~~~~~~~
经验法则："**做了语义抽取、但还没有被 GAP/Flatten 压扁的最后一处 4D 特征图**"。
在 ``SimpleCNN`` 中这就是 ``self.features[-2]``（也即 ``features[10]``，第 3 个 conv block
末尾的 ReLU 输出）；也可以传 ``self.features[-1]`` （MaxPool2d）—— 池化对 CAM 影响很小。

> 不确定？打印 ``model`` 看结构，或对中间张量临时 ``print(x.shape)``。
> 4D 张量 (B,C,H,W) 且 H、W ≥ 7 通常都是合理候选。

权重
~~~~
本 demo 用 **随机初始化** 演示 pipeline；现实使用时把
``torch.load("your_ckpt.pth")`` 的 ``state_dict`` 加载进来即可。

运行：
    D:\\miniconda\\envs\\py310\\python.exe examples/demo_custom_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# 让 "from custom_models import ..." 在脚本直接运行时也能工作
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from custom_models.my_simple_cnn import SimpleCNN  # noqa: E402

IMAGE = r"D:\codes\data\Gastrovision\Colon diverticula\3.jpg"
OUT = _REPO_ROOT / "outputs" / "demo_custom_model.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10


def build_model() -> SimpleCNN:
    model = SimpleCNN(num_classes=NUM_CLASSES, width=32)
    # ↓ 现实中：替换为你的 checkpoint
    # sd = torch.load("path/to/your.pth", map_location="cpu")
    # model.load_state_dict(sd, strict=True)
    print("[demo_custom_model] 警告：随机权重，仅展示 pipeline 通畅。")
    return model.eval().to(DEVICE)


def load_image(path: str, size: int = 224):
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb_float = np.asarray(pil, dtype=np.float32) / 255.0
    # 自训练模型 mean/std 通常按你自己的训练集统计；这里沿用 ImageNet 仅为演示
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
    print(f"[demo_custom_model] predicted class = {pred} (随机权重下无意义，仅作流程展示)")

    # ★ 关键：从自写模型里挑出最后一层"语义 + 4D"特征图
    #    SimpleCNN.features 末尾两层依次为 ReLU (-2) 和 MaxPool2d (-1)，都可以
    target_layers = [model.features[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred)])[0]

    Image.fromarray(show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)).save(OUT)
    print(f"[demo_custom_model] saved -> {OUT}")


if __name__ == "__main__":
    main()
