"""一个迷你自定义 CNN，作为 "如何把自写模型接入 CAM" 的最小演示。

要点
----
1. 模型必须最终输出 ``(B, num_classes)`` 的 logits（CAM 的 ClassifierOutputTarget 需要它）。
2. **必须存在一处显式的卷积特征图**——它就是天然的 CAM 目标层。
   这里命名为 ``features``，最后一层是 ``Conv2d``，输出 (B, C, H, W)。
3. 模型内部不能在 forward 期间禁用 grad（避免 ``torch.no_grad()`` 包裹）；CAM
   依赖 hook 的 forward/backward 各传一次。

接入方式
--------
- 在 yaml 中以 ``custom`` 源 + entrypoint 字符串引用本文件的 ``build_net``：

    model:
      source: custom
      entrypoint: custom_models.my_simple_cnn:build_net
      entrypoint_kwargs: { num_classes: 10 }
      target_layer: "features.6"   # 最后一个 Conv2d 的路径

- 在 standalone 脚本中 ``from custom_models.my_simple_cnn import build_net`` 即可。

不限制框架内还能加 BN/ReLU/Dropout —— 选 target_layer 时挑 **最后一个 Conv2d / 卷积块输出** 即可。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """3 个下采样级 + GAP + 线性分类头。

    输入 224x224 时，最后卷积特征图为 28x28（被 3 次 stride=2 下采样到 224/8）。
    """

    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.features = nn.Sequential(
            _conv_block(3, c1),             # [0..2]
            nn.MaxPool2d(2),                # [3]
            _conv_block(c1, c2),            # [4..6]   ← 想 hook 中层就选这里
            nn.MaxPool2d(2),                # [7]
            _conv_block(c2, c3),            # [8..10]  ← 推荐 target_layer
            nn.MaxPool2d(2),                # [11]
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


def build_net(num_classes: int = 10, width: int = 32) -> nn.Module:
    """工厂函数：供 ``model.source: custom`` 的 yaml 通过 entrypoint 调用。"""
    return SimpleCNN(num_classes=num_classes, width=width)
