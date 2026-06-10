"""目标层选择与 ViT/Swin 的 reshape_transform。

CAM 算法核心要素之一就是 *目标层*：通常选 **最后一个卷积块的输出**。
- ResNet / ResNeXt / WideResNet  : ``model.layer4[-1]``
- VGG / MobileNetV2 / EfficientNet (torchvision): ``model.features[-1]``
- DenseNet                       : ``model.features.norm5``
- ConvNeXt (torchvision)         : ``model.features[-1]``
- ViT (timm/torchvision)         : 最后一个 transformer block 的 ``norm1``
- Swin                           : 最后一个 stage 的 ``norm``

ViT/Swin 这类把空间维度展平为 token 序列的模型，必须提供 ``reshape_transform``，
把 (B, N, C) 的激活恢复为 (B, C, H, W)，CAM 才能在空间上可视化。

迁移到自定义模型时：在 yaml 中直接给 ``target_layer: "layer3.1.conv2"`` 之类的字符串路径，
由 :func:`resolve_target_layer` 解析。
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch
import torch.nn as nn

from .logger import get_logger

log = get_logger("layers")


# ---------- 1. 字符串路径解析 ----------

def _resolve_attr_path(root: nn.Module, path: str) -> nn.Module:
    """支持 'layer4.1.conv2' / 'features[3].block' 形式的路径。"""
    obj: object = root
    # 简化：把 [i] 形式替换为 .i
    norm = path.replace("[", ".").replace("]", "")
    for token in norm.split("."):
        if not token:
            continue
        if token.lstrip("-").isdigit():
            obj = obj[int(token)]  # type: ignore[index]
        else:
            obj = getattr(obj, token)
    if not isinstance(obj, nn.Module):
        raise TypeError(f"路径 {path} 解析到的不是 nn.Module: {type(obj)}")
    return obj


def resolve_target_layer(model: nn.Module, spec: str | Sequence[str]) -> List[nn.Module]:
    """把 yaml 里写的层路径解析为 ``List[nn.Module]``（pytorch_grad_cam 要求列表）。"""
    if isinstance(spec, str):
        specs = [spec]
    else:
        specs = list(spec)
    layers = [_resolve_attr_path(model, s) for s in specs]
    log.info("target_layers = %s", specs)
    return layers


# ---------- 2. 启发式自动选择（CNN 常见架构） ----------

def auto_pick_target_layer(model: nn.Module) -> List[nn.Module]:
    """对常见 torchvision CNN 给出合理的默认目标层；不确定时抛错让用户显式指定。"""
    # ResNet 家族
    if hasattr(model, "layer4"):
        return [model.layer4[-1]]
    # VGG / MobileNet / EfficientNet / ConvNeXt
    if hasattr(model, "features"):
        feats = model.features
        # DenseNet 的 features 末尾是 norm5
        if hasattr(feats, "norm5"):
            return [feats.norm5]
        return [feats[-1]]
    raise RuntimeError(
        "无法自动选择 target_layer，请在 yaml 中显式指定 model.target_layer，例如 'layer4.-1'"
    )


# ---------- 3. reshape_transform 工厂 ----------

def vit_reshape_transform(height: int = 14, width: int = 14, has_cls_token: bool = True) -> Callable:
    """ViT: (B, 1+H*W, C) -> (B, C, H, W)。

    14x14 对应 224 输入、patch_size=16 的常见 ViT-B。其它配置请按 patch 数量调整。
    """
    def _t(tensor: torch.Tensor) -> torch.Tensor:
        x = tensor[:, 1:, :] if has_cls_token else tensor
        x = x.reshape(x.size(0), height, width, x.size(-1))
        return x.permute(0, 3, 1, 2).contiguous()
    return _t


def swin_reshape_transform(height: int = 7, width: int = 7) -> Callable:
    """Swin: 最后一个 stage 输出形如 (B, H*W, C)，没有 cls token。"""
    def _t(tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        return x.permute(0, 3, 1, 2).contiguous()
    return _t


def build_reshape_transform(cfg: dict | None) -> Callable | None:
    """从 yaml 的 ``reshape_transform`` 段构建函数；None 表示 CNN，无需 reshape。"""
    if not cfg:
        return None
    kind = cfg.get("kind", "").lower()
    if kind == "vit":
        return vit_reshape_transform(
            height=int(cfg.get("height", 14)),
            width=int(cfg.get("width", 14)),
            has_cls_token=bool(cfg.get("has_cls_token", True)),
        )
    if kind == "swin":
        return swin_reshape_transform(
            height=int(cfg.get("height", 7)),
            width=int(cfg.get("width", 7)),
        )
    raise ValueError(f"未知 reshape_transform.kind: {kind}")
