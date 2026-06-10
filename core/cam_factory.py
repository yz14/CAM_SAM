"""按算法名构建 ``pytorch_grad_cam`` CAM 对象。

各算法关键差异（教学速查）：

| 方法              | 需要梯度 | 需要类别 | 速度  | 备注                                                |
|-------------------|----------|----------|-------|-----------------------------------------------------|
| GradCAM           | 是       | 是       | 快    | 经典基线                                             |
| GradCAM++         | 是       | 是       | 快    | 多目标场景更稳                                        |
| XGradCAM          | 是       | 是       | 快    | 对类别一致性更友好                                    |
| HiResCAM          | 是       | 是       | 快    | 元素级乘积，定位更精细                                |
| GradCAMElementWise| 是       | 是       | 快    | 仅保留正贡献，去掉 mean                              |
| LayerCAM          | 是       | 是       | 快    | 浅层也能用，结合多层效果好                           |
| EigenCAM          | 否       | 否       | 快    | 直接 SVD，类别无关，容易得到与类别无关的显著区域       |
| EigenGradCAM      | 是       | 是       | 中    | EigenCAM × 梯度，类别敏感                            |
| AblationCAM       | 否       | 是       | **慢** | 通过逐通道置零计算重要性，需要 batched 前向            |
| ScoreCAM          | 否       | 是       | **慢** | 用激活当 mask 重新前向，效果干净但开销大              |
| FullGrad          | 是       | 是       | 中    | 自动聚合所有偏置层，无需指定 target_layers（仍传一个占位） |
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping

import torch.nn as nn
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    FullGrad,
    GradCAM,
    GradCAMElementWise,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)

from .logger import get_logger

log = get_logger("cam")

# 名称（小写）-> 类
_REGISTRY: dict[str, type] = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "gradcampp": GradCAMPlusPlus,
    "gradcamplusplus": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "hirescam": HiResCAM,
    "gradcamelementwise": GradCAMElementWise,
    "layercam": LayerCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM,
    "fullgrad": FullGrad,
}

# 是否需要 target 类别（None 表示用预测最大类）
NEEDS_CLASS: dict[str, bool] = {
    "eigencam": False,
    "fullgrad": True,  # 库实现仍然要求传 targets
}

# 慢方法（建议小 batch / 子集调试）
SLOW_METHODS = {"ablationcam", "scorecam"}


def list_methods() -> list[str]:
    return sorted({v.__name__ for v in _REGISTRY.values()})


def build_cam(
    method: str,
    model: nn.Module,
    target_layers: List[nn.Module],
    reshape_transform: Callable | None = None,
    extra: Mapping[str, Any] | None = None,
):
    """构建 CAM 对象。

    extra: 算法特定参数（例如 AblationCAM 的 ``batch_size`` / ``ratio_channels_to_ablate``）。
    """
    key = method.lower().replace("_", "").replace("-", "")
    if key not in _REGISTRY:
        raise ValueError(
            f"未知 CAM 方法: {method}. 可选: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[key]
    kwargs: dict[str, Any] = {
        "model": model,
        "target_layers": target_layers,
        "reshape_transform": reshape_transform,
    }
    if extra:
        kwargs.update(extra)
    log.info("构建 %s (slow=%s)", cls.__name__, key in SLOW_METHODS)
    return cls(**kwargs)
