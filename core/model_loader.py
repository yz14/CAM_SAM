"""模型加载：支持 torchvision、timm、自定义 import 路径。

迁移到自定义模型的两种方式：
1. **torchvision/timm**：在 yaml 里设置 ``source: torchvision`` 或 ``timm`` 与 ``name``。
2. **自定义模型**：``source: custom``，给出 ``entrypoint: pkg.module:build_fn``，
   该函数应当返回一个 ``nn.Module``。可通过 ``entrypoint_kwargs`` 传参。

权重加载兼容三种保存方式：
    - 直接的 ``state_dict`` (OrderedDict)
    - {'state_dict': ...} / {'model': ...} 字典
    - 整个 ``nn.Module`` (torch.save(model))
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .logger import get_logger

log = get_logger("model")


def _import_entrypoint(spec: str):
    """解析 'pkg.module:attr' 形式的入口。"""
    if ":" not in spec:
        raise ValueError(f"entrypoint 必须形如 'pkg.module:func'，收到: {spec}")
    mod_path, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, attr):
        raise AttributeError(f"{mod_path} 中找不到 {attr}")
    return getattr(mod, attr)


def _build_torchvision(name: str, num_classes: int | None, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm

    if not hasattr(tvm, name):
        raise ValueError(f"torchvision.models 没有 {name}")
    ctor = getattr(tvm, name)
    # 新版 torchvision 推荐使用 weights=...，但为兼容性这里用 pretrained=False
    # 让用户通过 weights_path 显式提供权重，避免无网络环境下载失败。
    try:
        model = ctor(weights=None)
    except TypeError:
        model = ctor(pretrained=False)

    if num_classes is not None and hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            log.info("替换 fc: %d -> %d", model.fc.out_features, num_classes)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_timm(name: str, num_classes: int | None, pretrained: bool) -> nn.Module:
    import timm  # 延迟导入，未安装时不影响其它分支

    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    return timm.create_model(name, **kwargs)


def _extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in obj and isinstance(obj[key], Mapping):
                return obj[key]
        return obj  # type: ignore[return-value]
    raise TypeError(f"无法识别的权重对象类型: {type(obj)}")


def load_weights(model: nn.Module, weights_path: str | Path, strict: bool = False) -> nn.Module:
    """加载权重。``strict=False`` 允许 fc 维度不一致等情况下迁移。"""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    log.info("加载权重: %s", weights_path)
    obj = torch.load(str(weights_path), map_location="cpu")
    sd = _extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        log.warning("missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        log.warning("unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])
    return model


def build_model(cfg: Mapping[str, Any], device: str | torch.device = "cpu") -> nn.Module:
    """根据 yaml 中的 ``model:`` 段构建模型。

    cfg 字段约定：
        source        : 'torchvision' | 'timm' | 'custom'
        name          : torchvision/timm 模型名（custom 时忽略）
        num_classes   : 可选，覆盖最后一层
        pretrained    : 是否使用框架自带的预训练权重
        weights_path  : 可选，本地权重文件
        weights_strict: 加载权重的 strict（默认 False）
        entrypoint    : 仅 custom 时使用，'pkg.module:func'
        entrypoint_kwargs: 仅 custom 时使用
    """
    source = cfg.get("source", "torchvision").lower()
    num_classes = cfg.get("num_classes")
    pretrained = bool(cfg.get("pretrained", False))

    if source == "torchvision":
        model = _build_torchvision(cfg["name"], num_classes, pretrained)
    elif source == "timm":
        model = _build_timm(cfg["name"], num_classes, pretrained)
    elif source == "custom":
        entry = _import_entrypoint(cfg["entrypoint"])
        model = entry(**cfg.get("entrypoint_kwargs", {}))
        if not isinstance(model, nn.Module):
            raise TypeError("entrypoint 必须返回 nn.Module")
    else:
        raise ValueError(f"未知 model.source: {source}")

    if cfg.get("weights_path"):
        load_weights(model, cfg["weights_path"], strict=bool(cfg.get("weights_strict", False)))

    model.eval().to(device)
    log.info("model=%s params=%.2fM device=%s", cfg.get("name", source),
             sum(p.numel() for p in model.parameters()) / 1e6, device)
    return model
