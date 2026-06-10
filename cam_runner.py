"""统一 CAM 入口：通过 yaml 切换算法 / 模型 / 输入 / 目标层。

用法：
    python cam_runner.py --config configs/default.yaml
    python cam_runner.py --config configs/default.yaml --method gradcam++ --image path/to/x.jpg

设计原则：
- 业务逻辑全部在 ``core/``，本文件只做 *组装*。
- 任意一个算法都共用同一条管线，便于对比。
- 单文件 + yaml 即可批量切换；如需教学示例，请看 ``examples/``。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.cam_factory import build_cam, SLOW_METHODS
from core.image_io import load_image
from core.logger import get_logger, set_level
from core.model_loader import build_model
from core.target_layers import (
    auto_pick_target_layer,
    build_reshape_transform,
    resolve_target_layer,
)
from core.visualize import save_grid

log = get_logger("runner")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CAM unified runner")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--method", type=str, default=None, help="覆盖 yaml 中的 cam.method")
    p.add_argument("--image", type=str, default=None, help="覆盖 yaml 中的 input.image")
    p.add_argument("--target-class", type=int, default=None,
                   help="覆盖 yaml 中的 cam.target_class；不传则用预测最大类")
    p.add_argument("--output", type=str, default=None, help="覆盖 yaml 中的 output.path")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        set_level("DEBUG")

    cfg = load_config(args.config)
    if args.method:
        cfg.setdefault("cam", {})["method"] = args.method
    if args.image:
        cfg.setdefault("input", {})["image"] = args.image
    if args.output:
        cfg.setdefault("output", {})["path"] = args.output
    if args.target_class is not None:
        cfg.setdefault("cam", {})["target_class"] = args.target_class

    # 1. device
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # 2. 模型
    model = build_model(cfg["model"], device=device)

    # 3. target_layers
    spec = cfg["model"].get("target_layer")
    if spec:
        target_layers = resolve_target_layer(model, spec)
    else:
        log.info("model.target_layer 未指定，使用启发式自动选择")
        target_layers = auto_pick_target_layer(model)

    # 4. reshape_transform (ViT/Swin)
    reshape = build_reshape_transform(cfg["model"].get("reshape_transform"))

    # 5. 输入图像
    img = load_image(
        cfg["input"]["image"],
        image_size=cfg["input"].get("image_size", 224),
        mean=cfg["input"].get("mean", (0.485, 0.456, 0.406)),
        std=cfg["input"].get("std", (0.229, 0.224, 0.225)),
        device=device,
    )

    # 6. 预测最大类（仅在未指定 target_class 时使用，便于解释）
    cam_cfg = cfg.get("cam", {})
    target_class = cam_cfg.get("target_class")
    with torch.no_grad():
        logits = model(img.input_tensor)
    pred_class = int(logits.argmax(dim=1).item())
    log.info("predicted class index = %d (logit=%.3f)", pred_class, float(logits[0, pred_class]))
    if target_class is None:
        target_class = pred_class

    # 7. 构建 CAM
    method = cam_cfg.get("method", "gradcam")
    extra = dict(cam_cfg.get("extra", {}) or {})
    if method.lower() in SLOW_METHODS:
        log.warning("方法 %s 较慢；如需更快预览可减小 batch_size 或换用 GradCAM", method)

    cam = build_cam(method, model, target_layers, reshape_transform=reshape, extra=extra)

    # 8. 计算热力图
    targets = [ClassifierOutputTarget(target_class)]
    aug_smooth = bool(cam_cfg.get("aug_smooth", False))
    eigen_smooth = bool(cam_cfg.get("eigen_smooth", False))
    log.info("running CAM (aug_smooth=%s eigen_smooth=%s)", aug_smooth, eigen_smooth)

    # pytorch_grad_cam 内部需要梯度（即使是 ScoreCAM 也走 forward），保持默认即可
    grayscale_cam: np.ndarray = cam(
        input_tensor=img.input_tensor,
        targets=targets,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth,
    )
    cam_2d = grayscale_cam[0]  # (H,W)

    # 9. 保存
    out_path = Path(cfg.get("output", {}).get("path", "outputs/result.png"))
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    save_grid(img.rgb_float, cam_2d, out_path)
    log.info("done. method=%s class=%d -> %s", method, target_class, out_path)


if __name__ == "__main__":
    main()
