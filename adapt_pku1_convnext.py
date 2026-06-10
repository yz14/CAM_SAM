"""把 CAM 适配到自训练的 ConvNeXt-Base 内镜分类模型 (pku1_convnext_base).

背景
----
权重位于 ``D:/codes/work-projects/CAM/pku1_convnext_base/best_model.pth``，
训练框架基于 torchvision ``convnext_base``，但将 ``model.classifier`` 末端的
线性层替换为一个 MLP head（embedding_dim=512, dropout=0.2, num_classes=22）。

通过对 checkpoint 的 state_dict 推断出的网络等价结构：

    model = torchvision.models.convnext_base(weights=None)
    model.classifier = nn.Sequential(
        LayerNorm2d(1024),                # classifier.0 (来自 torchvision 原模型)
        nn.Flatten(1),                    # classifier.1
        nn.Sequential(                    # classifier.2 = MLP head
            nn.Linear(1024, 512),         #   .0
            nn.GELU(),                    #   .1
            nn.Dropout(0.2),              #   .2
            nn.Linear(512, 22),           #   .3
        ),
    )

CAM 目标层选 ``model.features[-1][-1].block[0]`` —— 最后一个 CNBlock **内部**
的 depthwise Conv2d 输出，**不是** CNBlock 整体输出。原因：

- torchvision ConvNeXt 的 CNBlock 形如 ``out = x + layer_scale * mlp(dwconv(x))``，
  其中 ``layer_scale ≈ 0.01`` 且 mlp 中含 GELU/Linear，``out`` 被恒等残差主导。
- 选 CNBlock 整体输出会让 **梯度在空间维度上几乎为常数**：
  GradCAM / XGradCAM / HiResCAM 退化成同一个公式（实测 14 位完全一致），
  GradCAM++ 因二/三阶梯度分母翻号 → 输出全零（视觉上"什么都没有"）。
- 选 ``block[0]`` 即 depthwise Conv2d 输出，纯 NCHW、无残差稀释，
  也是 jacobgil/pytorch-grad-cam 官方对 ConvNeXt 推荐的锚点。

使用方式
--------
1) 独立运行（推荐，自包含）::

    D:\\miniconda\\envs\\py310\\python.exe adapt_pku1_convnext.py \\
        --image D:/path/to/img.jpg \\
        --method gradcam \\
        --output outputs/pku1_gradcam.png

   一次跑多种算法对比::

    D:\\miniconda\\envs\\py310\\python.exe adapt_pku1_convnext.py \\
        --image D:/path/to/img.jpg --all-methods \\
        --output outputs/pku1_compare

2) 通过统一 runner（yaml 入口）::

    model:
      source: custom
      entrypoint: adapt_pku1_convnext:build_pku1_convnext
      entrypoint_kwargs: { num_classes: 22, embedding_dim: 512, dropout: 0.2 }
      weights_path: D:/codes/work-projects/CAM/pku1_convnext_base/best_model.pth
      weights_strict: true
      target_layer: "features.7.2.block.0"   # 最后 CNBlock 内的 depthwise Conv2d

注意事项
--------
- 训练时 ``image_size: 224``，且 ``pretrained: True``（ImageNet 预训练初始化），
  因此推断阶段沿用 ImageNet 标准 mean/std；如训练里另有 normalize 配置请覆盖。
- 加载权重时使用 ``strict=True`` 验证结构完全匹配，定位结构错误更容易。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
import json

# 复用 core/ 中的 IO、可视化、CAM 工厂；保持与 cam_runner.py 一致的视觉风格
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.cam_factory import build_cam, list_methods  # noqa: E402
from core.image_io import load_image  # noqa: E402
from core.logger import get_logger, set_level  # noqa: E402
from core.visualize import save_grid  # noqa: E402

log = get_logger("pku1")


# ---------- 模型构建 ----------

# DEFAULT_WEIGHTS = _HERE / "pku1_convnext_base" / "best_model.pth"
DEFAULT_WEIGHTS = "/data0/yzhen/projects/endoscope_pku1/output/pku1_convnext_base/best_model.pth"
DEFAULT_NUM_CLASSES = 22
DEFAULT_EMBED_DIM = 512
DEFAULT_DROPOUT = 0.2


def build_pku1_convnext(
    num_classes: int = DEFAULT_NUM_CLASSES,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Module:
    """重建训练时的 ConvNeXt-Base + MLP-head 结构（不含权重）。

    供 ``cam_runner.py`` 的 ``model.source: custom`` 通过 entrypoint 调用，
    也供本文件 ``main()`` 内部使用。
    """
    model = tvm.convnext_base(weights=None)
    in_features = 1024  # ConvNeXt-Base 末端通道数；torchvision 把它写在 classifier[2].in_features
    # 替换 classifier[2]（原 Linear(1024, 1000)）为 MLP head；保留 [0]=LayerNorm2d, [1]=Flatten
    mlp = nn.Sequential(
        nn.Linear(in_features, embedding_dim),  # classifier.2.0
        nn.GELU(),                              # classifier.2.1（无参数；与 ReLU/SiLU 加载等价）
        nn.Dropout(dropout),                    # classifier.2.2（无参数）
        nn.Linear(embedding_dim, num_classes),  # classifier.2.3
    )
    model.classifier[2] = mlp
    return model


def _extract_state_dict(obj):
    """兼容 {'model_state_dict': sd} / {'state_dict': sd} / 纯 state_dict / nn.Module。"""
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise TypeError(f"无法识别的 checkpoint 类型: {type(obj)}")


def load_pku1_convnext(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """构建并加载权重；返回 eval() 后的模型，已搬到 device。"""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    model = build_pku1_convnext(num_classes, embedding_dim, dropout)
    log.info("loading weights: %s", weights_path)
    ckpt = torch.load(str(weights_path), map_location="cpu")
    sd = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        log.warning("missing keys (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        log.warning("unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("convnext_base+MLP-head params=%.2fM num_classes=%d device=%s", n_params, num_classes, device)
    return model.eval().to(device)


def get_target_layers(model: nn.Module) -> list[nn.Module]:
    """ConvNeXt 推荐目标层：最后一个 CNBlock **内部** 的 depthwise Conv2d。

    选 CNBlock 整体输出会让残差主导梯度信号，使 GradCAM/XGradCAM/HiResCAM
    退化成同一公式，且 GradCAM++ 输出全零。详见模块 docstring 的解释。
    """
    return [model.features[-1][-1].block[0]]


def _min_window(marg: np.ndarray, frac: float) -> tuple[int, int]:
    """1D：返回覆盖 frac 比例能量的最短区间 [l, r]（含端点）。"""
    total = float(marg.sum())
    if total <= 0:
        return 0, len(marg) - 1
    target = frac * total
    n = len(marg)
    best_l, best_r, best_len = 0, n - 1, n
    s, l = 0.0, 0
    for r in range(n):
        s += float(marg[r])
        while s >= target and l <= r:
            if (r - l) < best_len:
                best_len, best_l, best_r = r - l, l, r
            s -= float(marg[l])
            l += 1
    return best_l, best_r


def cam_to_boxes(
    grayscale_cam: np.ndarray,
    orig_size: tuple[int, int],
    thresh_mode: str = "percentile",   # "percentile" | "ratio" | "abs" | "otsu"
    thresh_value: float = 80.0,        # percentile:分位(80=留前20%); ratio:×max(0~1); abs:0~1
    energy_keep: float | None = 0.7,   # 收缩框到只含该比例激活能量；None=不收缩。最直接的松紧钮
    morph_open: int = 3,               # 开运算核：先腐蚀去 halo/毛刺；<=1 关
    morph_close: int = 3,              # 闭运算核：后填洞；<=1 关
    interp: int = cv2.INTER_LINEAR,    # 上采样插值：LINEAR/NEAREST 的 halo 比 CUBIC 小
    min_area_ratio: float = 0.005,
    pad_ratio: float = 0.0,            # 框外扩比例，可为负→向内收
    topk: int | None = 1,
) -> list[tuple[int, int, int, int]]:
    """CAM 热图 -> 原图坐标系 bbox 列表 [(x0,y0,x1,y1), ...]。"""
    orig_w, orig_h = orig_size
    cam = cv2.resize(grayscale_cam.astype(np.float32), (orig_w, orig_h), interpolation=interp)
    cam = np.clip(cam, 0.0, 1.0)
    cam_u8 = (cam * 255).astype(np.uint8)

    # --- 阈值化 ---
    if thresh_mode == "otsu":
        _, mask = cv2.threshold(cam_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_mode == "percentile":
        t = int(np.percentile(cam_u8, thresh_value))
        _, mask = cv2.threshold(cam_u8, t, 255, cv2.THRESH_BINARY)
    elif thresh_mode == "abs":
        _, mask = cv2.threshold(cam_u8, int(thresh_value * 255), 255, cv2.THRESH_BINARY)
    else:  # ratio
        _, mask = cv2.threshold(cam_u8, int(thresh_value * int(cam_u8.max())), 255, cv2.THRESH_BINARY)

    # --- 形态学：先 open 去毛刺/halo，再 close 填洞 ---
    if morph_open and morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close and morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    img_area = orig_w * orig_h
    cand = []
    for i in range(1, n):  # 0 是背景
        x, y, w, h, area = stats[i]
        if area < min_area_ratio * img_area:
            continue
        score = float(cam[labels == i].mean())
        cand.append((score, i, int(x), int(y), int(w), int(h)))
    if not cand:
        return []
    cand.sort(key=lambda c: c[0], reverse=True)
    if topk is not None:
        cand = cand[:topk]

    boxes = []
    for _, i, x, y, w, h in cand:
        # 能量收缩：在该连通域 bbox 内，把框收到只含 energy_keep 的激活能量
        if energy_keep is not None and 0.0 < energy_keep < 1.0:
            sub = cam[y:y + h, x:x + w].copy()
            sub[labels[y:y + h, x:x + w] != i] = 0.0      # 只统计本连通域的能量
            cl, cr = _min_window(sub.sum(axis=0), energy_keep)  # 列向 -> x 范围
            rl, rr = _min_window(sub.sum(axis=1), energy_keep)  # 行向 -> y 范围
            x, w = x + cl, (cr - cl + 1)
            y, h = y + rl, (rr - rl + 1)

        px, py = int(round(w * pad_ratio)), int(round(h * pad_ratio))
        x0 = max(0, x - px);          y0 = max(0, y - py)
        x1 = min(orig_w, x + w + px); y1 = min(orig_h, y + h + py)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def save_box_artifacts(
    image_path: Path,
    boxes: list[tuple[int, int, int, int]],
    pred_class: int,
    box_root: Path,
) -> None:
    """画框可视化 + 写出 SAM3 可直接读取的 bbox(JSON)。"""
    stem = image_path.stem
    bgr = cv2.imread(str(image_path))          # 原图、原分辨率(boxes 已在此坐标系)
    if bgr is None:
        log.warning("cv2 读不到图，跳过画框: %s", image_path)
        return
    oh, ow = bgr.shape[:2]

    # 1) 可视化
    vis = bgr.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)  # BGR 红框
    vis_path = box_root / "vis" / f"{stem}.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), vis)

    # 2) SAM3 用的 bbox JSON
    rec = {
        "image": str(image_path),
        "image_size_wh": [ow, oh],
        "pred_class": int(pred_class),
        "boxes_xyxy": [[int(x0), int(y0), int(x1), int(y1)] for (x0, y0, x1, y1) in boxes],
    }
    json_path = box_root / "boxes" / f"{stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("[boxes] %s -> %d 个框, %s", stem, len(boxes), json_path)


# ---------- 单次 CAM 计算（独立运行入口使用） ----------

def run_single_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    method: str,
    target_class: int,
    target_layers: Sequence[nn.Module],
    out_path: Path,
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
    extra: dict | None = None,
) -> tuple[Path, np.ndarray]:
    cam = build_cam(method, model, list(target_layers), reshape_transform=None, extra=dict(extra or {}))
    targets = [ClassifierOutputTarget(target_class)]
    grayscale: np.ndarray = cam(
        input_tensor=image_tensor,
        targets=targets,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_grid(rgb_float, grayscale[0], out_path)
    log.info("[%s] class=%d -> %s", method, target_class, out_path)
    return out_path, grayscale[0]


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CAM adapter for pku1_convnext_base")
    p.add_argument("--image", required=True, type=str, help="输入图像路径")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                   help="checkpoint 路径（默认 pku1_convnext_base/best_model.pth）")
    p.add_argument("--method", type=str, default="gradcam",
                   help=f"CAM 算法名，可选: {', '.join(list_methods())}")
    p.add_argument("--all-methods", action="store_true",
                   help="对常用算法各跑一遍并把结果落到 --output 指定的目录")
    p.add_argument("--include-slow", action="store_true",
                   help="仅 --all-methods 时生效：额外跑 AblationCAM / ScoreCAM（扰动式，显著变慢）")
    p.add_argument("--ablation-batch-size", type=int, default=32,
                   help="AblationCAM 的 batch_size，显存不够时调小")
    p.add_argument("--target-class", type=int, default=None,
                   help="指定类别索引；不传则使用模型预测最大类")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="outputs/pku1_convnext_gradcam.png",
                   help="单方法时为图片路径；--all-methods 时为输出目录")
    p.add_argument("--aug-smooth", action="store_true")
    p.add_argument("--eigen-smooth", action="store_true")
    p.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBED_DIM)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-box", action="store_true", help="关闭 bbox 生成")
    p.add_argument("--box-method", type=str, default="gradcam++")
    p.add_argument("--box-thresh-mode", type=str, default="percentile",
                   choices=["percentile", "ratio", "abs", "otsu"])
    p.add_argument("--box-thresh-value", type=float, default=80.0,
                   help="percentile:分位(80=留前20%%); ratio:×max(0~1); abs:0~1; otsu 时忽略")
    p.add_argument("--box-energy-keep", type=float, default=0.7,
                   help="收缩框到只含该比例激活能量；<=0 或 >=1 关闭。越小框越紧")
    p.add_argument("--box-morph-open", type=int, default=3, help="开运算核去 halo；<=1 关")
    p.add_argument("--box-morph-close", type=int, default=3, help="闭运算核填洞；<=1 关")
    p.add_argument("--box-interp", type=str, default="linear",
                   choices=["linear", "nearest", "cubic"], help="上采样插值；cubic halo 最大")
    p.add_argument("--box-min-area-ratio", type=float, default=0.005)
    p.add_argument("--box-pad-ratio", type=float, default=0.0, help="可为负→向内收")
    p.add_argument("--box-topk", type=int, default=1, help="<=0 表示全部")
    return p.parse_args()


# --all-methods 默认跑的算法列表；ablationcam/scorecam 需加 --include-slow 才启用
_FAST_METHODS = (
    "gradcam", "gradcam++", "xgradcam", "hirescam",
    "gradcam_elementwise", "layercam", "eigencam", "eigengradcam", "fullgrad",
)
_SLOW_METHODS = ("ablationcam", "scorecam")


def main() -> None:
    args = parse_args()
    if args.debug:
        set_level("DEBUG")
        
    # ---- 新增：收集待处理图片列表 ----
    img_root = Path(args.image)
    if img_root.is_dir():
        _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = sorted(p for p in img_root.iterdir() if p.suffix.lower() in _EXTS)
        log.info("folder mode: found %d images in %s", len(image_paths), img_root)
    else:
        image_paths = [img_root]

    # 1. 模型
    model = load_pku1_convnext(
        weights_path=args.weights,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        device=args.device,
        strict=True,
    )
    target_layers = get_target_layers(model)
    log.info("target_layer = model.features[-1][-1].block[0]  (%s)", type(target_layers[0]).__name__)
    
    # 2. 输出根路径
    out_arg = Path(args.output)
    if not out_arg.is_absolute():
        out_arg = _HERE / out_arg
        box_root = out_arg if out_arg.suffix == "" else out_arg.parent  # NEW

    # 3. 逐图处理（for 循环包住所有图片相关逻辑）
    for image_path in image_paths:
        img = load_image(
            str(image_path),
            image_size=args.image_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            device=args.device,
        )

        with torch.no_grad():
            logits = model(img.input_tensor)
        probs = logits.softmax(dim=1)[0]
        pred = int(logits.argmax(dim=1).item())
        log.info("[%s] pred=%d  logit=%.3f  prob=%.3f",
                 image_path.name, pred, float(logits[0, pred]), float(probs[pred]))
        target_class = args.target_class if args.target_class is not None else pred

        stem = image_path.stem  # 原文件名（不含扩展名），用作输出文件名
        
        # ---- 生成 bbox(供 SAM3) ----
        if not args.no_box:
            box_heatmap_path = box_root / args.box_method / f"{stem}.png"
            _, gcam = run_single_cam(
                model, img.input_tensor, img.rgb_float,
                method=args.box_method, target_class=target_class,
                target_layers=target_layers, out_path=box_heatmap_path,
                aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth,
            )
            _INTERP = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}
            bgr0 = cv2.imread(str(image_path))
            oh, ow = bgr0.shape[:2]
            ek = args.box_energy_keep
            boxes = cam_to_boxes(
                gcam, orig_size=(ow, oh),
                thresh_mode=args.box_thresh_mode,
                thresh_value=args.box_thresh_value,
                energy_keep=(None if (ek <= 0 or ek >= 1) else ek),
                morph_open=args.box_morph_open,
                morph_close=args.box_morph_close,
                interp=_INTERP[args.box_interp],
                min_area_ratio=args.box_min_area_ratio,
                pad_ratio=args.box_pad_ratio,
                topk=(None if args.box_topk <= 0 else args.box_topk),
            )
            save_box_artifacts(image_path, boxes, target_class, box_root)

        if args.all_methods:
            methods_to_run = list(_FAST_METHODS)
            if args.include_slow:
                methods_to_run += list(_SLOW_METHODS)
            for m in methods_to_run:
                extra = {"batch_size": args.ablation_batch_size} if m == "ablationcam" else None
                try:
                    run_single_cam(
                        model, img.input_tensor, img.rgb_float,
                        method=m, target_class=target_class, target_layers=target_layers,
                        out_path=out_arg / m / f"{stem}.png",
                        aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth,
                        extra=extra,
                    )
                except Exception as e:
                    log.exception("method %s / image %s failed: %s", m, stem, e)
        else:
            out_path = out_arg / args.method / f"{stem}.png" if len(image_paths) > 1 else out_arg
            run_single_cam(
                model, img.input_tensor, img.rgb_float,
                method=args.method, target_class=target_class, target_layers=target_layers,
                out_path=out_path,
                aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth,
            )

    log.info("all done -> %s", out_arg)


if __name__ == "__main__":
    main()


# 方法	用法定位	一句话
# GradCAM	经典基线	通道权重 = 空间均值梯度。最被广泛引用，default 信哪个就信它。
# GradCAM++	多实例/小目标	用二/三阶梯度强调"集中正贡献"的通道，多个目标场景比 GradCAM 更稳。
# HiResCAM	忠实度	元素级 grad×act，理论上线性近似下 ∑=logit，最不"骗你"。
# XGradCAM	类别可分性	重新归一化让 ∑(w·A) 接近 logit，类别判别性更好。
# LayerCAM	浅层/多层融合	ReLU(grad)×act 元素级，浅层也有效。
# GradCAMElementWise	HiResCAM 变体	只保留正贡献，去掉 mean。
# EigenCAM	类无关显著性	不要用来解释类别决策。
# EigenGradCAM	EigenCAM + 类别	把 EigenCAM 拉回类别敏感版本。
# FullGrad	全网络解释	聚合所有 bias 层；视为全局 sanity check。
# AblationCAM / ScoreCAM	慢但可信	扰动式，接近"ground truth"，但开销大 ≈10–50×。


# D:\miniconda\envs\py310\python.exe adapt_pku1_convnext.py \
#     --image D:/path/to/your/image_folder \
#     --all-methods \
#     --output D:/path/to/outputs/pku1_compare

# D:\miniconda\envs\py310\python.exe adapt_pku1_convnext.py \
#     --image D:/path/to/your/image_folder \
#     --method gradcam \
#     --output D:/path/to/outputs/pku1_compare

# D:\miniconda\envs\py310\python.exe adapt_pku1_convnext.py \
#     --image D:/path/to/your/image_folder \
#     --all-methods \
#     --include-slow \
#     --output D:/path/to/outputs/pku1_compare

# box-energy-keep越大框越大  box-thresh-value越小框越大
# python adapt_pku1_convnext.py --image /data0/yzhen/data/endoscope_pku1/1 --all-methods --output ./pku1/1 --box-method gradcam++ --box-energy-keep 0.85 --box-thresh-value 70 --box-interp linear