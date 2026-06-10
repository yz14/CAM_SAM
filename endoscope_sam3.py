"""
SAM3 内镜分割预测（box prompt 版）
================================

输入
----
1) 内镜原图：单张图片，或一个图片文件夹
2) bbox JSON：由 CAM 脚本 save_box_artifacts() 产出的 boxes/<stem>.json，
   单个 JSON 文件或一个目录（按文件名 stem 与图片自动配对）

JSON 格式（回顾）::

    {
      "image": "/path/to/xxx.jpg",
      "image_size_wh": [W, H],
      "pred_class": 3,
      "boxes_xyxy": [[x0, y0, x1, y1], ...]   # 原图像素坐标
    }

输出
----
<out>/mask/<stem>.png      二值 mask（0/255，原图分辨率）
<out>/overlay/<stem>.png   原图 + 半透明 mask + 绿色提示框（肉眼检查用）

关键约定（已对齐仓库真实接口）
------------------------------
SAM3 的 ``processor.add_geometric_prompt(box=..., label=..., state=...)``
接收的是 **归一化的 [cx, cy, w, h]**（中心点 + 宽高，均除以图像宽/高），
**不是** 像素 xyxy。本脚本负责 xyxy(px) -> cxcywh(norm) 这步转换。

用法
----
单张::

    python predict_endoscope_sam3.py \\
        --image  /path/img.jpg \\
        --boxes  /path/pku1_compare/boxes/img.json \\
        --output /path/sam3_out \\
        --checkpoint /data0/.../ckpt/sam3.pt

批量（图片文件夹 + boxes 目录，按 stem 配对）::

    python predict_endoscope_sam3.py \\
        --image  /path/image_folder \\
        --boxes  /path/pku1_compare/boxes \\
        --output /path/sam3_out
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def xyxy_to_norm_cxcywh(box, W: int, H: int):
    """像素 [x0,y0,x1,y1] -> 归一化 [cx,cy,w,h]（SAM3 add_geometric_prompt 需要的格式）。"""
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0) / W
    cy = ((y0 + y1) / 2.0) / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    return [float(cx), float(cy), float(w), float(h)]


def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def pick_best_mask(masks, scores) -> np.ndarray | None:
    """从候选实例里按 score 取最优，返回二值 (H,W) uint8；无候选返回 None。"""
    masks = to_numpy(masks)
    scores = to_numpy(scores)
    if masks is None or len(masks) == 0:
        return None
    if masks.ndim == 4:           # (N,1,H,W) -> (N,H,W)
        masks = masks[:, 0]
    idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
    return (masks[idx] > 0).astype(np.uint8)


def load_boxes_json(json_path: Path):
    """返回 (boxes_xyxy, size_wh_or_None)。"""
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    boxes = rec.get("boxes_xyxy", [])
    size = rec.get("image_size_wh", None)
    return boxes, size


def save_overlay(image_pil: Image.Image, mask: np.ndarray,
                 boxes_xyxy, out_path: Path, alpha: float = 0.45):
    """原图 + 半透明红色 mask + 绿色提示框。"""
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    m = mask.astype(bool)
    if m.any():
        overlay = bgr.copy()
        overlay[m] = (0, 0, 255)  # 红
        bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    for (x0, y0, x1, y1) in boxes_xyxy:
        cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


# ──────────────────────────────────────────────
# 单图预测
# ──────────────────────────────────────────────

def predict_one(processor: Sam3Processor, image_path: Path, boxes_xyxy, label: bool = True):
    """对一张图、若干个框做 box-prompt 分割；多框 mask 取并集，返回 (PIL图, mask)。"""
    image = Image.open(image_path).convert("RGB")
    W, H = image.size  # 注意 PIL 是 (W,H)；boxes_xyxy 必须与此同坐标系

    union = np.zeros((H, W), dtype=np.uint8)
    for box in boxes_xyxy:
        # 每个框单独 set_image 一次，避免 add_geometric_prompt 把多框累积到同一 state
        # （单框时无额外开销；多框时这样保证各框互不串扰、再 union）
        state = processor.set_image(image)
        nb = xyxy_to_norm_cxcywh(box, W, H)
        output = processor.add_geometric_prompt(box=nb, label=label, state=state)

        mask = pick_best_mask(output.get("masks"), output.get("scores"))
        if mask is None:
            continue
        if mask.shape != (H, W):  # 理论上 SAM3 已插值回原分辨率，做个防御性对齐
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        union = np.logical_or(union, mask).astype(np.uint8)

    return image, union


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(image_arg: str, boxes_arg: str, output: str, checkpoint: str,
        device: str = "cuda", save_overlay_flag: bool = True):

    # 1) 加载模型（参数与你骨头脚本一致）
    print(f"[1/3] 加载 SAM3: {checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        bpe_path=str(Path(checkpoint).parent / "bpe_simple_vocab_16e6.txt.gz"),
        device=device,
        load_from_HF=False,
        eval_mode=True,
    )
    model.eval()
    processor = Sam3Processor(model)

    # 2) 收集 (图片, boxes_json) 配对
    img_path = Path(image_arg)
    box_path = Path(boxes_arg)
    pairs: list[tuple[Path, Path]] = []
    if img_path.is_dir():
        for p in sorted(img_path.iterdir()):
            if p.suffix.lower() not in _IMG_EXTS:
                continue
            j = (box_path / f"{p.stem}.json") if box_path.is_dir() else box_path
            if j.exists():
                pairs.append((p, j))
            else:
                print(f"  [skip] 找不到对应 bbox JSON: {p.name}")
    else:
        j = box_path if box_path.is_file() else (box_path / f"{img_path.stem}.json")
        pairs.append((img_path, j))
    print(f"      配对成功 {len(pairs)} 张")

    out_root = Path(output)

    # 3) 逐图预测
    print("[3/3] 推理")
    for image_path, json_path in tqdm(pairs, desc="SAM3"):
        boxes, size = load_boxes_json(json_path)
        if not boxes:
            tqdm.write(f"  {image_path.name}: boxes 为空，跳过")
            continue

        # JSON 里的尺寸和实际图做个一致性校验（不一致说明 CAM 出框时和现在读的不是同一张/同分辨率）
        actual_wh = Image.open(image_path).size
        if size is not None and tuple(size) != tuple(actual_wh):
            tqdm.write(f"  [warn] {image_path.name} 尺寸不一致 json={size} actual={list(actual_wh)}，"
                       f"框坐标可能错位")

        try:
            image, mask = predict_one(processor, image_path, boxes, label=True)
        except Exception as e:
            tqdm.write(f"  {image_path.name} 推理异常: {e}")
            continue

        stem = image_path.stem
        mask_path = out_root / "mask" / f"{stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

        if save_overlay_flag:
            save_overlay(image, mask, boxes, out_root / "overlay" / f"{stem}.png")

    print(f"完成 -> {out_root}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("SAM3 内镜 box-prompt 分割预测")
    ap.add_argument("--image", required=True, help="内镜原图：单张图片或图片文件夹")
    ap.add_argument("--boxes", required=True, help="bbox JSON：单个文件或 boxes/ 目录")
    ap.add_argument("--output", default="sam3_out", help="输出根目录")
    ap.add_argument("--checkpoint", default="/data0/yzhen/data/sam3_service/ckpt/sam3.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-overlay", action="store_true", help="不存 overlay 检查图")
    args = ap.parse_args()

    run(
        image_arg=args.image,
        boxes_arg=args.boxes,
        output=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
        save_overlay_flag=not args.no_overlay,
    )
    
# python endoscope_sam3.py --image /data0/yzhen/data/endoscope_pku1/1 --boxes /data0/yzhen/projects/CAM/pku1/1/boxes --output ./sam3_out/1 --checkpoint /data0/yzhen/data/sam3_service/ckpt/sam3.pt