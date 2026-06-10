"""从绿色涂抹的标注图中提取 bbox，输出 SAM3 可用的 JSON。

用法：
    python extract_green_bbox.py --image path/to/bbox_annotated.jpg --output path/to/output_dir

默认把绿色的 HSV 范围设得较宽，覆盖常见绘图软件中的"纯绿"到"黄绿"；
如果提取不全或多提，可调 --green-lower/--green-upper。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def extract_green_boxes(
    image_path: str | Path,
    green_lower: tuple[int, int, int] = (40, 180, 80),
    green_upper: tuple[int, int, int] = (90, 255, 255),
    morph_open: int = 3,
    morph_close: int = 3,
    min_area: int = 50,
) -> list[tuple[int, int, int, int]]:
    """
    用 HSV 阈值提取绿色连通域，返回原图坐标系的 bbox 列表 [(x0,y0,x1,y1), ...]。
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(green_lower), np.array(green_upper))

    # 形态学：先 open 去噪点，再 close 填洞
    if morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # 找到所有绿色像素的最大包围框（合并所有涂抹区域为一个框）
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return [(x0, y0, x1, y1)]


def save_box_artifacts(
    src_image_path: str | Path,
    boxes: list[tuple[int, int, int, int]],
    out_dir: str | Path,
    image_size_wh: tuple[int, int] | None = None,
) -> tuple[Path, Path]:
    """保存 SAM3 格式的 bbox JSON + 原图画框可视化（与 adapt_pku1_convnext.py 一致）。"""
    src_image_path = Path(src_image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(src_image_path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像以获取尺寸: {src_image_path}")
    oh, ow = bgr.shape[:2]
    image_size_wh = image_size_wh or (ow, oh)

    # 1) 画框可视化
    vis = bgr.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)  # BGR 红框
    vis_path = out_dir / "vis" / f"{src_image_path.stem}.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), vis)

    # 2) JSON
    rec = {
        "image": str(src_image_path),
        "image_size_wh": list(image_size_wh),
        "pred_class": 0,
        "boxes_xyxy": [[int(x0), int(y0), int(x1), int(y1)] for (x0, y0, x1, y1) in boxes],
    }
    json_path = out_dir / "boxes" / f"{src_image_path.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"[extract_green_bbox] {len(boxes)} 个框 -> json={json_path}, vis={vis_path}")
    return json_path, vis_path


def main() -> None:
    p = argparse.ArgumentParser("从绿色涂抹图中提取 bbox")
    p.add_argument("--image", required=True, type=str, help="绿色标注图路径（jpg/png 等）")
    p.add_argument("--output", type=str, default="outputs/green_boxes", help="JSON 输出目录")
    p.add_argument("--green-lower", type=int, nargs=3, default=[40, 180, 80], help="HSV 下界 (H,S,V)")
    p.add_argument("--green-upper", type=int, nargs=3, default=[90, 255, 255], help="HSV 上界 (H,S,V)")
    p.add_argument("--morph-open", type=int, default=3, help="开运算核大小；<=1 关")
    p.add_argument("--morph-close", type=int, default=3, help="闭运算核大小；<=1 关")
    p.add_argument("--min-area", type=int, default=50, help="最小连通域面积")
    p.add_argument("--pred-class", type=int, default=0, help="JSON 中的 pred_class 字段")
    args = p.parse_args()

    boxes = extract_green_boxes(
        args.image,
        green_lower=tuple(args.green_lower),
        green_upper=tuple(args.green_upper),
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_area=args.min_area,
    )

    json_path, vis_path = save_box_artifacts(args.image, boxes, args.output)

    # 如果需要覆盖 pred_class，读取-修改-写回
    if args.pred_class != 0:
        with open(json_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["pred_class"] = args.pred_class
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
