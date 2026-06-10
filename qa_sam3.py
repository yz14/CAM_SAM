"""
SAM3 内镜分割结果质检（无 GT 的自动 triage）
=============================================

读取 predict_endoscope_sam3.py 的输出（mask/ + 对应 boxes/），计算一组
几何 / 一致性「代理指标」，输出 qa_report.csv，并按可疑度排序打印需要
重点人工复核的样本。

重要定位
--------
这些指标是**代理质量**，不能替代真实 GT 的 Dice。它的作用是把「无法主观
判断的一大堆结果」压缩成「只需重点看被 flag 的少数几张」。clean 的样本不
代表一定正确，只代表没有触发明显错误模式。

用法
----
    python qa_endoscope_sam3.py \\
        --mask  ./sam3_out/1/mask \\
        --boxes /data0/yzhen/projects/CAM/pku1/1/boxes \\
        --image /data0/yzhen/data/endoscope_pku1/1 \\   # 可选，给了才算 mask 内亮度
        --meta  ./sam3_out/1/meta \\                     # 可选，读 SAM3 score
        --output ./sam3_out/1/qa_report.csv
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---- flag 阈值（启发式，按你的数据微调）----
TH = dict(
    in_box_frac_min=0.50,     # mask 落在框内比例下限：低=漏到框外
    box_fill_min=0.03,        # 框填充率下限：低=只抠出一条缝
    area_min=0.0015,          # mask 占图比例下限：低=近乎空
    area_max=0.60,            # mask 占图比例上限：高=吞了整片视野
    n_comp_max=3,             # 连通域数上限：多=碎
    border_max=0.25,          # 边界接触比例上限：高=抓到黑边/镜头圆环
    solidity_min=0.50,        # solidity 下限：低=形状散乱/凹陷
    mean_int_min=25.0,        # mask 内平均亮度下限（0~255）：低=抓到暗角
    score_min=0.40,           # SAM3 confidence 下限（若有 meta）
)


def load_mask(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (m > 127)


def boxes_to_mask(shape, boxes) -> np.ndarray:
    H, W = shape
    bm = np.zeros((H, W), dtype=bool)
    for (x0, y0, x1, y1) in boxes:
        bm[int(y0):int(y1), int(x0):int(x1)] = True
    return bm


def compute_metrics(mask: np.ndarray, boxes, image_bgr=None) -> dict:
    H, W = mask.shape
    area = int(mask.sum())
    m = dict(mask_area_frac=area / (H * W))
    if area == 0:
        m.update(in_box_frac=0.0, box_fill_frac=0.0, n_comp=0,
                 border_frac=0.0, solidity=0.0, mean_int=0.0)
        return m

    bm = boxes_to_mask((H, W), boxes)
    inter = int(np.logical_and(mask, bm).sum())
    m["in_box_frac"] = inter / area
    m["box_fill_frac"] = inter / max(1, int(bm.sum()))

    m8 = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    m["n_comp"] = sum(1 for i in range(1, n)
                      if stats[i, cv2.CC_STAT_AREA] >= 0.02 * area)

    bw = max(2, int(0.02 * min(H, W)))
    ring = np.zeros_like(mask)
    ring[:bw, :] = ring[-bw:, :] = ring[:, :bw] = ring[:, -bw:] = True
    m["border_frac"] = int(np.logical_and(mask, ring).sum()) / area

    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts) or 1.0
    m["solidity"] = area / hull_area

    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        m["mean_int"] = float(gray[mask].mean())
    else:
        m["mean_int"] = -1.0
    return m


def flags_for(m: dict, score: float | None) -> list[str]:
    f = []
    if m["in_box_frac"] < TH["in_box_frac_min"]:   f.append("漏到框外")
    if m["box_fill_frac"] < TH["box_fill_min"]:    f.append("只抠出一条缝")
    if m["mask_area_frac"] < TH["area_min"]:       f.append("近乎空")
    if m["mask_area_frac"] > TH["area_max"]:       f.append("吞了整片视野")
    if m["n_comp"] > TH["n_comp_max"]:             f.append("碎成多块")
    if m["border_frac"] > TH["border_max"]:        f.append("贴边/抓到黑环")
    if m["solidity"] < TH["solidity_min"]:         f.append("形状散乱")
    if 0 <= m["mean_int"] < TH["mean_int_min"]:    f.append("区域过暗(疑黑边)")
    if score is not None and score < TH["score_min"]: f.append(f"低置信({score:.2f})")
    return f


def run(mask_dir, boxes_dir, image_dir, meta_dir, output):
    mask_dir = Path(mask_dir)
    boxes_dir = Path(boxes_dir)
    image_dir = Path(image_dir) if image_dir else None
    meta_dir = Path(meta_dir) if meta_dir else None

    def find_image(stem):
        if not image_dir:
            return None
        for ext in _IMG_EXTS:
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    rows = []
    for mp in sorted(mask_dir.glob("*.png")):
        stem = mp.stem
        jp = boxes_dir / f"{stem}.json"
        if not jp.exists():
            print(f"  [skip] 无 boxes JSON: {stem}")
            continue
        boxes = json.loads(jp.read_text(encoding="utf-8")).get("boxes_xyxy", [])
        mask = load_mask(mp)

        img_bgr = None
        ip = find_image(stem)
        if ip is not None:
            img_bgr = cv2.imread(str(ip))
            if img_bgr is not None and img_bgr.shape[:2] != mask.shape:
                img_bgr = cv2.resize(img_bgr, (mask.shape[1], mask.shape[0]))

        score = None
        if meta_dir is not None and (meta_dir / f"{stem}.json").exists():
            score = json.loads((meta_dir / f"{stem}.json").read_text()).get("score")

        m = compute_metrics(mask, boxes, img_bgr)
        fl = flags_for(m, score)
        rows.append(dict(stem=stem, n_flag=len(fl), flags="; ".join(fl),
                         score=("" if score is None else round(score, 3)),
                         **{k: round(v, 4) for k, v in m.items()}))

    if not rows:
        print("没有可质检的样本"); return

    rows.sort(key=lambda r: r["n_flag"], reverse=True)
    fields = ["stem", "n_flag", "flags", "score", "mask_area_frac", "in_box_frac",
              "box_fill_frac", "n_comp", "border_frac", "solidity", "mean_int"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    n_flagged = sum(1 for r in rows if r["n_flag"] > 0)
    print(f"\n共 {len(rows)} 张，其中 {n_flagged} 张触发 flag（建议优先人工复核）：")
    for r in rows:
        if r["n_flag"] == 0:
            break
        print(f"  [{r['n_flag']}] {r['stem']:40s} -> {r['flags']}")
    print(f"\n报告已写出: {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("SAM3 内镜分割结果质检（无 GT）")
    ap.add_argument("--mask", required=True, help="mask/ 目录")
    ap.add_argument("--boxes", required=True, help="boxes/ 目录")
    ap.add_argument("--image", default=None, help="原图目录（可选，算 mask 内亮度）")
    ap.add_argument("--meta", default=None, help="meta/ 目录（可选，读 SAM3 score）")
    ap.add_argument("--output", default="qa_report.csv")
    args = ap.parse_args()
    run(args.mask, args.boxes, args.image, args.meta, args.output)
    

# python qa_sam3.py \
#     --mask  ./sam3_out/1/mask \
#     --boxes /data0/yzhen/projects/CAM/pku1/1/boxes \
#     --image /data0/yzhen/data/endoscope_pku1/1 \
#     --output ./sam3_out/1/qa_report.csv