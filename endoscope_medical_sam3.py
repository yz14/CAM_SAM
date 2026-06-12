"""Medical-SAM3 内镜分割预测（医学概念文本提示版）
=================================================

Medical-SAM3（https://github.com/AIM-Research-Lab/Medical-SAM3，arXiv 2601.10880）
= SAM3 基座 + 大规模医学数据**全量微调**（不是 LoRA 增量）。它和
Joey-S-Liu/MedSAM3 一样是「概念驱动」的医学分割模型，但权重组织方式不同：

- MedSAM3 (endoscope_medsam3.py)：通用 SAM3 基座 + 单独下载的 LoRA 增量，
  推理时要先 inject LoRA 结构再加载增量权重。
- Medical-SAM3 (本脚本)：直接发布一份完整微调过的 SAM3 权重，**加载方式与
  endoscope_sam3.py / endoscope_sam3_v2.py 完全一样**——build_sam3_image_model
  指向 Medical-SAM3 的 checkpoint 即可，无需 LoRA。

为保证 checkpoint（来自视频训练管线）能被正确加载/转换，本脚本用
**Medical-SAM3 仓库自带的 sam3 包**（--medical-sam3-repo 指向克隆目录，
脚本启动时把它插到 sys.path 最前，使所有 ``import sam3`` 都解析到这份）。

本脚本与 endoscope_medsam3.py 的 IO / 推理逻辑完全对齐（共用
seg_concept_predict.predict_text_only / predict_with_boxes），方便和 SAM3、
MedSAM3 做 A/B 对比，看哪个对结肠息肉效果更好。

输入
----
1) 内镜原图：单张图片或图片文件夹
2) （可选）bbox JSON：CAM 产出的 boxes/<stem>.json；给了就用框筛选/锚定
   文本检出的实例，不给则纯文本模式输出全部检出实例

输出
----
<out>/mask/<stem>.png      二值 mask（0/255，原图分辨率）
<out>/overlay/<stem>.png   原图 + 半透明 mask + 绿色提示框
<out>/meta/<stem>.json     score / 候选数（qa_sam3.py --meta 可读）

两种工作模式
------------
1. 纯文本模式（不传 --boxes）::

       python endoscope_medical_sam3.py \\
           --image /data0/yzhen/data/endoscope_pku1/1 \\
           --output ./medical_sam3_out/1 \\
           --medical-sam3-repo /data0/yzhen/projects/Medical-SAM3 \\
           --checkpoint /data0/yzhen/data/medical_sam3/checkpoint.pt \\
           --text-prompt "colon polyp"

2. 文本 + CAM 框联合模式（传 --boxes，行为对齐 endoscope_sam3_v2.py）::

       python endoscope_medical_sam3.py \\
           --image  /data0/yzhen/data/endoscope_pku1/1 \\
           --boxes  /data0/yzhen/projects/CAM/pku1/1/boxes \\
           --output ./medical_sam3_out/1 \\
           --medical-sam3-repo /data0/yzhen/projects/Medical-SAM3 \\
           --checkpoint /data0/yzhen/data/medical_sam3/checkpoint.pt

安装与权重下载见 docs/medical_sam3_usage.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# 引导：在 import sam3 之前把 Medical-SAM3 仓库插到 sys.path 最前
# ──────────────────────────────────────────────
#
# Medical-SAM3 自带一份（与官方 SAM3 兼容、且包含 checkpoint state_dict 转换逻辑的）
# sam3 包。下游 import（seg_concept_predict -> endoscope_sam3_v2 -> sam3）都在模块顶层
# 触发，所以必须在那之前完成 sys.path 注入，故这里先轻量扫一遍 argv 拿到 --medical-sam3-repo。

def _early_arg(*names: str) -> str | None:
    """在正式 argparse 之前从 sys.argv 里取某个选项的值（支持 --k v 与 --k=v）。"""
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        for name in names:
            if tok == name and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(name + "="):
                return tok.split("=", 1)[1]
    return None


def _bootstrap_medical_sam3_repo() -> None:
    repo = _early_arg("--medical-sam3-repo")
    if not repo:
        return  # 没给就在正式 argparse 阶段报必填错误
    repo_path = Path(repo).expanduser()
    if not (repo_path / "sam3" / "model_builder.py").exists():
        raise FileNotFoundError(
            f"{repo_path} 下找不到 sam3/model_builder.py，请确认 --medical-sam3-repo "
            f"指向克隆好的 https://github.com/AIM-Research-Lab/Medical-SAM3 目录")
    if str(repo_path) in sys.path:
        sys.path.remove(str(repo_path))
    sys.path.insert(0, str(repo_path))


_bootstrap_medical_sam3_repo()

import torch  # noqa: E402  （必须在 sys.path 注入之后再 import 依赖 sam3 的模块）
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

from seg_common import (  # noqa: E402
    check_size_consistency,
    collect_pairs,
    load_boxes_json,
    save_results,
)
from seg_concept_predict import predict_text_only, predict_with_boxes  # noqa: E402


# ──────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────

def _resolve_bpe_path(checkpoint: str, medical_sam3_repo: str,
                      bpe_path: str | None) -> str | None:
    """BPE 词表定位：--bpe-path > checkpoint 同目录 > 仓库 assets/ > 让 builder 用默认。"""
    if bpe_path:
        return bpe_path
    name = "bpe_simple_vocab_16e6.txt.gz"
    for cand in (Path(checkpoint).parent / name,
                 Path(medical_sam3_repo).expanduser() / "assets" / name):
        if cand.exists():
            return str(cand)
    return None  # build_sam3_image_model 会回退到包内 assets 默认


def build_medical_sam3(checkpoint: str, medical_sam3_repo: str,
                       bpe_path: str | None, device: str):
    """构建 Medical-SAM3：直接加载全量微调权重（无 LoRA）。"""
    if not Path(checkpoint).expanduser().exists():
        raise FileNotFoundError(f"找不到 Medical-SAM3 权重: {checkpoint}")
    resolved_bpe = _resolve_bpe_path(checkpoint, medical_sam3_repo, bpe_path)

    print(f"[1/2] 构建 Medical-SAM3（全量微调 SAM3）: {checkpoint}")
    model = build_sam3_image_model(
        checkpoint_path=str(Path(checkpoint).expanduser()),
        bpe_path=resolved_bpe,
        device=device,
        load_from_HF=False,
        eval_mode=True,
    )
    model.eval()
    return model


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(args) -> None:
    model = build_medical_sam3(args.checkpoint, args.medical_sam3_repo,
                               args.bpe_path, args.device)
    processor = Sam3Processor(model, device=args.device,
                              confidence_threshold=args.conf_threshold)

    pairs = collect_pairs(args.image, args.boxes)
    mode = "text+box" if args.boxes else "text_only"
    sel = (f"选法=框内 argmax(coverage×precision) min_iou>{args.min_iou} min_prec>{args.min_precision}"
           if args.boxes else f"conf>{args.conf_threshold}")
    print(f"[2/2] 推理  mode={mode}  prompt={args.text_prompt!r}  "
          f"{sel}  mask>{args.mask_threshold}  共 {len(pairs)} 张"
          f"{'  [debug]' if args.debug else ''}")

    out_root = Path(args.output)
    for image_path, json_path in tqdm(pairs, desc="Medical-SAM3"):
        boxes = []
        if json_path is not None:
            boxes, size = load_boxes_json(json_path)
            check_size_consistency(image_path, size, warn_fn=tqdm.write)

        try:
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode():
                if boxes:
                    mask, meta = predict_with_boxes(
                        processor, image, boxes, args.text_prompt,
                        args.expand_ratio, args.min_iou, args.min_precision,
                        args.max_mask_frac, args.keep_components,
                        name=image_path.name, debug=args.debug)
                else:
                    mask, meta = predict_text_only(
                        processor, image, args.text_prompt,
                        args.mask_threshold, args.keep_components)
        except Exception as e:
            tqdm.write(f"  {image_path.name} 推理异常: {e}")
            continue

        meta["image"] = str(image_path)
        save_results(out_root, image_path.stem, image, mask, boxes,
                     meta=meta, save_overlay_flag=not args.no_overlay)

    print(f"完成 -> {out_root}")
    print(f"建议质检: python qa_sam3.py --mask {out_root}/mask "
          + (f"--boxes {args.boxes} " if args.boxes else "")
          + f"--image {args.image} --meta {out_root}/meta --output {out_root}/qa_report.csv")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Medical-SAM3 内镜分割（医学概念文本提示）")
    ap.add_argument("--image", required=True, help="内镜原图：单张图片或图片文件夹")
    ap.add_argument("--boxes", default=None,
                    help="（可选）CAM bbox JSON：文件或 boxes/ 目录；给了走 文本+框 模式")
    ap.add_argument("--output", default="medical_sam3_out", help="输出根目录")
    ap.add_argument("--medical-sam3-repo", required=True,
                    help="Medical-SAM3 仓库克隆目录（提供自带的 sam3 包，启动时插到 sys.path 最前）")
    ap.add_argument("--checkpoint", required=True,
                    help="Medical-SAM3 全量微调权重 .pt 路径（HF 下载，见 docs/medical_sam3_usage.md）")
    ap.add_argument("--bpe-path", default=None,
                    help="（可选）BPE 词表；不给则按 checkpoint 同目录 / 仓库 assets / 包内默认 依次找")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-prompt", default="colon polyp",
                    help='医学概念词（英文），如 "colon polyp" / "polyp" / "lesion"')
    ap.add_argument("--expand-ratio", type=float, default=0.12,
                    help="（文本+框模式）bbox 外扩比例")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="实例置信度阈值；按任务在 0.5~0.8 之间调")
    ap.add_argument("--mask-threshold", type=float, default=0.5,
                    help="mask 概率二值化阈值")
    ap.add_argument("--min-iou", type=float, default=0.1,
                    help="（文本+框模式）候选框与提示框最小 IoU（廉价空间初筛）")
    ap.add_argument("--min-precision", type=float, default=0.5,
                    help="（文本+框模式）候选 mask 落在框内的最小比例（|mask∩框|/|mask|）；"
                         "默认 0.5，踢掉框外飞溅/覆盖整图的候选；目标被误丢调低 0.3")
    ap.add_argument("--max-mask-frac", type=float, default=0.6,
                    help="（文本+框模式）候选 mask 面积占全图上限（超过视为背景块，丢弃）")
    ap.add_argument("--keep-components", default="overlap",
                    choices=["all", "largest", "overlap"])
    ap.add_argument("--debug", action="store_true",
                    help="（文本+框模式）打印每个框的候选表与选中原因")
    ap.add_argument("--no-overlay", action="store_true", help="不存 overlay 检查图")
    run(ap.parse_args())
