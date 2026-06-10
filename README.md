# CAM 学习与实战框架

基于 [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) 封装的、面向"教学 + 可迁移工程化"的 CAM 工具集。

## 目录结构

```
CAM/
├── core/                  # 复用模块（model/image/layers/factory/visualize/logger）
├── configs/               # YAML 配置（统一入口使用）
│   ├── default.yaml       # ResNet34 + GradCAM
│   ├── vit.yaml           # ViT-B/16 + reshape_transform
│   ├── swin.yaml          # Swin-T + reshape_transform
│   ├── custom_model.yaml  # 自定义 nn.Module 接入示例
│   └── README.md          # 配置字段速查
├── custom_models/         # 自写模型示例（供 yaml entrypoint 引用）
│   └── my_simple_cnn.py
├── examples/              # 各算法独立 demo（教学用，少依赖）
│   ├── README.md          # demo 索引 + 决策速查
│   ├── demo_gradcam.py
│   ├── demo_gradcam_plusplus.py
│   ├── demo_xgradcam.py
│   ├── demo_hirescam.py
│   ├── demo_gradcam_elementwise.py
│   ├── demo_layercam.py
│   ├── demo_eigencam.py
│   ├── demo_eigengradcam.py
│   ├── demo_ablationcam.py
│   ├── demo_scorecam.py
│   ├── demo_fullgrad.py
│   ├── demo_vit.py
│   ├── demo_swin.py
│   └── demo_custom_model.py
├── cam_runner.py          # 统一入口：python cam_runner.py --config ...
├── requirements.txt
└── README.md
```

## 设计取舍

> 你最初的想法是"一个 py + yaml 统一所有算法"。我评估后采用 **双轨方案**：
>
> 1. **统一入口** `cam_runner.py + configs/*.yaml`：生产、批量、对比实验高效。
> 2. **每算法独立 demo** `examples/demo_*.py`：聚焦展示该算法的差异点（是否需要梯度 / 类别 / batch / reshape），不依赖 `core/`，方便复制走人。
>
> 两条路线共享同一份"模型加载 / 目标层 / 可视化"的概念模型。

## 快速开始

环境：`D:\miniconda\envs\py310\python.exe`（`grad-cam`、`torch`、`torchvision`、`pyyaml`、`opencv-python`、`pillow` 已就绪）。

### 方式 A：统一入口（推荐）

```powershell
D:\miniconda\envs\py310\python.exe cam_runner.py --config configs/default.yaml
# 切换算法（覆盖 yaml）：
D:\miniconda\envs\py310\python.exe cam_runner.py --config configs/default.yaml --method gradcam++
```

输出落到 `outputs/`，三联图：[原图 | 热力图 | 叠加图]。

### 方式 B：单算法教学脚本

```powershell
D:\miniconda\envs\py310\python.exe examples/demo_gradcam.py
```

## 迁移到自定义模型

YAML：

```yaml
model:
  source: custom
  entrypoint: my_pkg.models:build_resnet
  entrypoint_kwargs: { depth: 50, num_classes: 7 }
  weights_path: D:/path/to/your.pth
  weights_strict: false
  target_layer: "layer4.-1"   # 字符串路径；多层用列表
```

要点：
- **`target_layer` 是 CAM 算法的核心锚点**，通常选"最后一个卷积块的输出"。
- ViT/Swin 等 token 序列模型需 `reshape_transform`，参考 `configs/README.md`。
- AblationCAM/ScoreCAM 较慢；`extra: { batch_size: 16 }` 可节省显存。

## 算法速查

| 方法 | 需要梯度 | 需要类别 | 速度 | 一句话 |
|---|---|---|---|---|
| GradCAM | √ | √ | 快 | 经典基线 |
| GradCAM++ | √ | √ | 快 | 多目标更稳 |
| XGradCAM | √ | √ | 快 | 类别一致性更好 |
| HiResCAM | √ | √ | 快 | 元素级，定位更精细 |
| GradCAMElementWise | √ | √ | 快 | 仅保留正贡献 |
| LayerCAM | √ | √ | 快 | 浅层也可用 |
| EigenCAM | × | × | 快 | 类别无关，找显著区域 |
| EigenGradCAM | √ | √ | 中 | EigenCAM 的类别敏感版 |
| AblationCAM | × | √ | 慢 | 通道置零，前向重计算 |
| ScoreCAM | × | √ | 慢 | 激活当 mask，重新前向 |
| FullGrad | √ | √ | 中 | 自动聚合所有偏置层 |

## 路线图（多轮交付）

- [x] **Round 1**：核心框架 + 统一入口 + GradCAM demo + 文档
- [x] **Round 2**：补齐其余 10 个算法独立 demo（GradCAM++/XGradCAM/HiResCAM/GradCAMElementWise/LayerCAM/EigenCAM/EigenGradCAM/AblationCAM/ScoreCAM/FullGrad）+ `examples/README.md` 决策速查
- [x] **Round 3**：ViT/Swin `reshape_transform` 完整 demo (`examples/demo_vit.py`, `examples/demo_swin.py`) + 自写 `nn.Module` 接入示例 (`custom_models/my_simple_cnn.py` + `examples/demo_custom_model.py`) + 对应 `configs/{vit,swin,custom_model}.yaml`
- [ ] **Round 4**：批量评估脚本（CAM 指标：Drop / Increase / road score）

## 参考

- [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- 论文集合：参见上述仓库 README 的 "References"。
