# `examples/` — 各 CAM 算法独立教学脚本

每个文件都是 **自包含可运行** 的最小 demo：从 0 加载 ResNet34 + 一张图，跑出三联图风格的叠加可视化保存到 `outputs/`。所有文件的"骨架"完全一致（模型加载 + 图像预处理 + 调用 CAM），方便你 **横向对比每个算法的差异点**——差异都集中在每个文件顶部 docstring 与构造 CAM 那 1-2 行。

## 使用

```powershell
D:\miniconda\envs\py310\python.exe examples/demo_gradcam.py
D:\miniconda\envs\py310\python.exe examples/demo_gradcam_plusplus.py
# ...其余同理
```

如需换模型 / 图：编辑文件顶部 `WEIGHTS` / `IMAGE` / `build_model` / `target_layers` 即可。

## 索引（按推荐学习顺序）

| 顺序 | 脚本 | 算法 | 核心差异速记 |
|---|---|---|---|
| 1 | `demo_gradcam.py` | GradCAM | 基线：梯度全局平均当通道权重 |
| 2 | `demo_gradcam_plusplus.py` | GradCAM++ | 多目标 / 小目标更稳（梯度高阶项） |
| 3 | `demo_xgradcam.py` | XGradCAM | 用激活归一化，类别一致性更好 |
| 4 | `demo_hirescam.py` | HiResCAM | 不做空间平均，元素级乘积 → 定位更精细 |
| 5 | `demo_gradcam_elementwise.py` | GradCAMElementWise | 仅保留正贡献位置，热力图更"干净" |
| 6 | `demo_layercam.py` | LayerCAM | 浅层也能可视化；本 demo 展示多层融合 |
| 7 | `demo_eigencam.py` | EigenCAM | **无梯度 / 无类别**；SVD 找主显著模式 |
| 8 | `demo_eigengradcam.py` | EigenGradCAM | EigenCAM 的类别敏感版（A·grad 上做 SVD） |
| 9 | `demo_ablationcam.py` | AblationCAM | **慢**；通道置零看得分下降，无梯度 |
| 10 | `demo_scorecam.py` | ScoreCAM | **慢**；激活当 mask 重新前向打分 |
| 11 | `demo_fullgrad.py` | FullGrad | 聚合全网 bias 项的归因，覆盖更广 |
| 12 | `demo_vit.py` | GradCAM @ ViT-B/16 | **Round 3**：token 序列 → 空间图，`reshape_transform` 必看 |
| 13 | `demo_swin.py` | GradCAM @ Swin-T | **Round 3**：NHWC 张量如何接入；两种 hook 路径对比 |
| 14 | `demo_custom_model.py` | GradCAM @ 自写 CNN | **Round 3**：迁移到 `custom_models/my_simple_cnn.py` 的完整示例 |

## 迁移到自定义模型时只需改 3 处

1. **`build_model()`** —— 换成你自己的 `nn.Module` 并 `load_state_dict`。
2. **`target_layers`** —— 通常选"最后一个卷积块的输出"：
   - ResNet/ResNeXt → `[model.layer4[-1]]`
   - VGG/MobileNet/EfficientNet → `[model.features[-1]]`
   - DenseNet → `[model.features.norm5]`
   - ConvNeXt → `[model.features[-1]]`
   - ViT (timm) → `[model.blocks[-1].norm1]` + `reshape_transform`（见 `configs/README.md`）
   - Swin (timm) → `[model.layers[-1].blocks[-1].norm1]` + `reshape_transform`
3. **`load_image()`** 的 `mean/std`、`size` —— 与训练时一致。

> ViT / Swin 等 token 序列 / NHWC 模型需要 `reshape_transform`：见 `demo_vit.py` 与
> `demo_swin.py`，里面用文字 + 维度变化逐步讲明白这一步。

## 何时该用哪个？（决策速查）

- **不知道选啥 / 起步** → `demo_gradcam.py`
- **图中有多个目标 / 小目标** → `gradcam_plusplus` 或 `hirescam`
- **想要更细的定位边界** → `hirescam` / `layercam`（浅层）
- **类别相似容易混淆** → `xgradcam`
- **模型不可微 / 量化推理** → `eigencam` 或 `scorecam`
- **不想/不能指定类别，只想看显著区域** → `eigencam`
- **需要论文级"干净"可视化，不在乎慢** → `scorecam`
- **需要强解释力（"这个通道置零真的让得分掉了"）** → `ablationcam`
- **想要全网归因（含 bias 贡献）** → `fullgrad`

## 批量对比 / 生产用法

教学脚本 ≠ 生产工具。对单图跑全部算法做横向对比，建议用统一入口 + yaml：

```powershell
D:\miniconda\envs\py310\python.exe cam_runner.py --config configs/default.yaml --method gradcam++
```

参见仓库根 `README.md`。
