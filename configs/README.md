# YAML 配置字段说明

## 顶层
| 字段     | 含义                              |
|----------|-----------------------------------|
| `device` | `cuda` 或 `cpu`                   |
| `model`  | 模型与目标层配置                  |
| `input`  | 输入图像与预处理                  |
| `cam`    | 算法选择及其参数                  |
| `output` | 输出路径                          |

## `model`
| 字段                 | 含义                                                                  |
|----------------------|-----------------------------------------------------------------------|
| `source`             | `torchvision` / `timm` / `custom`                                     |
| `name`               | torchvision/timm 模型名，例如 `resnet34`、`vit_base_patch16_224`       |
| `num_classes`        | 覆盖最后一层输出维度（None 表示沿用默认）                              |
| `pretrained`         | 是否使用框架自带预训练（torchvision 已废弃；推荐通过 `weights_path` 显式加载） |
| `weights_path`       | 本地权重文件                                                          |
| `weights_strict`     | `load_state_dict` 的 `strict`                                          |
| `entrypoint`         | 仅 `custom` 时使用，形如 `my_pkg.models:build_net`                     |
| `entrypoint_kwargs`  | 传给 `entrypoint` 的关键字参数                                         |
| `target_layer`       | 目标层路径，字符串或字符串列表。可写负索引：`layer4.-1` / `blocks.11.norm1` |
| `reshape_transform`  | 仅 ViT/Swin 必填，例如 `{kind: vit, height: 14, width: 14, has_cls_token: true}` |

### `target_layer` 常用对照
| 架构                         | 推荐 target_layer                  |
|------------------------------|------------------------------------|
| ResNet/ResNeXt/WideResNet    | `layer4.-1`                        |
| VGG / MobileNetV2 / EfficientNet (torchvision) | `features.-1`        |
| DenseNet                     | `features.norm5`                   |
| ConvNeXt                     | `features.-1`                      |
| ViT-B/16 (timm)              | `blocks.-1.norm1`                  |
| Swin-T (timm)                | `layers.-1.blocks.-1.norm1`        |

## `input`
| 字段         | 含义                                  |
|--------------|---------------------------------------|
| `image`      | 输入图像路径                          |
| `image_size` | int 或 `[H, W]`                       |
| `mean`,`std` | 归一化均值方差；自训练模型按需替换    |

## `cam`
| 字段             | 含义                                                                  |
|------------------|-----------------------------------------------------------------------|
| `method`         | 见 `core/cam_factory.py::_REGISTRY`                                   |
| `target_class`   | 要解释的类别索引；`null` 表示预测最大类                                |
| `aug_smooth`     | TTA 平滑（更稳但更慢）                                                |
| `eigen_smooth`   | SVD 主成分平滑                                                        |
| `extra`          | 算法特定关键字。`AblationCAM`: `{batch_size: 32, ratio_channels_to_ablate: 1.0}` |
