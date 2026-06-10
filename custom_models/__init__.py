"""自定义模型示例包。

在 yaml 里这样引用：

    model:
      source: custom
      entrypoint: custom_models.my_simple_cnn:build_net
      entrypoint_kwargs: { num_classes: 10, width: 32 }
      target_layer: "features.6"
"""
