"""CAM 框架核心模块。

子模块职责（保持单一职责，便于复用与测试）：
    logger        : 统一日志输出
    image_io      : 图像加载、预处理、反归一化
    model_loader  : 按配置加载 torchvision / timm / 自定义模型与权重
    target_layers : 根据架构自动/手动选择目标层；ViT/Swin 的 reshape_transform
    cam_factory   : 按算法名构建 pytorch_grad_cam 的 CAM 对象
    visualize     : 热力图叠加、网格保存
"""
