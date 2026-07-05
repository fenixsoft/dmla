# 数据目录结构

用户宿主机数据目录（默认 `~/dmla-data`，可通过 `dmla data mount` 自定义）：

```
~/dmla-data/
├── datasets/                          # 数据集目录
│   ├── tiny-imagenet-200/             # Tiny ImageNet (200 类)
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
│   ├── cartoon-face/                  # Cartoon Face (卡通人脸)
│   └── custom/                        # 用户自定义数据集
│
├── models/                            # 模型目录
│   ├── alexnet/                       # AlexNet 相关模型
│   │   ├── checkpoints/               # 训练中间 checkpoint
│   │   └── final/                     # 最终模型
│   ├── vgg/                           # VGG 系列模型
│   ├── resnet/                        # ResNet 系列模型
│   ├── gan/                           # GAN 模型 (预留)
│   ├── llm/                           # 大语言模型 (预留)
│   └── pretrained/                    # 预训练模型下载
│
├── outputs/                           # 输出目录
│   ├── training_logs/                 # 训练日志
│   ├── visualizations/                # 可视化结果
│   └── exports/                       # 导出文件 (ONNX等)
│
└── cache/                             # 缓存目录
    ├── downloads/                     # 数据集下载临时文件
    ├── preprocessing/                 # 预处理缓存
    └── torch_hub/                     # torch hub 缓存
```

**使用方式**：
- 容器内访问：`/data/` 目录映射到宿主机数据目录
- 代码中保存模型：`torch.save(model, '/data/models/alexnet/final/model.pth')`
- 代码中加载数据集：`torchvision.datasets.ImageFolder('/data/datasets/tiny-imagenet-200/train')`

**数据集下载**：
运行 `dmla data` 进入 TUI 菜单，选择"下载数据集"，支持：
- Tiny ImageNet 200 (250MB)
- CIFAR-10 (170MB)
- CIFAR-100 (170MB)
- MNIST (11MB，通过 torchvision 自动下载)
- Cartoon Face (288MB，卡通人脸图片)

**数据集配置维护**：

数据集的下载地址、名称等配置定义在 `packages/cli/src/commands/data.js` 的 `DATASETS` 数组中，每个条目包含 `id`、`name`、`url`、`size`、`format`、`targetDir`、`source` 字段。更新数据集地址时直接修改该数组中对应条目的 `url` 和 `source` 即可。
