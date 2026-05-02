# Dataset Auto Download

数据集自动下载能力，支持常用深度学习数据集的下载、解压和管理。

## 支持的数据集

| 名称 | URL | 大小 | 格式 | 目标目录 |
|:-----|:----|:-----|:-----|:---------|
| Tiny ImageNet 200 | `https://cs231n.stanford.edu/tiny-imagenet-200.zip` | 250MB | ZIP | `datasets/tiny-imagenet-200/` |
| CIFAR-10 | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | 170MB | TAR.GZ | `datasets/cifar-10/` |
| CIFAR-100 | `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` | 170MB | TAR.GZ | `datasets/cifar-100/` |
| MNIST | torchvision 内置 | 11MB | 自动 | `datasets/mnist/` |

## 功能需求

### 下载流程

1. 用户在 TUI 菜单选择要下载的数据集
2. 系统检查数据集是否已下载（通过目录存在性判断）
3. 显示下载信息：URL、目标路径、预计大小
4. 使用 curl（Linux/Mac）或 wget（Windows fallback）下载
5. 下载进度实时显示（使用工具原始输出）
6. 下载完成后解压到 cache/downloads/
7. 移动解压内容到目标 datasets 目录
8. 清理临时文件
9. 更新配置文件记录已下载数据集

### 进度显示

下载时直接显示 curl/wget 的进度输出：

```
下载 Tiny ImageNet 200...
URL: https://cs231n.stanford.edu/tiny-imagenet-200.zip

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  45  250M   45  112M    0     0   5.2M      0  0:00:48  0:00:21  0:00:27  5.1M
```

### 错误处理

- URL 不可达：显示错误信息，提供重试选项
- 下载中断：保留部分文件，下次继续下载（使用 `-C -` 参数）
- 解压失败：清理临时文件，显示错误信息
- 目标目录已存在：提示用户选择覆盖或跳过

### MNIST 特殊处理

MNIST 数据集通过 torchvision 自动下载，不使用外部工具：

```python
from torchvision.datasets import MNIST
MNIST(root='/data/datasets', download=True)
```

## 技术约束

- 下载工具优先使用 curl（更好的进度显示）
- Windows 环境使用 Invoke-WebRequest 或 wget
- 解压使用系统原生工具（unzip/tar）或 Node.js 库
- 下载超时设置为 30 分钟
- 支持断点续传

## 文件存储位置

- 下载临时文件：`cache/downloads/<dataset-name>.zip`
- 解压临时目录：`cache/downloads/<dataset-name>/`
- 最终存储目录：`datasets/<dataset-name>/`