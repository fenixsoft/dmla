## 数据目录结构

用户宿主机目录（默认 `~/dmla-data`，可通过 `dmla data mount` 自定义）：

```
~/dmla-data/
├── datasets/                          # 数据集目录
│   ├── tiny-imagenet-200/             # Tiny ImageNet
│   │   ├── train/                     # 训练集 (200类 × 500张)
│   │   ├── val/                       # 验证集
│   │   ├── test/                      # 测试集
│   │   ├── wnids.txt                  # 类别 ID 列表
│   │   └── words.txt                  # 类别名称映射
│   ├── cifar-10/                      # CIFAR-10
│   ├── cifar-100/                     # CIFAR-100
│   ├── mnist/                         # MNIST
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
    ├── downloads/                     # 下载临时文件
    ├── preprocessing/                 # 预处理缓存
    └── torch_hub/                     # torch hub 缓存
```

## CLI `dmla data` 命令设计

### TUI 菜单结构

```
DMLA 数据管理
================

当前挂载路径: ~/dmla-data
数据集: 2 个已下载
模型: 3 个已保存

------------------------------------
  1. 挂载路径设置        [当前: ~/dmla-data]
  2. 下载数据集
  3. 查看数据集列表
  4. 清空数据内容
  5. 显示实际路径
  6. 删除数据卷
  7. 退出
------------------------------------

选择: [1-7]
```

### 下载数据集子菜单

```
可用数据集:

  [x] Tiny ImageNet 200  (已下载, 250MB)
  [ ] CIFAR-10           (170MB)
  [ ] CIFAR-100          (170MB)
  [ ] MNIST              (11MB)

[空格] 选择  [Enter] 开始下载  [q] 返回
```

### 下载进度显示

使用 curl/wget 原始进度输出，直接显示在 TUI 界面：

```
下载 Tiny ImageNet 200...
URL: https://cs231n.stanford.edu/tiny-imagenet-200.zip
目标: ~/dmla-data/cache/downloads/tiny-imagenet-200.zip

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  45  250M   45  112M    0     0   5.2M      0  0:00:48  0:00:21  0:00:27  5.1M

下载完成，正在解压...
解压完成，已移动到 ~/dmla-data/datasets/tiny-imagenet-200/
```

## 数据集下载地址

| 数据集 | 官方地址 | 大小 | 文件格式 |
|:-------|:---------|:-----|:---------|
| Tiny ImageNet | `https://cs231n.stanford.edu/tiny-imagenet-200.zip` | 250MB | ZIP |
| CIFAR-10 | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | 170MB | TAR.GZ |
| CIFAR-100 | `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` | 170MB | TAR.GZ |
| MNIST | torchvision 内置 | 11MB | 自动 |

## 进度反馈机制

### 架构设计

```
[容器内 Python] ──write──> [/workspace/progress.json] ──mount──> [宿主机文件]
                                                              │
                                                              │ poll (每 2s)
                                                              ▼
                                                         [sandbox.js]
                                                              │
                                                              │ SSE push
                                                              ▼
                                                        [前端进度条]
```

### Python API

```python
from dmla_progress import ProgressReporter

progress = ProgressReporter(
    total_steps=100,           # 总 epoch 数
    description="训练 AlexNet"
)

for epoch in range(100):
    train_one_epoch()
    val_acc = validate()
    progress.update(epoch + 1, message=f"Epoch {epoch+1}: Acc={val_acc:.2%}")

progress.complete(message="训练完成")
```

### progress.json 格式

```json
{
  "description": "训练 AlexNet",
  "total_steps": 100,
  "current_step": 45,
  "percent": 45.0,
  "message": "Epoch 45: Acc=72.35%",
  "status": "running",
  "start_time": "2026-05-02T10:00:00Z",
  "elapsed_seconds": 1800
}
```

## 超时参数扩展

### Markdown 语法

```markdown
```python runnable gpu timeout=600
# 10 分钟超时
```

```python runnable gpu timeout=unlimited
# 无超时限制，长时间训练任务
```
```

### 前端处理逻辑

- `timeout <= 60`: 使用现有逻辑，不显示进度条
- `timeout > 60 或 unlimited`: 启用进度轮询，显示进度条组件

## Dockerfile 修改

```dockerfile
# 新增内容

# 创建数据目录结构
RUN mkdir -p /data/datasets \
             /data/models \
             /data/outputs \
             /data/cache/downloads \
             /data/cache/preprocessing \
             /data/cache/torch_hub

# 复制进度报告模块
COPY local-server/src/dmla_progress.py /workspace/dmla_progress.py

# 设置 TORCH_HOME 环境变量
ENV TORCH_HOME=/data/cache/torch_hub
```

## sandbox.js 修改

```javascript
// 新增 data volume 挂载
function getDataVolumePath() {
  // 从 ~/.dmla/config.json 读取用户配置的路径
  // 默认 ~/dmla-data
}

// 容器创建时添加绑定
const dataPath = getDataVolumePath()
if (dataPath && fs.existsSync(dataPath)) {
  containerConfig.HostConfig.Binds.push(`${dataPath}:/data`)
}

// 进度轮询（仅对长任务启用）
function startProgressPolling(containerId, timeout) {
  if (timeout <= 60) return

  const interval = setInterval(() => {
    // 读取 progress.json
    const progress = readProgressFile(containerId)
    if (progress) {
      // 通过 SSE 推送到前端
      sendProgressUpdate(progress)
    }
  }, 2000)
}
```

## AlexNet 实验文档结构

```markdown
# AlexNet PyTorch 训练实验

## 实验准备
- 挂载数据目录
- 下载 Tiny ImageNet 数据集

## 第一阶段：数据准备
[runnable gpu] - 自动下载/验证数据集

## 第二阶段：数据预处理
[runnable gpu timeout=120] - DataLoader 创建

## 第三阶段：模型定义
[runnable gpu extract-class="AlexNet"] - 网络结构定义

## 第四阶段：模型训练
[runnable gpu timeout=unlimited] - 使用 ProgressReporter

## 第五阶段：模型推理
[runnable gpu] - 加载模型进行预测
```