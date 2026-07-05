# 训练实验支持

## 数据路径自动注入（DATA_DIR）

kernel_runner.py 在执行代码前自动注入 `DATA_DIR` 全局变量，兼容 Docker 和 Native 模式。

**注入机制**：
```python
import os
DATA_DIR = os.environ.get('DMLA_DATA_PATH', '/data')
```

**模式差异**：
- Docker 模式：`DMLA_DATA_PATH` 未设置 → `DATA_DIR = '/data'`（容器挂载路径）
- Native 模式：`DMLA_DATA_PATH` 已设置 → `DATA_DIR = '~/dmla-data'`（宿主机数据目录）

**使用方式**：
```python
# 在代码中使用 DATA_DIR（无需导入，自动注入）
dataset_path = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200')
model_path = os.path.join(DATA_DIR, 'models', 'alexnet', 'final', 'model.pth')
cache_path = os.path.join(DATA_DIR, 'cache', 'preprocessing', 'lmdb')
```

**自定义数据路径**（Native 模式）：
```bash
# 通过环境变量自定义数据目录
export DMLA_DATA_PATH=/custom/data/path
dmla start --dev

# 或在 CLI 启动时指定
dmla data mount /custom/data/path
```

**文档代码块适配**：
文档中的代码块应使用 `DATA_DIR` 变量而非硬编码 `/data` 路径：
```python
# 错误（仅 Docker 模式可用）
data_dir = '/data/datasets/tiny-imagenet-200'

# 正确（Docker 和 Native 模式均可用）
data_dir = os.path.join(DATA_DIR, 'datasets', 'tiny-imagenet-200')
```

## 共享模块（可复用 Python 类）

文档中多个 runnable code 块可通过共享模块复用类定义：

```bash
# 从文档提取标记的类定义到共享模块
npm run extract:shared

# 构建镜像时自动执行提取
npm run build:sandbox:gpu
```

**使用方式**：

1. 在文档中标记需要提取的类：
   ```markdown
   ```python runnable extract-class="LogisticRegression"
   class LogisticRegression:
       ...
   ```
   ```

2. 在其他代码块中导入使用：
   ```python
   from shared.linear.logistic_regression import LogisticRegression
   model = LogisticRegression()
   ```

**目录结构**：
- 文档：`docs/statistical-learning/linear-models/*.md`
- 共享模块：`local-server/shared/linear/*.py`

**重要规则**：
- **禁止手动创建共享模块文件**：共享模块必须通过 `node scripts/extract-shared-modules.js` 脚本从文档中自动提取生成
- **模块路径自动推断**：脚本会根据文档路径自动推断模块名（如 `deep-learning/sequence-models/` → `shared/sequence_models/`），也可在脚本的 `CHAPTER_MAPPING` 中配置显式映射
- **开发流程**：先在文档中编写带 `extract-class` 标记的代码块 → 运行提取脚本 → 在其他代码块中导入使用

**开发模式**：Volume Mount 自动启用，修改代码无需重建镜像
**生产模式**：设置 `MOUNT_SHARED_MODULES=false` 禁用挂载
