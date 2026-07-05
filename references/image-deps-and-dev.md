# 镜像依赖与开发

## 依赖管理

**每次修改 Python 依赖关系后，必须通过 Native 模式启动验证依赖检测和自动安装是否正常工作。**

**依赖定义位置**：

| 文件 | 修改位置 | 说明 |
|------|---------|------|
| `local-server/src/native_env_check.js` | `SOFT_DEPS` 数组 | Native 模式软依赖列表（缺失时自动安装） |
| `local-server/Dockerfile.sandbox` | `RUN pip install` 段落 | Docker GPU 镜像依赖 |
| `local-server/Dockerfile.sandbox.cpu` | `RUN pip install` 段落 | Docker CPU 镜像依赖 |

三处依赖定义需保持一致。新增 Python 包时，三个文件都要同步更新。

**依赖变更验证流程**：

1. 修改 `SOFT_DEPS` 数组，新增或移除包名
2. 运行 `cd packages/cli && npm run build` 将改动同步到 CLI 包
3. 以 Native 模式启动服务，观察输出：
   - 已安装的包：不输出（静默跳过）
   - 新增缺失的包：显示 `安装 xxx...` 并自动 pip install
   - 安装成功：显示 `已安装: xxx, yyy`
   - 安装失败：显示 `安装失败: xxx` 和手动安装命令
4. 确认服务正常启动后，用 Python 验证包可导入：
   ```bash
   python3 -c "import transformers; print(transformers.__version__)"
   ```
5. 清理端口：`lsof -ti:3001 -sTCP:LISTEN | xargs kill 2>/dev/null`

**当前 SOFT_DEPS 完整列表**：

```
numpy, pandas, matplotlib, scipy, scikit-learn, pillow,
opencv-python-headless, jupyter_client, ipykernel, lmdb, requests,
transformers, tokenizers, datasets, ipywidgets, accelerate, bitsandbytes
```

## 镜像开发约束（Volume Mount 机制）

Docker 沙箱镜像高达 10GB，编译耗时约 15-30 分钟。为避免频繁重编译，**所有涉及镜像内容的改动必须先用 Volume Mount 机制验证通过后才能触发编译**。

**Volume Mount 机制说明**：

开发模式下，以下文件通过 Volume Mount 自动挂载到容器，修改无需重建镜像：
- `kernel_runner.py`: 挂载到 `/workspace/kernel_runner.py`
- `dmla_progress.py`: 挂载到 `/workspace/dmla_progress.py`
- `shared_modules/`: 挂载到 `/usr/local/lib/python3.11/site-packages/shared`

**两种开发模式启动方式**：

| 启动方式 | 命令 | 适用场景 | Volume Mount |
|---------|------|---------|--------------|
| 源码目录启动 | `npm run server` 或 `cd local-server && npm start` | 项目源码开发 | ✅ 自动启用 |
| CLI 开发模式 | `dmla start --dev` | npm 包安装后开发 | ✅ 需指定 --dev |

**验证流程**：

1. **修改 Python 文件**：编辑 `local-server/src/kernel_runner.py`、`dmla_progress.py` 或 `shared_modules/` 下的文件
2. **启动服务测试**：
   - 源码目录：`npm run server`
   - CLI 安装后：`dmla start --dev`
3. **验证功能正确**：通过 API 或前端执行代码块，确认新功能正常工作
4. **更新 tasks.md**：标记验证通过的子任务
5. **触发镜像编译**：**仅在所有验证通过后**执行 `npm run build:sandbox:gpu`

**环境变量控制**：

```bash
# 默认启用 Volume Mount（开发模式）
MOUNT_KERNEL_RUNNER=true    # 挂载 kernel_runner.py
MOUNT_SHARED_MODULES=true   # 挂载共享模块

# 禁用挂载（生产模式，使用镜像内置文件）
MOUNT_KERNEL_RUNNER=false
MOUNT_SHARED_MODULES=false
```

**适用范围**：

必须先验证后编译的改动类型：
- 新增 Python 模块（如 `dmla_progress.py`）
- 修改 `kernel_runner.py` 执行逻辑
- 修改 `shared_modules/` 下的共享类
- 修改 Dockerfile 中的 Python 脚本引用
- 新增数据目录挂载点

无需验证可直接编译的改动：
- 修改系统依赖（apt 包）
- 修改 Python 版本
- 修改 PyTorch/CUDA 版本
- 修改基础镜像（FROM 指令）

**验证检查清单**：

```bash
# 1. 确认服务启动时 Volume Mount 生效
# 日志应显示：
# [Sandbox] kernel_runner.py Volume Mount: /path/to/local-server/src/kernel_runner.py
# [Sandbox] dmla_progress.py Volume Mount: /path/to/local-server/src/dmla_progress.py

# 2. 测试新模块导入
curl -s -X POST http://localhost:3001/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "from dmla_progress import ProgressReporter\nprint(\"OK\")"}'

# 3. 测试共享模块导入
curl -s -X POST http://localhost:3001/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "from shared.linear.logistic_regression import LogisticRegression\nprint(\"OK\")"}'
```
