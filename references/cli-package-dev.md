# CLI 包开发规则

**修改源码目录而非构建产物**：

CLI 包的构建脚本（`packages/cli/scripts/build.js`）会从 `local-server/src/` 复制文件到 `packages/cli/src/server/`。因此：

- **正确做法**：修改 `local-server/src/` 目录下的源文件
- **错误做法**：直接修改 `packages/cli/src/server/` 目录下的文件（会被构建脚本覆盖）

**需要修改的正确位置**：
| 文件 | 正确位置 | 说明 |
|------|---------|------|
| `native_env_check.js` | `local-server/src/native_env_check.js` | Native 模式环境检测 |
| `native_executor.js` | `local-server/src/native_executor.js` | Native 模式代码执行器 |
| `kernel_runner.py` | `local-server/src/kernel_runner.py` | Python 代码执行内核 |
| `index.js` | `local-server/src/index.js` | 服务入口 |
| 其他 `.js/.py` 文件 | `local-server/src/` | 所有服务端代码 |

**构建流程**：
1. 修改 `local-server/src/` 下的源文件
2. 运行 `cd packages/cli && npm run build` 复制到 CLI 包
3. 提交时确保两个目录都已更新（构建脚本会自动更新 `packages/cli/src/server/`）
