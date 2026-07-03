# DMLA 沙箱服务迁移至阿里云函数计算（FC）设计文档

日期: 2026-07-03
作者: icyfenix
状态: 待审核

## 目标

将 DMLA 文档站的 Python 代码沙箱执行服务从本地 Docker 模式迁移到阿里云函数计算（FC）Serverless 平台，实现按需调用、闲置缩容到零，降低运维成本。

## 背景

当前 DMLA 代码沙箱架构：

```
浏览器 → VuePress 文档站 → local-server (Express) → Docker 容器 → kernel_runner.py
```

其中 `local-server` 是一个 Node.js Express 服务，通过 Dockerode 管理 Docker 容器生命周期，每次代码执行请求时创建容器、运行 `kernel_runner.py`、返回结果后销毁容器。用户需要在自己机器上启动该服务才能运行文档中的代码。

目标架构：

```
浏览器 → VuePress 文档站 → FC HTTP 触发器 → fc_handler.py → kernel_runner.py
```

FC 平台替代 Express 的容器管理职责，`fc_handler.py` 是一个轻量 Python HTTP 入口，接收代码执行请求后直接调用 `kernel_runner.py`。

## 现有限制

- **仅 CPU 沙箱迁移**：GPU 镜像高达 7GB，不适合 Serverless 场景（冷启动过长、费用高）。GPU 训练实验仍需本地 Docker 模式。
- **不支持数据集**：FC 环境下无法挂载 `/data` 目录，AlexNet 等需要数据集的实验无法运行。依赖数据集读取的代码块不适用于 FC 模式。
- **单实例并发为 1**：每个 FC 实例每次处理一个请求，避免并发带来的资源竞争。

## 技术方案

### 镜像构建

#### 新增 Dockerfile

`local-server/Dockerfile.sandbox.fc` — FC 专用轻量镜像，相比 CPU 镜像做以下精简：

**保留（共用代码所需的库）：**

| 类别 | 库 | 原因 |
|------|-----|------|
| 运行时 | python:3.11-slim | 基础镜像 |
| **Jupyter 协议** | jupyter_client, ipykernel | 富输出捕获（图片、进度）的核心依赖 |
| PyTorch | torch (CPU), torchvision, torchaudio | 文章代码片段使用 |
| HuggingFace | transformers, tokenizers, datasets, accelerate, bitsandbytes | LLM 实验文档使用 |
| 科学计算 | numpy, scipy, scikit-learn | 数据处理、模型训练 |
| 可视化 | matplotlib, matplotlib_inline, pillow | 内联图片输出 |
| CV | opencv-python-headless | 图像处理代码片段 |
| 工具 | requests, ipywidgets | 通用工具 |
| 核心 | kernel_runner.py, dmla_progress.py, shared/ | 代码执行和进度报告 |
| 字体 | fonts-wqy-microhei, fonts-noto-cjk | 中文字体支持 |

**移除（FC 不需要或代码未使用）：**

| 类别 | 库 | 原因 |
|------|-----|------|
| 数据处理 | pandas | 文档中无 runnable 代码使用 |
| 数据库 | lmdb | FC 无法挂载数据目录，数据依赖实验无法运行 |
| 运行时 | Node.js, Express, Dockerode | FC 替代容器管理 |

#### Dockerfile 结构要点

- 基础镜像与 CPU 版一致，保证执行环境兼容
- CMD 设为 `python3 /workspace/fc_handler.py`
- 不需要 `EXPOSE` 端口（FC 自动处理 HTTP 端口监听，默认端口 9000）

### FC HTTP Handler

`local-server/src/fc_handler.py` 是新增的 FC 入口文件，约 100-150 行。

#### 核心职责

1. 启动 HTTP 服务（Flask 或内置 `http.server`）
2. 接收 POST 请求，获取 `{code, timeout?}`
3. 调用 `kernel_runner.py` 的 `run_code()` 函数执行代码
4. 将执行结果包装为 JSON 返回

#### API 设计

**请求（POST /）：**

```json
{
  "code": "print('Hello')",
  "timeout": 60
}
```

**响应（200 OK）：**

```json
{
  "success": true,
  "outputs": [
    {"type": "stream", "name": "stdout", "text": "Hello\n"},
    {"type": "display_data", "data": {"image/png": "base64..."}, "metadata": {}}
  ],
  "executionTime": 0.523
}
```

**错误响应：**

```json
{
  "success": false,
  "outputs": [
    {"type": "error", "ename": "SyntaxError", "evalue": "...", "traceback": [...]}
  ],
  "executionTime": 0.01
}
```

#### 超时处理

- FC 函数级别超时通过 FC 控制台/API 配置为 100 秒
- `kernel_runner.py` 的 `DEFAULT_TIMEOUT` 保持 60 秒不变
- `fc_handler.py` 可将请求中的 `timeout` 参数透传给 `run_code()`
- 冷启动时间由 FC 平台计算，不计入执行超时

#### 安全考虑

- 不使用 `subprocess` 调用外部命令，直接 import `kernel_runner.py` 的 `run_code()` 函数
- 用户代码通过 IPython kernel 在进程内执行，已具备基础隔离
- FC 实例间天然隔离，每个请求使用独立实例或复用空闲实例

### 前端改动

#### Settings.vue 增强

当前 `Settings.vue` 的"沙箱服务" Tab 已支持自定义地址输入。改动为增加服务模式选择：

**新增 UI 元素：**

```
服务模式:  ○ FC（默认）    ○ 自定义地址

  [当选择"FC（默认）"时]
  FC 沙箱地址: https://xxxxx.cn-hangzhou.fc.aliyuncs.com/...  (只读显示)

  [当选择"自定义地址"时，显示输入框]
  服务地址: [http://localhost:3001        ]
```

**配置存储逻辑：**

- `sandboxEndpoint` 字段含义不变，仍为沙箱服务的完整 URL
- 新增 `sandboxMode` 字段：`'fc'` | `'custom'`
- FC 默认地址作为常量 `FC_DEFAULT_URL` 写在 `Settings.vue` 中（非 client.js）
- 切换模式时自动更新 endpoint 值

#### sandbox-config.js 改动

`getSandboxConfig()` 逻辑不变，保持读取 `localStorage['site-config'].sandboxEndpoint`。设置页面负责保证该值正确。

#### client.js 改动

- 连接超时 `CONNECTION_TIMEOUT` 考虑 FC 冷启动，建议从 10000ms 调整为 20000ms
- 不再新增 `FC_ENDPOINT` 硬编码常量，全部通过 `getSandboxEndpoint()` 获取

### 镜像构建与自动部署

#### GitHub Actions 工作流扩展

在现有 `publish-docker.yml` 中新增 `build-fc` job（构建 FC 精简镜像）和 `update-fc` job（在 ACR 推送后更新 FC 函数）。服务名、函数名、地域固定写在 workflow 文件中，仅 AccessKey 通过 Secrets 注入：

```yaml
  build-fc:
    runs-on: ubuntu-latest
    needs: build-images
    steps:
      # ... (checkout + extract shared modules + setup Docker Buildx)
      - name: Build FC image
        uses: docker/build-push-action@v7
        with:
          context: .
          file: local-server/Dockerfile.sandbox.fc
          tags: ${{ env.IMAGE_NAME }}:fc
          outputs: type=docker,dest=/tmp/image-fc.tar

  update-fc:
    runs-on: ubuntu-latest
    needs: [push-acr, push-acr-fc]
    steps:
      - name: Setup Aliyun CLI
        run: curl -fsSL https://aliyuncli.alicdn.com/install.sh | bash
      - name: Configure Aliyun CLI
        run: |
          aliyun configure set \
            --access-key-id ${{ secrets.ALIBABA_CLOUD_ACCESS_KEY_ID }} \
            --access-key-secret ${{ secrets.ALIBABA_CLOUD_ACCESS_KEY_SECRET }} \
            --region cn-hangzhou
      - name: Update FC function
        run: |
          IMAGE_URI="${{ env.ACR_REGISTRY }}/${{ env.ACR_NAMESPACE }}/${{ env.IMAGE_NAME }}:fc"
          aliyun fc update-function \
            --service-name dmla \
            --function-name sandbox-cpu \
            --code "{\"imageUri\": \"$IMAGE_URI\"}"
```

**硬编码常量（不随环境变动）：**

| 常量 | 值 |
|------|-----|
| FC 地域 | `cn-hangzhou`（与 ACR 同地域） |
| FC 服务名 | `dmla` |
| FC 函数名 | `sandbox-cpu` |

#### ACR 镜像命名

新增 `fc` tag 用于 FC 专用镜像（区别于 `cpu` tag 的完整镜像）：

- 完整 CPU 镜像：`dmla-sandbox:cpu`（保持现有，NPM + 用户本地使用）
- FC 精简镜像：`dmla-sandbox:fc`（新增，仅 FC 使用）

#### 首次创建（本地 CLI 手动操作）

FC 服务和函数首次由开发者在本地通过 aliyun CLI 一次性完成，不写入 CI：

```bash
# 安装 aliyun CLI（如未安装）
curl -fsSL https://aliyuncli.alicdn.com/install.sh | bash

# 配置认证
aliyun configure set --access-key-id XXXX --access-key-secret XXXX --region cn-hangzhou

# 创建 FC 服务
aliyun fc create-service --service-name dmla

# 创建 FC 函数（Custom Container，首次先占位 fc 镜像）
aliyun fc create-function \
  --service-name dmla \
  --function-name sandbox-cpu \
  --runtime custom-container \
  --handler not-used \
  --cpu 1 \
  --memory-size 2048 \
  --disk-size 512 \
  --timeout 100 \
  --code "{\"imageUri\": \"crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc\"}"

# 创建 HTTP 触发器（匿名访问，获取公网 URL）
aliyun fc create-trigger \
  --service-name dmla \
  --function-name sandbox-cpu \
  --trigger-name http-trigger \
  --trigger-type http \
  --trigger-config '{"authType":"anonymous","methods":["POST","GET"]}'
```

创建完成后，将触发器的公网 URL 配置到 `Settings.vue` 的 `FC_DEFAULT_URL` 常量中。

### 费用预算

#### 定价参数

基于阿里云函数计算国内站 CU 阶梯计费：

| 阶梯 | CU 用量 | 单价 |
|------|---------|------|
| 阶梯 1 | 0 ~ 1 亿 CU | $0.000020/CU |
| 阶梯 2 | 1 亿 ~ 5 亿 CU | $0.000017/CU |
| 阶梯 3 | 5 亿以上 | $0.000014/CU |

**CU 转换系数：**

| 资源 | 系数 |
|------|------|
| 活跃 vCPU | 1 CU/vCPU·秒 |
| 闲置 vCPU | 0（免费） |
| 内存 | 0.15 CU/GB·秒 |
| 磁盘 | 0.05 CU/GB·秒（512 MB 内免费） |
| 函数调用次数 | 0.0075 CU/次（75 CU/万次） |

#### 假设参数

| 参数 | 取值 | 说明 |
|------|------|------|
| 函数规格 | 1 vCPU / 2 GB 内存 / 512 MB 磁盘 | — |
| 平均执行时间 | 3 秒 | 代码片段通常很短 |
| 日均调用量 | 1000 ~ 5000 次 | 中低流量场景 |
| 冷启动时间 | 由 FC 平台承担 | 不计费 |

#### 单次调用 CU 消耗

```
vCPU:     1.0 核 × 3 秒 × 1.0     = 3.00 CU
内存:     2.0 GB × 3 秒 × 0.15    = 0.90 CU
调用次数:  1 次 × 0.0075          = 0.008 CU
磁盘:     512 MB 以内              = 0
─────────────────────────────────────────
合计:                              ≈ 3.91 CU/次
```

#### 月费用估算

| 场景 | 月调用量 | 月 CU | 月费用（USD） | 月费用（CNY 估算） |
|------|----------|-------|-------------|-------------------|
| 保守 | 3 万次 | 12 万 | $2.40 | ≈¥17 |
| **预期** | **15 万次** | **59 万** | **$11.71** | **≈¥85** |
| 活跃 | 30 万次 | 117 万 | $23.40 | ≈¥170 |

**免费试用**：阿里云 FC 新用户可获得 3 个月每月 15 万 CU 免费额度，基本覆盖保守场景全部费用。

#### 对比

| 方案 | 月费用 | 运维 | 闲置 |
|------|--------|------|------|
| ECS 2vCPU/4GB 包月 | ~$30-40 | 需管理服务器 | 24/7 付费 |
| FC Serverless | ~$12 | 零运维 | 缩容到零，不付费 |

### 业务价值

**为什么使用 FC：**
- 文档站的代码运行是典型的"突发请求"模式，用户点击"运行"时才需要计算资源
- 大部分时间沙箱无人使用，ECS 按包月付费的资源利用率极低
- 镜像已通过 GitHub Actions 自动推送到阿里云 ACR，对接 FC 的边际成本很小

**为什么之前不用：**
- 沙箱代码的图片输出依赖 IPython kernel 的 `display_data` 消息协议
- 直接用 `subprocess` 执行 Python 只能拿到 stdout/stderr，图片丢失
- 现在明确了 Jupyter 协议必须保留，直接在 FC 容器中运行 kernel_runner.py

### 凭证清单

| 信息 | 获取方式 | 用途 | 存储位置 |
|------|---------|------|---------|
| Alibaba Cloud AccessKey ID + Secret | RAM 访问控制 → 创建子账号 AccessKey | FC API 调用（首次创建 + 后续 CI 更新函数） | GitHub Secrets |
| ACR 认证 | 已有（ACR_USERNAME, ACR_PASSWORD） | 镜像拉取 | GitHub Secrets |
| FC HTTP 触发器公网 URL | FC 控制台 → 触发器 → 公网访问地址 | 前端请求地址 | Settings.vue 常量 |

FC 地域、服务名、函数名固定硬编码在 workflow 文件中，不作为 Secrets 管理。

**RAM 权限建议：** 子账号授权 `AliyunFCFullAccess`（FC 函数管理）+ `AliyunContainerRegistryReadOnlyAccess`（ACR 镜像拉取）。

### 测试用例

#### 端到端等价性测试

测试脚本位置：`local-server/tests/fc-sandbox-e2e.test.py`

测试框架：每个测试用例同时向 FC 和 CPU 沙箱发起请求，比较输出结构的一致性。

**测试运行方式：**

```bash
# FC 地址
export FC_ENDPOINT="https://xxxxx.cn-hangzhou.fc.aliyuncs.com/..."

# CPU 沙箱地址（需要本地服务运行）
export CPU_SANDBOX_ENDPOINT="http://localhost:3001"

# 运行测试
python3 local-server/tests/fc-sandbox-e2e.test.py
```

**测试用例清单：**

| # | 测试名称 | 代码 | 比较内容 |
|---|---------|------|---------|
| 1 | 纯文本输出 | `print("Hello, DMLA!")` | `success`, stdout text 内容 |
| 2 | 多行输出 | `for i in range(3): print(i)` | stdout 中各行内容 |
| 3 | 表达式结果 | `3.14 * 2` | `execute_result` 中的计算结果 |
| 4 | 图片输出 | `plt.plot([1,2,3]); plt.show()` | `display_data` 中 image/png 存在且为有效 base64 |
| 5 | 中文图片 | `plt.title("中文"); plt.show()` | display_data 存在, stderr 无字体缺失警告 |
| 6 | 运行时异常 | `1/0` | `error.ename == "ZeroDivisionError"` |
| 7 | 语法异常 | `if True print` | error 输出存在 |
| 8 | NumPy 浮点计算 | `np.array([1,2,3]).mean()` | 数值精度一致 |
| 9 | 执行时间 | `import time; time.sleep(0.5)` | `executionTime >= 0.5` |
| 10 | 无输出执行 | `x = 1 + 1` | success=true, outputs 可能为空 |
| 11 | 大数据流输出 | `print("x"*10000)` | 大文本不截断、结构完整 |
| 12 | matplotlib 导入后图片 | `plt.imshow(np.random.rand(10,10)); plt.show()` | display_data 存在有效 PNG |

**比较策略：**

- `success` 字段：必须完全一致
- `outputs` 数组长度：必须一致
- 每个 output 的 `type`：必须一致
- 文本输出（stream/error/en/evalue）：内容一致
- 图片输出（display_data image/png）：验证是否为有效 base64 编码的 PNG（允许像素级差异，因字体渲染等可能导致细微差别）
- 执行时间：均合理范围内即可（FC 冷启动首次可能较慢）

**FC 冷启动影响：**

首次调用触发冷启动（拉取镜像 + 启动容器），后续调用复用热实例。测试时需注意：
- 第一个测试用例允许较长的响应时间（含冷启动）
- 后续用例应在正常时间内完成
- 镜像加速功能（ACR 企业版的按需加载）可显著缩短冷启动

## 待确认事项

- [ ] 是否需要自定义域名绑定 FC HTTP 触发器（而非使用 FC 默认域名）
- [ ] Settings.vue 中 FC 默认 URL 是否需要通过环境变量注入（而非写死在组件代码中）
- [ ] 是否需要购买 ACR 企业版以获得镜像加速功能（缩短冷启动，年费约 ¥600）

## 相关资源

- 阿里云 FC 自定义容器文档：https://help.aliyun.com/zh/functioncompute/fc-2-0/user-guide/overview-3
- 阿里云 FC 计费概述：https://help.aliyun.com/zh/functioncompute/fc/product-overview/billing-overview-of-fc
- Serverless CI/CD 博客：https://www.alibabacloud.com/blog/602895
- 现有 publish-docker.yml：`.github/workflows/publish-docker.yml`
- 现有 Settings.vue：`docs/.vuepress/theme/components/Settings.vue`
- kernel_runner.py：`local-server/src/kernel_runner.py`
