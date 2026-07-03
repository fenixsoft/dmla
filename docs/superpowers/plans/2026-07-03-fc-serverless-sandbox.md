# DMLA 沙箱服务迁移至阿里云 FC 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 FC 专用精简镜像和 HTTP 入口，在阿里云函数计算上创建沙箱执行服务，前端支持 FC/自定义双模式切换，并通过端到端等价性测试

**Architecture:** 新增 `fc_handler.py` 作为 FC HTTP 入口，直接调用 `kernel_runner.py` 的 `run_code()` 执行代码；新增 `Dockerfile.sandbox.fc` 构建精简镜像（移除 pandas、lmdb）；CI 扩展 `build-fc` 和 `update-fc` job 自动构建及部署；前端 Settings 增加 FC/自定义双模式选择

**Tech Stack:** Python 3.11, http.server (内置), IPython kernel, PyTorch CPU, Alibaba Cloud FC CLI

## Global Constraints

- `kernel_runner.py` 的 `DEFAULT_TIMEOUT` 保持 60 秒不变
- FC 函数级别超时 100 秒
- FC 地域 `cn-hangzhou`，服务名 `dmla`，函数名 `sandbox-cpu`
- 仅 CPU 模式，不支持 GPU 和数据集读取
- 镜像 tag：`dmla-sandbox:fc`，推送到已有 ACR 仓库
- 前端沙箱模式默认 FC，用户可通过设置页切换为自定义地址
- FC handler 端口：读取 `FC_SERVER_PORT` 环境变量，默认 9000

## File Map

| 文件 | 操作 | 职责 |
|------|------|------|
| `local-server/src/fc_handler.py` | 新建 | FC HTTP 入口，接收请求并调用 kernel_runner |
| `local-server/Dockerfile.sandbox.fc` | 新建 | FC 精简镜像定义 |
| `.github/workflows/publish-docker.yml` | 修改 | 新增 build-fc + push-acr-fc + update-fc job |
| `docs/.vuepress/theme/components/Settings.vue` | 修改 | 新增 FC/自定义模式选择 |
| `docs/.vuepress/plugins/runnable-code/sandbox-config.js` | 修改 | 新增 sandboxMode 字段 |
| `docs/.vuepress/plugins/runnable-code/client.js` | 修改 | 调整连接超时 |
| `local-server/tests/fc-sandbox-e2e.test.py` | 新建 | FC vs CPU 沙箱等价性测试 |

---

### Task 1: 创建 FC HTTP Handler

**Files:**
- Create: `local-server/src/fc_handler.py`

**Interfaces:**
- Consumes: `kernel_runner.run_code(code: str, timeout: int, stream: bool) -> dict`
- Produces: HTTP `GET /api/sandbox/health`, `POST /api/sandbox/run`, `POST /api/sandbox/stream`

- [ ] **Step 1: 编写 fc_handler.py**

```python
#!/usr/bin/env python3
"""
FC (Function Compute) HTTP Handler
轻量 HTTP 入口，接收代码执行请求并调用 kernel_runner.py
"""

import json
import os
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# 将 /workspace 加入路径，确保可以 import kernel_runner
sys.path.insert(0, '/workspace')

# 延迟导入 kernel_runner（内部 import matplotlib，首次有开销）
from kernel_runner import run_code

FC_SERVER_PORT = int(os.environ.get('FC_SERVER_PORT', 9000))
DEFAULT_TIMEOUT = 60


class SandboxHandler(BaseHTTPRequestHandler):
    """FC 沙箱 HTTP 请求处理器"""

    def log_message(self, format, *args):
        """重写日志方法，输出到 stderr（FC 日志采集）"""
        print(f"[fc_handler] {format % args}", file=sys.stderr)

    def _send_json(self, status_code, data):
        """发送 JSON 响应"""
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """健康检查（Settings 测试连接使用）"""
        self._send_json(200, {
            'status': 'ok',
            'mode': 'fc',
            'timestamp': __import__('time').time()
        })

    def do_POST(self):
        """代码执行"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {
                    'success': False,
                    'outputs': [{
                        'type': 'error',
                        'ename': 'EmptyRequest',
                        'evalue': 'Request body is empty',
                        'traceback': []
                    }],
                    'executionTime': 0
                })
                return

            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body)

            code = data.get('code', '')
            timeout = data.get('timeout', DEFAULT_TIMEOUT)
            stream = data.get('stream', False)

            if not code:
                self._send_json(400, {
                    'success': False,
                    'outputs': [{
                        'type': 'error',
                        'ename': 'MissingCode',
                        'evalue': 'No code provided in request',
                        'traceback': []
                    }],
                    'executionTime': 0
                })
                return

            result = run_code(code, timeout=timeout, stream=stream)

            if stream:
                self.send_response(200)
                self.end_headers()
            else:
                self._send_json(200, result)

        except json.JSONDecodeError:
            self._send_json(400, {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': 'InvalidJSON',
                    'evalue': 'Request body is not valid JSON',
                    'traceback': []
                }],
                'executionTime': 0
            })
        except Exception as e:
            print(f"[fc_handler] Unexpected error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._send_json(500, {
                'success': False,
                'outputs': [{
                    'type': 'error',
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': traceback.format_exc().split('\n')
                }],
                'executionTime': 0
            })


def main():
    server = HTTPServer(('0.0.0.0', FC_SERVER_PORT), SandboxHandler)
    print(f"[fc_handler] Listening on port {FC_SERVER_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 验证语法**

```bash
python3 -c "import py_compile; py_compile.compile('local-server/src/fc_handler.py', doraise=True)"
```

- [ ] **Step 3: 验证 kernel_runner 可 import（需要 ipykernel 已安装）**

```bash
python3 -c "import sys; sys.path.insert(0, 'local-server/src'); from kernel_runner import run_code; print('run_code imported OK')"
```

Expected: `run_code imported OK`

- [ ] **Step 4: Commit**

```bash
git add local-server/src/fc_handler.py
git commit -m "feat: 新增 FC HTTP handler，接收代码执行请求并调用 kernel_runner

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 创建 FC 专用 Dockerfile

**Files:**
- Create: `local-server/Dockerfile.sandbox.fc`

**Interfaces:**
- Consumes: `local-server/src/fc_handler.py`, `local-server/src/kernel_runner.py`, `local-server/src/dmla_progress.py`, `local-server/shared/`
- Produces: `dmla-sandbox:fc` Docker 镜像

- [ ] **Step 1: 对比 CPU Dockerfile 确认保留和移除的库**

当前 CPU Dockerfile (`local-server/Dockerfile.sandbox.cpu`) 中的 pip 包清单：

```
# 保留的包（共 18 个）：
numpy, matplotlib, scipy, scikit-learn, requests, pillow,
opencv-python-headless, ipykernel, jupyter_client,
torch torchvision torchaudio (CPU),
transformers, tokenizers, datasets, ipywidgets, accelerate, bitsandbytes

# 移除的包：
pandas, lmdb
```

- [ ] **Step 2: 编写 Dockerfile.sandbox.fc**

```dockerfile
# ============================================
# DMLA FC Sandbox Image (Serverless CPU)
# 适用于阿里云函数计算 Custom Container 运行时
# 镜像名称: dmla-sandbox:fc
# ============================================

FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace
ENV FC_SERVER_PORT=9000

# 替换为阿里云镜像源
RUN sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    fonts-wqy-microhei \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip - 使用清华 PyPI 镜像
RUN pip install --no-cache-dir --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装科学计算库 - 使用清华镜像（移除 pandas）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    requests \
    pillow \
    opencv-python-headless \
    ipykernel \
    jupyter_client

# 安装 PyTorch (CPU 版本) - 使用 PyTorch 官方镜像
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 HuggingFace Transformers 生态（LLM 预训练实验所需）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers \
    tokenizers \
    datasets \
    ipywidgets \
    accelerate \
    bitsandbytes

# 复制 FC handler
COPY local-server/src/fc_handler.py /workspace/fc_handler.py

# 复制执行器和进度报告模块
COPY local-server/src/kernel_runner.py /workspace/kernel_runner.py
COPY local-server/src/dmla_progress.py /workspace/dmla_progress.py

# 复制共享模块
COPY local-server/shared /workspace/shared

# 配置 matplotlib 中文字体支持
RUN mkdir -p /root/.config/matplotlib && \
    printf "font.family: sans-serif\nfont.sans-serif: WenQuanYi Micro Hei, WenQuanYi Zen Hei, DejaVu Sans\nfont.monospace: WenQuanYi Micro Hei Mono, WenQuanYi Zen Hei Mono, DejaVu Sans Mono\naxes.unicode_minus: False\n" > /root/.config/matplotlib/matplotlibrc && \
    rm -rf /root/.cache/matplotlib

# FC 需要的健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/api/sandbox/health || exit 1

# 启动 FC handler
CMD ["python3", "/workspace/fc_handler.py"]
```

- [ ] **Step 3: Commit**

```bash
git add local-server/Dockerfile.sandbox.fc
git commit -m "feat: 新增 FC 专用 Dockerfile，移除 pandas 和 lmdb 以精简镜像

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 修改 CI 工作流（publish-docker.yml）

**Files:**
- Modify: `.github/workflows/publish-docker.yml`

**Interfaces:**
- Consumes: `Dockerfile.sandbox.fc`, ACR secrets
- Produces: `build-fc` job 构建并导出镜像，`push-acr-fc` job 推送 fc tag，`update-fc` job 更新 FC 函数

- [ ] **Step 1: 在 build-images job 后新增 build-fc job**

在 `build-images` job 和 `push-dockerhub` job 之间插入 `build-fc` job（依赖 `build-images` 完成）。找到 `push-dockerhub:` 这一行，在其上方插入：

```yaml
  build-fc:
    runs-on: ubuntu-latest
    needs: build-images
    steps:
      - name: Free disk space
        uses: jlumbroso/free-disk-space@main
        with:
          tool-cache: false
          android: true
          dotnet: true
          haskell: true
          large-packages: true
          docker-images: true
          swap-storage: true

      - name: Checkout
        uses: actions/checkout@v5

      - name: Extract shared modules
        run: npm run extract:shared

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v4

      - name: Build FC image
        uses: docker/build-push-action@v7
        with:
          context: .
          file: local-server/Dockerfile.sandbox.fc
          tags: ${{ env.IMAGE_NAME }}:fc
          outputs: type=docker,dest=/tmp/image-fc.tar
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Test FC image
        run: |
          docker load -i /tmp/image-fc.tar
          docker run --rm -d --name test-fc ${{ env.IMAGE_NAME }}:fc
          sleep 5
          docker exec test-fc curl -f http://localhost:9000/api/sandbox/health
          docker stop test-fc

      - name: Upload FC image artifact
        uses: actions/upload-artifact@v5
        with:
          name: image-fc
          path: /tmp/image-fc.tar
          retention-days: 1
```

- [ ] **Step 2: 在 push-acr job 后新增 push-acr-fc job**

```yaml
  push-acr-fc:
    runs-on: ubuntu-latest
    needs: build-fc
    if: ${{ always() && github.event.inputs.skip_acr != 'true' && needs.build-fc.result == 'success' }}
    steps:
      - name: Free disk space
        uses: jlumbroso/free-disk-space@main
        with:
          tool-cache: false
          android: true
          dotnet: true
          haskell: true
          large-packages: true
          docker-images: true
          swap-storage: true

      - name: Download FC image artifact
        uses: actions/download-artifact@v5
        with:
          name: image-fc
          path: /tmp

      - name: Load image
        run: docker load -i /tmp/image-fc.tar

      - name: Login to Alibaba Cloud ACR
        uses: docker/login-action@v4
        with:
          registry: crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Tag and push FC image to ACR
        run: |
          ACR_REGISTRY="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com"
          ACR_NS="fenixsoft"
          IMAGE_NAME="dmla-sandbox"

          docker tag ${IMAGE_NAME}:fc \
            ${ACR_REGISTRY}/${ACR_NS}/${IMAGE_NAME}:fc
          docker push ${ACR_REGISTRY}/${ACR_NS}/${IMAGE_NAME}:fc

          echo "✅ Pushed FC image to ACR:"
          echo "  - ${ACR_REGISTRY}/${ACR_NS}/${IMAGE_NAME}:fc"
```

- [ ] **Step 3: 新增 update-fc job**

```yaml
  update-fc:
    runs-on: ubuntu-latest
    needs: [push-acr, push-acr-fc]
    if: ${{ always() && needs.push-acr-fc.result == 'success' }}
    steps:
      - name: Setup Aliyun CLI
        run: curl -fsSL https://aliyuncli.alicdn.com/install.sh | bash

      - name: Configure Aliyun CLI
        run: |
          aliyun configure set \
            --access-key-id ${{ secrets.ALIBABA_CLOUD_ACCESS_KEY_ID }} \
            --access-key-secret ${{ secrets.ALIBABA_CLOUD_ACCESS_KEY_SECRET }} \
            --region cn-hangzhou

      - name: Update FC function image
        run: |
          IMAGE_URI="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc"
          aliyun fc update-function \
            --service-name dmla \
            --function-name sandbox-cpu \
            --code "{\"imageUri\": \"$IMAGE_URI\"}"
          echo "✅ FC function updated"
```

- [ ] **Step 4: 更新 summary job 的 needs 字段**

找到 `summary` job 的 `needs` 行，改为：

```yaml
needs: [build-images, build-fc, push-dockerhub, push-acr, push-acr-fc, update-fc]
```

同时在 summary 输出中增加 FC 镜像信息：

```yaml
echo "#### 阿里云 FC" >> $GITHUB_STEP_SUMMARY
echo "- crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/publish-docker.yml
git commit -m "ci: publish-docker 新增 build-fc、push-acr-fc、update-fc job

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 本地构建并推送 FC 镜像 + 首次创建 FC 函数

**Files:**
- None（纯 CLI 操作）

**Prerequisites:**
- 已安装 Docker 且 Docker daemon 运行中
- 已安装 aliyun CLI
- 有 ACR 仓库推送权限

> **注意：** 此任务在本地执行一次即可。后续镜像更新由 CI 自动完成。

- [ ] **Step 1: 构建 FC 镜像**

```bash
cd /root/dmla
npm run extract:shared
docker build -f local-server/Dockerfile.sandbox.fc -t dmla-sandbox:fc .
```

Expected: 构建成功，无错误

- [ ] **Step 2: 本地验证 FC 镜像**

```bash
docker run --rm -d --name test-fc dmla-sandbox:fc
sleep 10
# 健康检查
curl -s http://localhost:9000/api/sandbox/health
# 预期输出: {"status": "ok", "mode": "fc", "timestamp": ...}

# 代码执行测试
curl -s -X POST http://localhost:9000/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1+1)"}'
# 预期输出: {"success": true, "outputs": [...], "executionTime": ...}

docker stop test-fc
```

- [ ] **Step 3: 登录 ACR 并推送**

```bash
# 登录 ACR（使用已有凭据）
ACR_REGISTRY="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com"

docker login --username=<你的ACR用户名> --password=<你的ACR密码> ${ACR_REGISTRY}

docker tag dmla-sandbox:fc ${ACR_REGISTRY}/fenixsoft/dmla-sandbox:fc
docker push ${ACR_REGISTRY}/fenixsoft/dmla-sandbox:fc
```

Expected: 推送成功

- [ ] **Step 4: 安装并配置 aliyun CLI**

```bash
# 安装（如未安装）
curl -fsSL https://aliyuncli.alicdn.com/install.sh | bash

# 配置认证
aliyun configure set \
  --access-key-id REDACTED_KEY_ID \
  --access-key-secret REDACTED_KEY_SECRET \
  --region cn-hangzhou
```

- [ ] **Step 5: 创建 FC 服务**

```bash
aliyun fc create-service --service-name dmla
```

如果提示 "service already exists"，忽略即可。

- [ ] **Step 6: 创建 FC 函数**

```bash
aliyun fc create-function \
  --service-name dmla \
  --function-name sandbox-cpu \
  --runtime custom-container \
  --handler not-used \
  --cpu 1 \
  --memory-size 2048 \
  --disk-size 512 \
  --timeout 100 \
  --code '{"imageUri": "crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc"}'
```

Expected: `{"functionName": "sandbox-cpu", ...}` 包含函数 ARN

- [ ] **Step 7: 创建 HTTP 触发器**

```bash
aliyun fc create-trigger \
  --service-name dmla \
  --function-name sandbox-cpu \
  --trigger-name http-trigger \
  --trigger-type http \
  --trigger-config '{"authType":"anonymous","methods":["POST","GET"]}'
```

Expected: 返回触发器信息，包含 `urlInternet`（公网 URL）

- [ ] **Step 8: 记录 FC 公网 URL**

复制 Step 7 输出中的 `urlInternet` 字段值，格式类似：
```
https://<account-id>.<region>.fc.aliyuncs.com/2016-08-15/proxy/dmla/sandbox-cpu/
```

将此 URL 记下，后续配置到 `Settings.vue`。

- [ ] **Step 9: 验证 FC 函数**

```bash
# 健康检查
curl -s "https://<url>/api/sandbox/health"
# 预期: {"status": "ok", "mode": "fc", ...}

# 代码执行
curl -s -X POST "https://<url>/api/sandbox/run" \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello FC!\")"}'
# 预期: {"success": true, "outputs": [{...}], "executionTime": ...}
```

注意：如果首次调用触发了冷启动，响应会比较慢（可能 10-30 秒），后续调用会快很多。

- [ ] **Step 10: 记录完成状态**

将 FC 公网 URL 写入 Settings.vue 的 `FC_DEFAULT_URL` 常量（在 Task 5 中完成）。

---

### Task 5: 更新 Settings.vue（FC/自定义模式选择）

**Files:**
- Modify: `docs/.vuepress/theme/components/Settings.vue`

**Interfaces:**
- Consumes: `getSiteConfig()`, `saveSiteConfig()` from `../utils/configMigration.js`
- Produces: `sandboxMode` 字段 `'fc' | 'custom'`，`sandboxEndpoint` 字段

> **注意：** 修改前请先执行 `Task 4` 获取 FC 公网 URL，替换下面代码中 `FC_DEFAULT_URL` 占位符的值。

- [ ] **Step 1: 修改 Settings.vue template 的沙箱服务 Tab**

找到 `<!-- 沙箱服务配置 Tab -->` 部分（约第 31 行），替换为：

```vue
<!-- 沙箱服务配置 Tab -->
<div v-show="activeTab === 'sandbox'" class="tab-content">
  <div class="form-group">
    <label>服务模式</label>
    <div class="mode-selector">
      <label class="mode-option" :class="{ active: sandboxMode === 'fc' }">
        <input
          type="radio"
          v-model="sandboxMode"
          value="fc"
          @change="onModeChange"
        />
        <span class="mode-label">FC（默认）</span>
        <span class="mode-hint">云端 Serverless，闲置免费</span>
      </label>
      <label class="mode-option" :class="{ active: sandboxMode === 'custom' }">
        <input
          type="radio"
          v-model="sandboxMode"
          value="custom"
          @change="onModeChange"
        />
        <span class="mode-label">自定义地址</span>
        <span class="mode-hint">自建沙箱服务</span>
      </label>
    </div>
  </div>

  <div v-if="sandboxMode === 'fc'" class="form-group">
    <label for="sandbox-endpoint">FC 沙箱地址</label>
    <input
      id="sandbox-endpoint"
      :value="FC_DEFAULT_URL"
      type="text"
      readonly
      class="readonly-input"
    />
    <p class="help-text">阿里云函数计算提供，无需自行部署</p>
  </div>

  <div v-if="sandboxMode === 'custom'" class="form-group">
    <label for="sandbox-endpoint">服务地址</label>
    <input
      id="sandbox-endpoint"
      v-model="endpoint"
      type="url"
      placeholder="http://localhost:3001"
      @input="resetStatus"
    />
    <p class="help-text">用于执行教程中的 Python 代码</p>
  </div>

  <div class="connection-status">
    <span class="status-label">连接状态:</span>
    <span class="status-value" :class="statusClass">
      <span class="status-dot"></span>
      {{ statusText }}
    </span>
  </div>
</div>
```

- [ ] **Step 2: 修改 Settings.vue script 部分**

在 `<script setup>` 中添加 FC 相关变量（约第 103 行，在 `const testing = ref(false)` 之后添加）：

```javascript
// FC 模式相关
// ⚠️ 部署后替换为 FC HTTP 触发器的实际公网 URL
const FC_DEFAULT_URL = 'https://REPLACE_WITH_FC_URL'

// 沙箱模式: 'fc' | 'custom'
const sandboxMode = ref('fc')

// 原 endpoint 变量保留（用于自定义模式）
const endpoint = ref('')
```

修改 `loadConfig()` 函数（约第 137 行）：

```javascript
function loadConfig() {
  const config = getSiteConfig()
  sandboxMode.value = config.sandboxMode || 'fc'
  endpoint.value = config.sandboxEndpoint || 'http://localhost:3001'

  // 如果处于 FC 模式，endpoint 固定为 FC URL
  if (sandboxMode.value === 'fc') {
    endpoint.value = FC_DEFAULT_URL
  }

  selectedTheme.value = config.highlightTheme || DEFAULT_THEME
}
```

新增 `onModeChange` 函数（在 `loadConfig` 之后）：

```javascript
function onModeChange() {
  if (sandboxMode.value === 'fc') {
    endpoint.value = FC_DEFAULT_URL
  }
  resetStatus()
}
```

修改 `save()` 函数（约第 189 行）：

```javascript
function save() {
  const config = {
    sandboxMode: sandboxMode.value,
    sandboxEndpoint: sandboxMode.value === 'fc' ? FC_DEFAULT_URL : (endpoint.value.trim() || 'http://localhost:3001'),
    highlightTheme: selectedTheme.value
  }

  saveSiteConfig(config)

  if (typeof window !== 'undefined') {
    window.__SITE_CONFIG__ = config
    window.dispatchEvent(new CustomEvent('site-config-changed', { detail: config }))
  }

  emit('save', config)
  close()
}
```

修改 `watch(() => props.visible, ...)`（约第 221 行，确保打开设置页时自动检测连接）：

```javascript
watch(() => props.visible, (newVal) => {
  if (newVal) {
    loadConfig()
    testConnection()  // 自动测试连接
  }
})
```

- [ ] **Step 3: 修改 Settings.vue style 部分（新增模式选择器样式）**

在 `</style>` 之前添加：

```css
.mode-selector {
  display: flex;
  gap: 12px;
}

.mode-option {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #E4E4E7;
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.mode-option input[type="radio"] {
  display: none;
}

.mode-option:hover {
  border-color: #A1A1AA;
}

.mode-option.active {
  border-color: #2563EB;
  background: #EFF6FF;
}

.mode-label {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: #18181B;
  margin-bottom: 4px;
}

.mode-hint {
  display: block;
  font-size: 12px;
  color: #71717A;
}

.readonly-input {
  background: #F4F4F5 !important;
  color: #71717A !important;
  cursor: not-allowed;
}
```

- [ ] **Step 4: Commit**

```bash
git add docs/.vuepress/theme/components/Settings.vue
git commit -m "feat: Settings 新增 FC/自定义双模式沙箱选择

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 更新 sandbox-config.js 和 client.js

**Files:**
- Modify: `docs/.vuepress/plugins/runnable-code/sandbox-config.js`
- Modify: `docs/.vuepress/plugins/runnable-code/client.js`

- [ ] **Step 1: 修改 sandbox-config.js —— 处理 sandboxMode 字段**

修改 `setSandboxConfig()` 函数（约第 118 行），在合并时也保留 `sandboxMode`：

```javascript
export function setSandboxConfig(config) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    const existing = localStorage.getItem(STORAGE_KEY)
    const existingConfig = existing ? JSON.parse(existing) : {}

    const newConfig = {
      ...existingConfig,
      sandboxEndpoint: config.endpoint || DEFAULT_ENDPOINT,
      sandboxMode: config.sandboxMode || 'custom'
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig))

    window.__SANDBOX_CONFIG__ = { endpoint: newConfig.sandboxEndpoint, mode: newConfig.sandboxMode }
    window.__SITE_CONFIG__ = newConfig
  } catch (error) {
    console.error('[Sandbox Config] 保存配置失败:', error)
  }
}
```

无需修改 `getSandboxConfig()` 和 `getSandboxEndpoint()`，它们继续从 localStorage 读取 `sandboxEndpoint`。

- [ ] **Step 2: 修改 client.js —— 调整连接超时**

找到 `CONNECTION_TIMEOUT` 常量定义（约第 X 行），修改为：

```javascript
const CONNECTION_TIMEOUT = 20000  // 从 10000ms 调整为 20000ms，适应 FC 冷启动
```

- [ ] **Step 3: Commit**

```bash
git add docs/.vuepress/plugins/runnable-code/sandbox-config.js docs/.vuepress/plugins/runnable-code/client.js
git commit -m "fix: 支持 sandboxMode 配置，连接超时延长至 20 秒适应 FC 冷启动

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 创建端到端等价性测试

**Files:**
- Create: `local-server/tests/fc-sandbox-e2e.test.py`

**Interfaces:**
- Consumes: `FC_ENDPOINT` 环境变量（FC HTTP 触发器 URL），`CPU_SANDBOX_ENDPOINT` 环境变量（CPU 沙箱地址）
- Produces: 测试结果报告

- [ ] **Step 1: 编写测试脚本**

```python
"""
FC vs CPU Sandbox 端到端等价性测试
验证 FC 部署的沙箱与本地 CPU 沙箱输出一致

用法:
    FC_ENDPOINT="https://xxx.fc.aliyuncs.com/..." \
    CPU_SANDBOX_ENDPOINT="http://localhost:3001" \
    python3 local-server/tests/fc-sandbox-e2e.test.py
"""

import requests
import json
import base64
import sys
import os

FC_URL = os.environ.get('FC_ENDPOINT', 'http://localhost:9000')
CPU_URL = os.environ.get('CPU_SANDBOX_ENDPOINT', 'http://localhost:3001')

TEST_TIMEOUT = 120  # HTTP 请求超时（秒），含 FC 冷启动

PASS = 0
FAIL = 0
SKIP = 0


def run_fc(code, timeout=60):
    """在 FC 上执行代码"""
    resp = requests.post(
        FC_URL + '/api/sandbox/run',
        json={'code': code, 'timeout': timeout},
        timeout=TEST_TIMEOUT
    )
    return resp.json()


def run_cpu(code, timeout=60):
    """在 CPU 沙箱上执行代码"""
    resp = requests.post(
        CPU_URL + '/api/sandbox/run',
        json={'code': code, 'timeout': timeout},
        timeout=TEST_TIMEOUT
    )
    return resp.json()


def test(name, code, assertions):
    """运行单个测试用例"""
    global PASS, FAIL, SKIP

    try:
        fc_result = run_fc(code)
        cpu_result = run_cpu(code)
    except requests.exceptions.ConnectionError as e:
        print(f"  ⚠ SKIP: 无法连接 ({e})")
        SKIP += 1
        return
    except Exception as e:
        print(f"  ✗ FAIL: 请求异常 ({e})")
        FAIL += 1
        return

    try:
        assertions(fc_result, cpu_result)
        print(f"  ✓ PASS")
        PASS += 1
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        print(f"    FC:   {json.dumps(fc_result, ensure_ascii=False)[:200]}")
        print(f"    CPU:  {json.dumps(cpu_result, ensure_ascii=False)[:200]}")
        FAIL += 1


def assert_success_equal(fc, cpu):
    assert fc.get('success') == cpu.get('success'), \
        f"success 不一致: FC={fc.get('success')}, CPU={cpu.get('success')}"


def assert_output_count_equal(fc, cpu):
    assert len(fc.get('outputs', [])) == len(cpu.get('outputs', [])), \
        f"outputs 数量不一致: FC={len(fc.get('outputs', []))}, CPU={len(cpu.get('outputs', []))}"


def assert_image_present(fc, cpu):
    """验证图片输出中 image/png 存在且为有效 base64"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        display_datas = [o for o in outputs if o.get('type') == 'display_data']
        assert len(display_datas) > 0, "没有 display_data 输出"
        for dd in display_datas:
            png_data = dd.get('data', {}).get('image/png', '')
            assert png_data, "image/png 字段为空"
            # 验证为有效 base64
            try:
                decoded = base64.b64decode(png_data)
                assert decoded[:8] == b'\x89PNG\r\n\x1a\n', "image/png 不是有效 PNG 文件头"
            except Exception as e:
                raise AssertionError(f"base64 解码失败: {e}")


def assert_no_font_warnings(fc, cpu):
    """验证 stderr 中没有字体缺失警告"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        stderr_outs = [o for o in outputs
                       if o.get('type') == 'stream' and o.get('name') == 'stderr']
        stderr_text = ''.join(o.get('text', '') for o in stderr_outs)
        assert 'does not have a glyph for' not in stderr_text, \
            f"存在字体缺失警告: {stderr_text[:200]}"


def assert_stdout_contains(fc, cpu, text):
    """验证 stdout 中包含指定文本"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        stdout_text = ''.join(
            o.get('text', '') for o in outputs
            if o.get('type') == 'stream' and o.get('name') == 'stdout'
        )
        assert text in stdout_text, f"stdout 中未找到 '{text}': {stdout_text[:200]}"


def assert_error_type(fc, cpu, ename):
    """验证错误类型"""
    for outputs in [fc.get('outputs', []), cpu.get('outputs', [])]:
        errors = [o for o in outputs if o.get('type') == 'error']
        assert len(errors) > 0, "没有 error 输出"
        assert errors[0].get('ename') == ename, \
            f"错误类型不匹配: 期望 {ename}, 实际 {errors[0].get('ename')}"


def assert_execution_time_reasonable(fc, cpu, min_time=0.5):
    """验证执行时间合理"""
    assert fc.get('executionTime', 0) >= min_time, \
        f"FC 执行时间 {fc.get('executionTime')} < {min_time}"
    assert cpu.get('executionTime', 0) >= min_time, \
        f"CPU 执行时间 {cpu.get('executionTime')} < {min_time}"


# ═══════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════

SETUP_MPL = """
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
"""

if __name__ == '__main__':
    print("=" * 60)
    print("FC vs CPU Sandbox 等价性测试")
    print(f"FC:  {FC_URL}")
    print(f"CPU: {CPU_URL}")
    print("=" * 60)

    # 1. 纯文本输出
    print("\n[1] 纯文本输出")
    test("print_text", 'print("Hello, DMLA!")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'Hello, DMLA!')
    ])

    # 2. 多行输出
    print("\n[2] 多行输出")
    test("multi_line", 'for i in range(3): print(f"Line {i}")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'Line 0'),
        assert_stdout_contains(fc, cpu, 'Line 2')
    ])

    # 3. 表达式结果
    print("\n[3] 表达式结果")
    test("expression", '3.14 * 2', lambda fc, cpu: [
        assert_success_equal(fc, cpu)
    ])

    # 4. 图片输出
    print("\n[4] matplotlib 图片输出")
    test("matplotlib_plot", SETUP_MPL + """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Test Plot")
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu)
    ])

    # 5. 中文标题图片
    print("\n[5] 中文图片（含字体检查）")
    test("chinese_plot", SETUP_MPL + """
import matplotlib.pyplot as plt
plt.figure()
plt.title("中文标题测试")
plt.text(0.5, 0.5, "你好，世界", ha='center', transform=plt.gca().transAxes)
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu),
        assert_no_font_warnings(fc, cpu)
    ])

    # 6. 运行时异常
    print("\n[6] 运行时异常 (ZeroDivisionError)")
    test("runtime_error", 'x = 1/0', lambda fc, cpu: [
        assert_error_type(fc, cpu, 'ZeroDivisionError')
    ])

    # 7. 语法异常
    print("\n[7] 语法异常 (SyntaxError)")
    test("syntax_error", 'if True print("oops")', lambda fc, cpu: [
        lambda fc, _: [o for o in fc.get('outputs', []) if o.get('type') == 'error'],
        lambda _, cpu: [o for o in cpu.get('outputs', []) if o.get('type') == 'error']
    ])

    # 8. NumPy 计算
    print("\n[8] NumPy 浮点计算")
    test("numpy_calc", """
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
print(f"mean={arr.mean():.6f}")
print(f"std={arr.std():.6f}")
""", lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_stdout_contains(fc, cpu, 'mean=2.000000'),
        assert_stdout_contains(fc, cpu, 'std=0.816497')
    ])

    # 9. 执行时间记录
    print("\n[9] 执行时间记录")
    test("exec_time", 'import time; time.sleep(0.5); print("done")', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        assert_execution_time_reasonable(fc, cpu, 0.5)
    ])

    # 10. 无输出执行
    print("\n[10] 无输出执行")
    test("no_output", 'x = 1 + 1', lambda fc, cpu: [
        assert_success_equal(fc, cpu)
    ])

    # 11. 大数据流输出
    print("\n[11] 大数据流输出（10000 字符）")
    test("large_output", 'print("x" * 10000)', lambda fc, cpu: [
        assert_success_equal(fc, cpu),
        lambda fc, _: (len(''.join(
            o.get('text', '') for o in fc.get('outputs', [])
            if o.get('type') == 'stream' and o.get('name') == 'stdout'
        )) >= 10000) or (_ for _ in ()).throw(AssertionError("FC 输出被截断")),
        lambda _, cpu: (len(''.join(
            o.get('text', '') for o in cpu.get('outputs', [])
            if o.get('type') == 'stream' and o.get('name') == 'stdout'
        )) >= 10000) or (_ for _ in ()).throw(AssertionError("CPU 输出被截断"))
    ])

    # 12. imshow 图片
    print("\n[12] matplotlib imshow 图片")
    test("imshow", SETUP_MPL + """
import matplotlib.pyplot as plt
import numpy as np
img = np.random.rand(10, 10)
plt.imshow(img, cmap='viridis')
plt.colorbar()
plt.show()
""", lambda fc, cpu: [
        assert_image_present(fc, cpu)
    ])

    # ═══════════════════════════════════════════
    # 结果汇总
    # ═══════════════════════════════════════════
    total = PASS + FAIL + SKIP
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败, {SKIP} 跳过 (共 {total})")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
```

- [ ] **Step 2: Commit**

```bash
git add local-server/tests/fc-sandbox-e2e.test.py
git commit -m "test: FC vs CPU 沙箱端到端等价性测试（12 用例）

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 端到端集成验证

**Prerequisites:**
- Task 4 已完成（FC 函数已创建且可访问）
- Task 5 的 `FC_DEFAULT_URL` 已填写正确值

- [ ] **Step 1: 运行完整测试套件**

```bash
# 设置正确的 FC URL（替换为 Task 4 获取的实际 URL）
export FC_ENDPOINT="https://<account-id>.cn-hangzhou.fc.aliyuncs.com/2016-08-15/proxy/dmla/sandbox-cpu"

# 启动本地 CPU 沙箱用于对比测试
lsof -ti:3001 -sTCP:LISTEN | xargs kill 2>/dev/null
cd /root/dmla
npm run server &
sleep 5

# 运行 E2E 测试
python3 local-server/tests/fc-sandbox-e2e.test.py
```

Expected: 12/12 通过

- [ ] **Step 2: 清理本地服务**

```bash
lsof -ti:3001 -sTCP:LISTEN | xargs kill 2>/dev/null
```

- [ ] **Step 3: 验证前端构建**

```bash
cd /root/dmla
npm run build
```

Expected: 构建成功，无错误

- [ ] **Step 4: Commit 所有变更**

```bash
git add -A
git status
git commit -m "chore: FC Serverless 迁移完成，集成验证通过

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 5: 安全清理 —— 禁用/轮换 AccessKey**

部署验证完成后，**务必**在阿里云 RAM 控制台禁用或删除本次使用的 AccessKey（`REDACTED_KEY_ID`），生成新 Key 供 CI 使用。当前 AccessKey 已在聊天记录中暴露。

---

## 任务依赖关系

```
Task 1 (fc_handler.py)
    │
Task 2 (Dockerfile.sandbox.fc)
    │
Task 3 (CI workflow) ──┐
    │                  │
Task 4 (本地构建+推送+FC创建) ←── 需先完成 Task 1, 2
    │                  │
    ├──────────────────┤
    │                  │
Task 5 (Settings.vue) Task 6 (sandbox-config + client)
    │                  │
    └──────┬───────────┘
           │
Task 7 (E2E 测试)
           │
Task 8 (集成验证 + 安全清理)
```

## 执行顺序建议

1. **Tasks 1-3** 可以按顺序逐个执行（Task 3 依赖 1、2 的文件存在）
2. **Task 4** 在 1-3 完成后执行，本地构建推送 + FC CLI 创建
3. **Tasks 5-6** 可并行执行
4. **Task 7** 在 4 + 5 + 6 之后执行
5. **Task 8** 最后执行，包含完整验证和安全清理
