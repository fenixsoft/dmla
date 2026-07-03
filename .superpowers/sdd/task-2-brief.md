# Task 2: 创建 FC 专用 Dockerfile

## 目标

创建 `local-server/Dockerfile.sandbox.fc` —— FC 函数计算专用的精简 Docker 镜像定义，相比 CPU 版移除 pandas 和 lmdb。

## 文件

- Create: `local-server/Dockerfile.sandbox.fc`

## 接口

- Consumes: `local-server/src/fc_handler.py`, `local-server/src/kernel_runner.py`, `local-server/src/dmla_progress.py`, `local-server/shared/`
- Produces: `dmla-sandbox:fc` Docker 镜像

## 完整代码

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

# 安装科学计算库 - 使用清华镜像（移除 pandas 和 lmdb）
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

## 对比 CPU Dockerfile 的变更

**移除的 pip 包：** pandas, lmdb
**新增的 COPY：** fc_handler.py
**新增的环境变量：** FC_SERVER_PORT=9000
**CMD 变更：** 从 `CMD ["python3"]` 改为 `CMD ["python3", "/workspace/fc_handler.py"]`
**新增 HEALTHCHECK**

## 验证步骤

1. 验证 Dockerfile 语法：`docker build --check -f local-server/Dockerfile.sandbox.fc .` (如果 docker 版本不支持 --check，可跳过此步骤)

## 全局约束

- 镜像 tag：`dmla-sandbox:fc`
- FC handler 端口 9000
- 基础镜像 python:3.11-slim，与 CPU 版一致
