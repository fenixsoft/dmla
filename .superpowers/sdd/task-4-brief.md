# Task 4: 本地构建推送 FC 镜像 + 首次创建 FC 函数

## 目标

在本地完成 FC 镜像构建和推送，并使用 aliyun CLI 首次创建 FC 服务、函数和 HTTP 触发器。

## 前置条件

- 已安装 Docker 且 daemon 运行中
- 已安装 aliyun CLI
- 有 ACR 推送权限
- Task 1 和 Task 2 的文件已创建

## 步骤

### Step 1: 构建 FC 镜像

```bash
cd /root/dmla
npm run extract:shared
docker build -f local-server/Dockerfile.sandbox.fc -t dmla-sandbox:fc .
```

### Step 2: 本地验证 FC 镜像

```bash
docker run --rm -d --name test-fc dmla-sandbox:fc
sleep 10
# 健康检查
curl -s http://localhost:9000/api/sandbox/health
# 预期: {"status": "ok", ...}

# 代码执行测试
curl -s -X POST http://localhost:9000/api/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1+1)"}'
# 预期: {"success": true, ...}

docker stop test-fc
```

### Step 3: 登录 ACR 并推送

```bash
ACR_REGISTRY="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com"

docker login --username=<ACR_USERNAME> --password=<ACR_PASSWORD> ${ACR_REGISTRY}

docker tag dmla-sandbox:fc ${ACR_REGISTRY}/fenixsoft/dmla-sandbox:fc
docker push ${ACR_REGISTRY}/fenixsoft/dmla-sandbox:fc
```

注：ACR 凭据已通过环境变量 `ACR_USERNAME`、`ACR_PASSWORD` 注入到当前会话。

### Step 4: 安装并配置 aliyun CLI

```bash
# 安装（如未安装）
curl -fsSL https://aliyuncli.alicdn.com/install.sh | bash

# 配置认证
aliyun configure set \
  --access-key-id REDACTED_KEY_ID \
  --access-key-secret REDACTED_KEY_SECRET \
  --region cn-hangzhou
```

### Step 5: 创建 FC 服务

```bash
aliyun fc create-service --service-name dmla
```

如果返回 "service already exists" 则忽略。

### Step 6: 创建 FC 函数

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

### Step 7: 创建 HTTP 触发器

```bash
aliyun fc create-trigger \
  --service-name dmla \
  --function-name sandbox-cpu \
  --trigger-name http-trigger \
  --trigger-type http \
  --trigger-config '{"authType":"anonymous","methods":["POST","GET"]}'
```

### Step 8: 记录并验证 FC 公网 URL

从 Step 7 输出中提取 `urlInternet` 值。

```bash
# 验证健康检查
curl -s "https://<urlInternet>/api/sandbox/health"

# 验证代码执行
curl -s -X POST "https://<urlInternet>/api/sandbox/run" \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello FC!\")"}'
```

### Step 9: 将 FC URL 写入 Task 5 brief

将获取到的 `urlInternet` 值（格式：`https://xxx.cn-hangzhou.fc.aliyuncs.com/2016-08-15/proxy/dmla/sandbox-cpu/`）写入文件 `/root/dmla/.superpowers/sdd/fc-url.txt`，供 Task 5 使用。

## 全局约束

- FC 地域 cn-hangzhou，服务名 dmla，函数名 sandbox-cpu
- CPU 1核，内存 2048MB，磁盘 512MB，超时 100秒
- 运行时 custom-container
- HTTP 触发器 authType=anonymous, methods=POST+GET
