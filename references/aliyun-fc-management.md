# 阿里云 FC 函数计算管理

## 认证凭据

阿里云 AccessKey 存储在 `~/id.key`，格式为第一行 AccessKey ID、第二行 AccessKey Secret。

每次使用 aliyun CLI 前，从该文件读取凭据并配置：

```bash
ACCESS_KEY_ID=$(sed -n '1p' ~/id.key)
ACCESS_KEY_SECRET=$(sed -n '2p' ~/id.key)
aliyun configure set --access-key-id "$ACCESS_KEY_ID" --access-key-secret "$ACCESS_KEY_SECRET" --region cn-hangzhou
```

## 已部署的 FC 资源

| 资源 | 名称 | 说明 |
|------|------|------|
| 函数 | `sandbox-cpu` | CPU 沙箱执行服务，Custom Container 运行时 |
| 触发器 | `http-trigger` | HTTP 触发器，匿名访问，GET/POST |
| 公网 URL | `https://sandbox-cpu-dcheerjqde.cn-hangzhou.fcapp.run` | 前端 Settings 中的 FC 默认地址 |
| 镜像 | `dmla-sandbox:fc` | ACR 上的 FC 专用精简镜像 |

## FC 函数配置

| 配置项 | 值 |
|--------|-----|
| vCPU | 1 核 |
| 内存 | 1024 MB |
| 磁盘 | 512 MB |
| 超时 | 100 秒 |
| 端口 | 9000 |
| 实例并发 | 1 |

## 常用操作

**查看函数状态：**
```bash
aliyun fc GET /2023-03-30/functions/sandbox-cpu
```

**更新函数镜像（本地构建推送后）：**
```bash
# 构建并推送
docker build -f local-server/Dockerfile.sandbox.fc -t dmla-sandbox:fc .
ACR="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com"
cat ~/id.key  # 用于 docker login 密码
docker tag dmla-sandbox:fc ${ACR}/fenixsoft/dmla-sandbox:fc
docker push ${ACR}/fenixsoft/dmla-sandbox:fc

# 更新 FC 函数
aliyun fc update-function \
  --function-name sandbox-cpu \
  --custom-container-config "image=${ACR}/fenixsoft/dmla-sandbox:fc port=9000"
```

**查看触发器公网 URL：**
```bash
aliyun fc GET /2023-03-30/functions/sandbox-cpu/triggers/http-trigger
```

**FC API 版本说明：**
- 使用 FC 3.0 API（`2023-03-30`），无需管理 Service 概念
- 需要安装插件：`aliyun plugin install --names aliyun-cli-fc`
- 镜像配置在 `customContainerConfig` 字段，而非 `code`
- ACR 认证：`docker login --username=icyfenix crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com`

**镜像预热说明：**
- FC 首次拉取新镜像需要预热（`imagePrewarmStatus: InProgress`）
- 3.66GB 镜像预热约需 5-10 分钟
- 更新已存在的镜像 tag 时，仅需拉取变更层
- 预热期间函数仍可能响应 `PreconditionFailed`，需等待 `lastUpdateStatus: Successful`
- `accelerationType: Default` 仅 ACR 企业版支持，个人版默认为全量拉取

**镜像更新流程（每次代码变更后）：**
```bash
# 1. 构建并推送（同时打 :cpu 和时间戳 tag）
docker build --provenance=false --platform linux/amd64 \
  -f local-server/Dockerfile.sandbox.cpu -t dmla-sandbox:cpu .
ACR="crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com"
echo '[efflying]' | docker login --username=icyfenix --password-stdin ${ACR}
docker tag dmla-sandbox:cpu ${ACR}/fenixsoft/dmla-sandbox:cpu
docker push ${ACR}/fenixsoft/dmla-sandbox:cpu

# 推送时间戳 tag（FC 需要唯一 tag 才能触发重新解析）
FC_TAG="cpu-$(date +%Y%m%d-%H%M%S)"
docker tag dmla-sandbox:cpu ${ACR}/fenixsoft/dmla-sandbox:${FC_TAG}
docker push ${ACR}/fenixsoft/dmla-sandbox:${FC_TAG}

# 2. 用时间戳 tag 更新 FC 函数
CONFIG='{"image":"crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:'${FC_TAG}'","port":9000,"command":["python3","/workspace/fc_handler.py"]}'
aliyun fc update-function --function-name sandbox-cpu --custom-container-config "$CONFIG"

# 3. 等待预热完成（lastUpdateStatus == "Successful"）
aliyun fc GET /2023-03-30/functions/sandbox-cpu
```

**重要：FC 镜像 tag 缓存机制**
- FC 对同一 image URI 会缓存解析结果，重复推送同 tag 不会触发新部署
- 必须每次使用唯一 tag（如 cpu-YYYYMMDD-HHMMSS）才能强制 FC 重新拉取
- `:cpu` tag 仅供 docker pull 使用，FC 部署使用时间戳 tag

**镜像构建注意事项（重要）：**
- 构建 FC 镜像时必须禁用 provenance：`docker build --provenance=false --platform linux/amd64`
- FC 不支持 OCI image index 格式（`application/vnd.oci.image.index.v1+json`）
- 不支持时会导致 `invalid image, platform of image is unknown/unknown` 错误
- GitHub Actions 的 `docker/build-push-action@v7` 默认生成 OCI 格式，需要在 CI 中也禁用 provenance

**常用速查**：

```bash
# 配置凭据
ACCESS_KEY_ID=$(sed -n '1p' ~/id.key)
ACCESS_KEY_SECRET=$(sed -n '2p' ~/id.key)
aliyun configure set --access-key-id "$ACCESS_KEY_ID" --access-key-secret "$ACCESS_KEY_SECRET" --region cn-hangzhou

# 查看函数状态
aliyun fc GET /2023-03-30/functions/sandbox-cpu

# 查看公网 URL
aliyun fc GET /2023-03-30/functions/sandbox-cpu/triggers/http-trigger
```