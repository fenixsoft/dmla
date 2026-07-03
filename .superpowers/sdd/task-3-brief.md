# Task 3: 修改 CI 工作流（publish-docker.yml）

## 目标

在现有的 `.github/workflows/publish-docker.yml` 中新增三个 job：
1. **build-fc**: 构建 FC 精简镜像
2. **push-acr-fc**: 推送 fc tag 到阿里云 ACR
3. **update-fc**: 调用 aliyun CLI 更新 FC 函数镜像

## 文件

- Modify: `.github/workflows/publish-docker.yml`

## 操作步骤

### Step 1: 新增 build-fc job

在文件末尾（`summary` job 之前），插入 `build-fc` job。注意：需要 `needs: build-images`，并且需要 checkout + extract:shared + setup Docker Buildx。

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

### Step 2: 新增 push-acr-fc job

在现有 `push-acr` job 之后新增：

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

### Step 3: 新增 update-fc job

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

### Step 4: 更新 summary job

找到 `summary` job 的 `needs` 行，追加 `build-fc`, `push-acr-fc`, `update-fc`：

```yaml
    needs: [build-images, build-fc, push-dockerhub, push-acr, push-acr-fc, update-fc]
```

并在 summary 输出中增加 FC 信息（找到现有的 ACR summary 输出，在下面添加）：

```yaml
          echo "#### 阿里云 FC（Serverless 沙箱）" >> $GITHUB_STEP_SUMMARY
          echo "- crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc" >> $GITHUB_STEP_SUMMARY
```

## 重要说明

- 三个新 job 需插入到 `summary` job 之前
- build-fc 需要在 checkout 后先 `npm run extract:shared`（生成 shared 模块）
- update-fc 中的 service-name 和 function-name 硬编码为 `dmla` 和 `sandbox-cpu`
- region 硬编码为 `cn-hangzhou`
- AccessKey 使用 GitHub Secrets（`ALIBABA_CLOUD_ACCESS_KEY_ID` 和 `ALIBABA_CLOUD_ACCESS_KEY_SECRET`）

## 全局约束

- FC 地域 cn-hangzhou，服务名 dmla，函数名 sandbox-cpu
- 镜像 tag fc，推送到已有 ACR 仓库 crpi-aani1ibpows293b8.cn-hangzhou.personal.cr.aliyuncs.com/fenixsoft/dmla-sandbox:fc
