## 1. Docker 镜像更新

- [x] 1.1 更新 `Dockerfile.sandbox` 添加 ipykernel 和 jupyter_client 依赖
- [x] 1.2 更新 `Dockerfile.sandbox.cpu` 添加相同依赖
- [x] 1.3 构建并测试新镜像（GPU 和 CPU 版本）
  - CPU 版本已验证通过
  - GPU 版本需要在有 GPU 的环境中验证

## 2. 后端 Kernel 执行器实现

- [x] 2.1 创建 `local-server/src/kernel_runner.py` - IPython Kernel 执行器
  - 实现 Kernel 启动和关闭
  - 实现 execute_request 发送
  - 实现 IOPub 消息收集（stream, display_data, error, status）
  - 实现 JSON 格式输出
  - 实现超时处理
- [x] 2.2 编写 kernel_runner.py 单元测试
- [x] 2.3 修改 `local-server/src/sandbox.js` 调用 kernel_runner.py
  - 更新容器执行命令
  - 更新响应格式解析
  - 保持超时和错误处理逻辑

## 3. 前端输出渲染实现

- [x] 3.1 修改 `RunnableCode.vue` 解构化输出渲染
  - 解析 outputs 数组
  - 实现 stream 类型渲染（stdout/stderr）
  - 实现 display_data 类型渲染（image/png）
  - 实现 error 类型渲染（traceback 格式化）
- [x] 3.2 实现图片点击放大功能
  - 创建模态框组件
  - 实现 ESC 键和点击背景关闭
- [x] 3.3 更新输出区域样式
  - 图片样式（复用 Markdown 图片样式）
  - 错误输出样式
  - 模态框样式

## 4. 测试

- [x] 4.1 更新 `local-server/tests/sandbox.test.js` API 测试
  - 测试新的响应格式
  - 测试图片输出场景
  - 测试错误输出格式
- [x] 4.2 更新 Docker 执行测试
  - 测试 Kernel 执行流程
  - 测试 matplotlib 图片生成
  - 测试多图输出
- [x] 4.3 前端功能测试（使用 playwright-cli）
  - 测试文本输出渲染
  - 测试图片输出渲染
  - 测试图片点击放大
  - 测试错误输出显示

## 5. 文档更新

- [x] 5.1 更新 `docs/arch/design.md` Python 沙箱设计章节
  - 添加 IPython Kernel 架构说明
  - 添加 Jupyter 消息协议说明
  - 添加富输出支持说明
  - 更新 API 接口定义
  - 更新 Docker 镜像设计说明
- [x] 5.2 更新 README.md
  - 更新沙箱功能说明（支持图片输出）
  - 更新 Docker 镜像构建说明

## 6. 验证与集成

- [x] 6.1 本地验证 CPU 镜像构建和图片输出
- [ ] 6.2 本地验证 GPU 镜像构建和图片输出
- [x] 6.3 验证错误输出格式正确显示
- [x] 6.4 运行所有测试通过