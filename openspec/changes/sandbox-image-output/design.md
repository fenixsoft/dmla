## Context

### 当前架构

```
┌─────────────────────────────────────────────────────────────────┐
│  前端 RunnableCode.vue                                          │
│  └── 点击 Run → POST /api/sandbox/run                          │
│      └── 接收 { success, output, error, executionTime }        │
│          └── 仅支持文本渲染                                      │
├─────────────────────────────────────────────────────────────────┤
│  后端 sandbox.js                                                │
│  └── Cmd: ['python3', '-c', code]                              │
│      └── 直接执行，返回 stdout/stderr                           │
├─────────────────────────────────────────────────────────────────┤
│  Docker 容器                                                    │
│  └── 已安装: matplotlib, numpy, pandas, torch...               │
│      └── 问题: 无显示器 → plt.show() 无法工作                   │
└─────────────────────────────────────────────────────────────────┘
```

### 约束条件

- **Kernel 生命周期**: 每次请求启动新 Kernel，与现有 Docker 容器模式一致
- **运行位置**: Kernel 在 Docker 容器内运行，保持环境隔离
- **超时控制**: 复用现有 60 秒超时机制
- **GPU 支持**: 复用现有 GPU 镜像选择逻辑

### 利益相关者

- 教学用户：需要在文章中展示数据可视化
- 开发者：需要维护和扩展沙箱功能

## Goals / Non-Goals

**Goals:**
- 支持 matplotlib 图片输出，用户直接调用 `plt.show()` 即可看到图片
- 支持多图显示，每次 `plt.show()` 都生成独立图片
- 支持图片点击放大查看
- 支持结构化错误输出，显示完整 Python traceback
- 为未来扩展富输出（DataFrame 表格、LaTeX 公式）打下基础

**Non-Goals:**
- 不实现 Kernel 池或复用机制
- 不实现交互式输入（input() 函数）
- 不实现跨请求变量持久化
- 不实现其他绘图库支持（如 plotly、bokeh）

## Decisions

### 决策 1: 采用 IPython Kernel 方案

**选择**: IPython Kernel + Jupyter 消息协议

**备选方案对比**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: matplotlib 后端注入 + Base64 标记 | 改动最小，用户零感知 | 仅支持 matplotlib，扩展性差 |
| B: 执行后提取图片文件 | 实现简单 | 需要用户改用 savefig 或仍需 monkey-patch |
| **C: IPython Kernel** | **Jupyter 标准，扩展性强** | **实现复杂度较高** |

**理由**:
- Jupyter 消息协议是成熟的富输出标准
- 天然支持 matplotlib 的 inline 后端
- 为未来支持 DataFrame、LaTeX 等打下基础
- 社区文档完善，问题容易排查

### 决策 2: Kernel 生命周期策略

**选择**: 每次请求启动新 Kernel

**理由**:
- 与现有 Docker 容器模式一致
- 完全隔离，无状态污染
- 实现简单，无需管理 Kernel 池

**备选**: Kernel 池 + 复用
- 响应更快，但需要管理池、状态重置
- 当前教学场景不需要跨请求共享变量

### 决策 3: Kernel 运行位置

**选择**: 在 Docker 容器内运行

**理由**:
- 复用现有 sandbox 镜像和环境隔离机制
- 支持 GPU 镜像选择
- 统一的安全模型

### 决策 4: 前端图片渲染策略

**选择**: 图片内嵌 + 点击放大模态框

**实现**:
- 图片作为 base64 内嵌在 `<img>` 标签
- 点击图片显示全屏模态框
- 复用现有 Markdown 图片渲染样式

### 决策 5: API 响应格式

**选择**: 结构化输出数组

```javascript
{
  success: true,
  executionTime: 1.234,
  outputs: [
    { type: 'stream', name: 'stdout', text: '...' },
    { type: 'display_data', data: { 'image/png': 'base64...' } },
    { type: 'error', ename: '...', evalue: '...', traceback: [...] }
  ]
}
```

**理由**:
- 与 Jupyter 消息协议输出格式对齐
- 支持多种输出类型混合
- 前端可按类型分别渲染

## Risks / Trade-offs

### 风险 1: Kernel 启动开销

**风险**: IPython Kernel 启动约需 1-2 秒，增加响应延迟

**缓解**:
- 接受这个延迟（教学场景可接受）
- 前端显示更明确的"执行中"状态
- 未来可考虑 Kernel 池优化

### 风险 2: 内存占用

**风险**: Kernel 进程占用额外内存

**缓解**:
- 执行完成后立即关闭 Kernel
- 复用现有 4GB 内存限制
- 监控容器内存使用

### 风险 3: 协议版本兼容性

**风险**: Jupyter 消息协议版本更新可能带来变化

**缓解**:
- 使用稳定的 protocol version 5.x
- 在 kernel_runner.py 中记录协议版本
- 测试覆盖关键消息类型

### 权衡: 复杂度 vs 扩展性

**选择**: 接受更高实现复杂度，换取更好的扩展性

**理由**:
- 方案 C 虽然实现复杂，但一次投入长期受益
- 未来添加新的输出类型只需前端渲染支持
- 符合 Jupyter 生态标准，便于理解和使用

## Migration Plan

### 部署步骤

1. **更新 Docker 镜像**
   ```bash
   npm run build:sandbox      # GPU 版本
   npm run build:sandbox:cpu  # CPU 版本
   ```

2. **更新后端代码**
   - 新增 `kernel_runner.py`
   - 修改 `sandbox.js` 调用方式

3. **更新前端组件**
   - 修改 `RunnableCode.vue` 输出渲染逻辑

4. **验证**
   - 运行测试
   - 手动验证图片输出

### 回滚策略

- 后端代码通过 Git 回滚
- Docker 镜像保留旧版本标签
- 前端重新部署上一版本

### 向后兼容

API 响应格式变更，但前端同步更新，无兼容性问题。如有外部调用方，可：
- 在过渡期保留旧字段 `output` 作为 `outputs` 中 stream 文本的合并
- 或直接声明 API 为内部接口

## Open Questions

1. ~~Kernel 生命周期策略？~~ → 已决定：每次请求新 Kernel
2. ~~Kernel 运行位置？~~ → 已决定：Docker 容器内
3. 是否需要支持 `input()` 交互？ → 暂不实现，`allow_stdin: False`
4. 是否需要支持 `display()` 函数返回对象？ → 暂不实现，等待需求明确