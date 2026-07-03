# Task 5 报告: 更新 Settings.vue（FC/自定义模式选择）

## 状态: 完成

## 修改文件

- `docs/.vuepress/theme/components/Settings.vue`

## 修改内容

1. **Template** — 将沙箱服务配置 Tab 替换为双模式 UI：
   - 包含 FC（默认）和自定义地址两个 radio 选项
   - FC 模式下显示只读的 FC 沙箱地址输入框
   - 自定义模式下显示可编辑的服务地址输入框
   - 保持原有的连接状态显示区域

2. **Script**:
   - 新增 `FC_DEFAULT_URL` 常量（值来自 `fc-url.txt`: `https://sandbox-cpu-dcheerjqde.cn-hangzhou.fcapp.run`）
   - 新增 `sandboxMode` ref 变量（默认 `'fc'`）
   - 修改 `loadConfig()` 加载 sandboxMode 并处理 FC 模式下的 endpoint 赋值
   - 新增 `onModeChange()` 函数，切换 FC 模式时自动填充 URL
   - 修改 `save()` 函数，config 中增加 `sandboxMode` 字段，FC 模式下存储 FC URL

3. **Style**:
   - 新增 `.mode-selector`、`.mode-option`、`.mode-label`、`.mode-hint`、`.readonly-input` 样式

## 验证

- `npm run build` 构建成功，无 Vue 模板编译错误
- 提交哈希: `e61e59c`
