# Data Volume Management

数据卷管理能力，支持数据持久化存储和用户自定义挂载路径。

## 功能需求

### CLI 命令

`dmla data` 命令提供 TUI 菜单式交互，包含以下子功能：

1. **mount**: 设置挂载路径
   - 输入自定义路径（如 `~/dmla-data` 或 `D:\dmla-data`）
   - 验证路径存在性，不存在则提示创建
   - 保存配置到 `~/.dmla/config.json`

2. **path**: 显示实际路径
   - 输出当前挂载路径的绝对路径
   - 显示目录大小统计

3. **clear**: 清空数据内容
   - 确认提示，防止误操作
   - 清空 datasets、models、outputs、cache 目录内容
   - 保留目录结构

4. **remove**: 删除数据卷
   - 确认提示
   - 删除整个数据目录
   - 清除配置文件中的挂载路径设置

### 配置文件

`~/.dmla/config.json` 存储用户配置：

```json
{
  "dataVolumePath": "/home/user/dmla-data",
  "installedDatasets": ["tiny-imagenet-200", "cifar-10"],
  "lastModified": "2026-05-02T10:00:00Z"
}
```

### 目录结构

数据目录必须包含以下子目录：

- `datasets/`: 数据集存储
- `models/`: 模型文件存储
- `outputs/`: 输出文件存储
- `cache/`: 临时缓存

首次 mount 时自动创建完整目录结构。

## 技术约束

- 路径必须是绝对路径或可解析的相对路径（如 `~/dmla-data`）
- Windows 路径支持（如 `D:\dmla-data`）
- 路径中不允许包含空格或特殊字符
- 挂载路径修改需要重启沙箱服务才能生效

## 用户交互

- TUI 菜单不使用 emoji，确保 Windows CMD 兼容
- 确认提示使用 `[y/N]` 格式
- 进度显示使用 ASCII 字符