## Why

DMLA 项目是一个深度学习教学网站，在讲解神经网络架构时需要可视化的架构图。目前只能手动使用 nn-arch 工具生成 SVG 图片然后插入文档，流程繁琐且图片难以更新。

将 nn-arch 作为 VuePress 插件集成到文档系统中，可以让作者直接在 Markdown 中以代码块形式定义网络架构，渲染时自动生成 SVG 图片，实现"所见即所得"的可视化编写体验。

## What Changes

- 新增 VuePress 插件 `vuepress-plugin-nn-arch`，集成 nn-arch 神经网络可视化功能
- 支持 Markdown 中以 `nn-arch` 代码块语法插入 YAML 网络定义
- 支持在代码块语言标识符后指定图片尺寸（如 `nn-arch width=800 height=400`）
- 渲染时自动调用 nn-arch API 生成 SVG 并嵌入页面
- 支持 nn-arch 的所有功能：自定义网络、预置模板、复杂拓扑结构

## Capabilities

### New Capabilities

- `nn-arch-markdown`: Markdown 代码块语法解析能力，识别 `nn-arch` 代码块并提取 YAML 内容和尺寸参数
- `nn-arch-rendering`: 页面渲染时调用 nn-arch API 将 YAML 转换为 SVG 并嵌入文档

### Modified Capabilities

无（这是新功能，不修改现有能力）

## Impact

- 新增 npm 包依赖：`@icyfenix-dmla/nn-arch`
- 新增插件目录：`docs/.vuepress/plugins/vuepress-plugin-nn-arch/`
- 影响 Markdown 渲染流程，新增代码块处理逻辑
- 影响客户端脚本，新增 nn-arch 初始化和渲染逻辑