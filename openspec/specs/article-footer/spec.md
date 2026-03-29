## ADDED Requirements

### Requirement: 文章底部显示更新时间

系统 SHALL 在文章底部左侧显示文章的最后更新时间。

时间数据来源于 VuePress 的 `@vuepress/plugin-git`，通过 `page.value.git.updatedTime` 获取。

显示格式为 "更新于 YYYY-MM-DD"，使用 lucide calendar 图标。

#### Scenario: 文章有 Git 更新记录
- **WHEN** 文章页面有 Git 提交历史记录
- **THEN** 底部左侧显示 "更新于 YYYY-MM-DD" 格式的更新时间

#### Scenario: 文章无 Git 更新记录
- **WHEN** 文章页面无 Git 提交历史（如新创建的草稿）
- **THEN** 不显示更新时间信息

---

### Requirement: 文章底部显示字数统计

系统 SHALL 在文章底部左侧显示文章的总字数。

字数数据来源于构建时计算，通过 `page.data.wordCount` 获取。

显示格式为 "文章字数：N"（N 为千分位格式），使用 lucide file-text 图标。

#### Scenario: 文章有内容
- **WHEN** 文章页面有 Markdown 内容
- **THEN** 底部左侧显示 "文章字数：N" 格式的字数统计

#### Scenario: 文章无内容
- **WHEN** 文章页面为空或仅包含 frontmatter
- **THEN** 显示 "文章字数：0"

---

### Requirement: 文章底部显示 GitHub Star 按钮

系统 SHALL 在文章底部右侧显示 GitHub Star 按钮。

按钮包含两部分：
1. Star 按钮：深色背景 (#24292E)，白色文字，点击跳转到 GitHub Star 页面
2. Star 数徽章：显示当前仓库的实时 Star 数量

仓库地址为 `https://github.com/fenixsoft/ideaspaces`。

#### Scenario: GitHub buttons 加载成功
- **WHEN** github-buttons npm 包正常加载
- **THEN** 显示完整的 Star 按钮 + Star 数徽章

#### Scenario: GitHub buttons 加载失败
- **WHEN** 网络问题导致 github-buttons 无法加载
- **THEN** 显示静态 Star 链接，点击仍可跳转到 GitHub

---

### Requirement: 文章底部布局符合设计稿

系统 SHALL 按以下布局规范渲染文章底部：

- 顶部 1.5px 分割线，颜色 #E4E4E7
- 左侧区域：justify-content: center，layout: vertical，gap: 4px
  - 字数行：file-text icon (14x14) + "文章字数：N"，颜色 #71717A，字号 13px
  - 时间行：calendar icon (14x14) + "更新于 YYYY-MM-DD"，颜色 #A1A1AA，字号 12px
- 右侧区域：GitHub Star 按钮 + Star 数徽章，gap: 12px
- 整体：justify-content: space-between，padding-top: 32px，height: 69px

#### Scenario: 标准文章页面
- **WHEN** 用户访问任意文章页面
- **THEN** 底部显示符合上述布局规范的元信息区域