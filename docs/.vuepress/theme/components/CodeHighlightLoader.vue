<template>
  <!-- 预览页面的样式辅助组件，CDN 主题 CSS 由 client.js 统一加载 -->
</template>

<style>
/* 自定义高亮主题样式（当使用 'default' 时） */
.custom-highlight-theme pre {
  background: var(--code-bg, #282C34);
}

.custom-highlight-theme pre code {
  color: var(--code-text, #ABB2BF);
}

/* 保持代码字体样式 */
div[class*="language-"] > pre > code {
  font-family: 'Fira Code', 'SF Mono', Monaco, Consolas, monospace;
}
</style>
