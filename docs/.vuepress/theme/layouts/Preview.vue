<template>
  <div class="preview-layout">
    <!-- 代码高亮加载器 - 确保主题 CSS 正确加载 -->
    <CodeHighlightLoader />
    <div class="theme-default-content">
      <Content />
    </div>
  </div>
</template>

<script setup>
import { Content } from 'vuepress/client'
import { onMounted } from 'vue'
import CodeHighlightLoader from '../components/CodeHighlightLoader.vue'

// 预览模式 - 隐藏所有导航元素
onMounted(() => {
  if (typeof document !== 'undefined') {
    // 添加预览模式标记类
    document.body.classList.add('preview-mode')
  }
})
</script>

<style>
/* 预览模式 - 隐藏导航栏和侧边栏 */
.preview-mode .vp-navbar,
.preview-mode .vp-sidebar,
.preview-mode .vp-page-nav,
.preview-mode .vp-page-meta,
.preview-mode .vp-footer,
.preview-mode .vp-back-to-top-button {
  display: none !important;
}

/* 预览模式 - 内容区域占满 */
.preview-mode .vp-page {
  padding: 0 !important;
}

.preview-mode .theme-default-content {
  max-width: 100% !important;
  padding: 16px 24px !important;
}

.preview-mode .vp-page-content-top,
.preview-mode .vp-page-content-bottom,
.preview-mode .vp-page-bottom {
  display: none !important;
}

</style>

<style scoped>
.preview-layout {
  width: 100%;
  height: 100%;
}
</style>