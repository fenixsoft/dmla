<template>
  <div class="comments-section">
    <!-- Giscus 评论容器 -->
    <div class="giscus-container" ref="giscusContainer"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const giscusContainer = ref(null)
let giscusScript = null

// Giscus 配置
const giscusConfig = {
  repo: 'fenixsoft/ideaspaces',
  repoId: 'R_kgDORvEFhQ',
  category: 'Comments',
  categoryId: 'DIC_kwDORvEFhc4C5_39',
  mapping: 'pathname',
  strict: '0',
  reactionsEnabled: '0', // 禁用表情反应
  emitMetadata: '0',
  inputPosition: 'top', // 输入框在顶部
  theme: 'preferred_color_scheme', // 跟随系统主题
  lang: 'zh-CN',
  loading: 'lazy'
}

// 加载 Giscus
const loadGiscus = () => {
  if (!giscusContainer.value) return

  // 清空容器
  giscusContainer.value.innerHTML = ''

  // 创建 script 元素
  giscusScript = document.createElement('script')
  giscusScript.src = 'https://giscus.app/client.js'
  giscusScript.setAttribute('data-repo', giscusConfig.repo)
  giscusScript.setAttribute('data-repo-id', giscusConfig.repoId)
  giscusScript.setAttribute('data-category', giscusConfig.category)
  giscusScript.setAttribute('data-category-id', giscusConfig.categoryId)
  giscusScript.setAttribute('data-mapping', giscusConfig.mapping)
  giscusScript.setAttribute('data-strict', giscusConfig.strict)
  giscusScript.setAttribute('data-reactions-enabled', giscusConfig.reactionsEnabled)
  giscusScript.setAttribute('data-emit-metadata', giscusConfig.emitMetadata)
  giscusScript.setAttribute('data-input-position', giscusConfig.inputPosition)
  giscusScript.setAttribute('data-theme', giscusConfig.theme)
  giscusScript.setAttribute('data-lang', giscusConfig.lang)
  giscusScript.setAttribute('data-loading', giscusConfig.loading)
  giscusScript.setAttribute('crossorigin', 'anonymous')
  giscusScript.async = true

  giscusContainer.value.appendChild(giscusScript)
}

// 更新 Giscus 主题（响应系统主题变化）
const updateTheme = (theme) => {
  const iframe = giscusContainer.value?.querySelector('iframe.giscus-frame')
  if (iframe) {
    iframe.contentWindow.postMessage(
      { giscus: { setConfig: { theme } } },
      'https://giscus.app'
    )
  }
}

onMounted(() => {
  loadGiscus()

  // 监听系统主题变化
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
  mediaQuery.addEventListener('change', (e) => {
    updateTheme(e.matches ? 'dark' : 'light')
  })
})

// 路由变化时更新评论
watch(() => route.path, (newPath) => {
  const iframe = giscusContainer.value?.querySelector('iframe.giscus-frame')
  if (iframe) {
    iframe.contentWindow.postMessage(
      { giscus: { setConfig: { term: newPath } } },
      'https://giscus.app'
    )
  } else {
    loadGiscus()
  }
})

onBeforeUnmount(() => {
  if (giscusScript && giscusScript.parentNode) {
    giscusScript.parentNode.removeChild(giscusScript)
  }
})
</script>

<style lang="scss" scoped>
.comments-section {
  max-width: var(--content-width);
  margin: 0 auto;
  padding-top: 2rem;
}

.giscus-container {
  min-height: 200px;
}

/* 响应式适配 - 与 VuePress 默认主题一致 */
@media (max-width: 959px) {
  .comments-section {
    padding: 32px 1rem 0;
  }
}

@media (max-width: 719px) {
  .comments-section {
    padding: 24px 1rem 0;
    margin-top: 24px;
  }
}

@media (max-width: 419px) {
  .comments-section {
    padding: 16px 0.5rem 0;
    margin-top: 16px;
  }
}
</style>