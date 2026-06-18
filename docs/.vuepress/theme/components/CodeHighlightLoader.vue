<template>
  <!-- 这是一个无 UI 的客户端组件，负责动态加载高亮主题 CSS -->
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import { getSiteConfig } from '../utils/configMigration.js'
import { getThemeConfig, HIGHLIGHT_THEMES } from '../config/highlightThemes.js'

// 当前加载的主题 CSS link 元素
let currentThemeLink = null
const THEME_LINK_ID = 'prism-theme-css'

// 各主题对应的背景色
const THEME_BG_COLORS = {
  'default': '#282C34',      // One Dark Pro 风格
  'prism': '#f5f2f0',        // Default
  'prism-coy': '#fdfdfd',    // Coy
  'prism-dark': '#1d1f21',   // Dark
  'prism-funky': '#000',     // Funky
  'prism-okaidia': '#272822', // Okaidia
  'prism-solarizedlight': '#fdf6e3', // Solarized Light
  'prism-tomorrow': '#1d1f21', // Tomorrow Night
  'prism-twilight': '#141414'  // Twilight
}

/**
 * 动态加载 PrismJS 主题 CSS
 * @param {string} themeId - 主题 ID
 */
function loadThemeCSS(themeId) {
  // 移除之前加载的主题
  removeThemeCSS()

  const themeConfig = getThemeConfig(themeId)

  // 如果主题不需要加载 CSS（如默认主题），直接返回
  if (!themeConfig.cssPath) {
    // 添加自定义样式标记类到 body
    document.body.classList.add('custom-highlight-theme')
    // 设置 CSS 变量
    document.documentElement.style.setProperty('--code-theme-bg', THEME_BG_COLORS['default'])
    return
  }

  // 移除自定义样式标记
  document.body.classList.remove('custom-highlight-theme')

  // 设置主题背景色 CSS 变量
  const bgColor = THEME_BG_COLORS[themeId] || '#282C34'
  document.documentElement.style.setProperty('--code-theme-bg', bgColor)

  // 动态创建 link 元素加载 CSS
  const link = document.createElement('link')
  link.id = THEME_LINK_ID
  link.rel = 'stylesheet'
  link.type = 'text/css'

  // 使用 CDN 加载 PrismJS 主题
  const themeName = themeConfig.cssPath.replace('prismjs/themes/', '').replace('.min.css', '')
  link.href = `https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/${themeName}.min.css`

  document.head.appendChild(link)
  currentThemeLink = link

  console.log(`[HighlightLoader] 加载主题: ${themeConfig.name}, 背景色: ${bgColor}`)
}

/**
 * 移除已加载的主题 CSS
 */
function removeThemeCSS() {
  const existingLink = document.getElementById(THEME_LINK_ID)
  if (existingLink) {
    existingLink.remove()
    currentThemeLink = null
  }
}

/**
 * 处理配置变更事件
 */
function handleConfigChange(event) {
  const config = event.detail
  if (config && config.highlightTheme) {
    loadThemeCSS(config.highlightTheme)
  }
}

// 初始化时加载配置的主题
onMounted(() => {
  const config = getSiteConfig()
  loadThemeCSS(config.highlightTheme || 'default')

  // 监听配置变更事件
  window.addEventListener('site-config-changed', handleConfigChange)
})

// 清理事件监听
onUnmounted(() => {
  window.removeEventListener('site-config-changed', handleConfigChange)
})
</script>

<style>
/* 自定义高亮主题样式（当使用 'default' 时） */
.custom-highlight-theme pre {
  background: var(--code-bg, #282C34);
}

.custom-highlight-theme pre code {
  color: var(--code-text, #ABB2BF);
}

/* VuePress 代码块结构兼容：div.language-* > pre > code
   使用 CSS 变量传递主题背景色，应用到 div.language-* */
div[class*="language-"] {
  background: var(--code-theme-bg, #282C34);
  border-radius: 12px;
  max-width: 100%;
  overflow-x: auto;
  overflow: hidden;
}

/* 让 pre 继承 div 的背景 */
div[class*="language-"] > pre {
  background: transparent;
  margin: 0;
max-width: 100%;
  overflow-x: auto;
}

/* 保持代码字体样式 */
div[class*="language-"] > pre > code {
  font-family: 'Fira Code', 'SF Mono', Monaco, Consolas, monospace;
}
</style>