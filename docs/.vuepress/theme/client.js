import { defineClientConfig } from 'vuepress/client'
import { setupHeaders, setupSidebarItems } from '@theme/useSidebarItems'
import { setupDarkMode } from '@theme/useDarkMode'
import Layout from './layouts/Layout.vue'
import Preview from './layouts/Preview.vue'
import { getSiteConfig } from './utils/configMigration.js'
import { getThemeConfig } from './config/highlightThemes.js'
import { useRoute, useRouter } from 'vuepress/client'
import { watch, nextTick, onMounted } from 'vue'
import './styles/index.scss'

// 各主题对应的背景色
const THEME_BG_COLORS = {
  'default': '#282C34',
  'prism': '#f5f2f0',
  'prism-coy': '#fdfdfd',
  'prism-dark': '#1d1f21',
  'prism-funky': '#000',
  'prism-okaidia': '#272822',
  'prism-solarizedlight': '#fdf6e3',
  'prism-tomorrow': '#1d1f21',
  'prism-twilight': '#141414'
}

const THEME_LINK_ID = 'prism-theme-css'

/**
 * 动态加载 PrismJS 主题 CSS
 */
function loadThemeCSS(themeId) {
  if (typeof document === 'undefined') return

  // 移除之前加载的主题
  const existingLink = document.getElementById(THEME_LINK_ID)
  if (existingLink) {
    existingLink.remove()
  }

  const themeConfig = getThemeConfig(themeId)

  // 设置主题背景色 CSS 变量
  const bgColor = THEME_BG_COLORS[themeId] || '#282C34'
  document.documentElement.style.setProperty('--code-theme-bg', bgColor)

  // 如果主题不需要加载 CSS（如默认主题），添加标记类
  if (!themeConfig.cssPath) {
    document.body.classList.add('custom-highlight-theme')
    return
  }

  // 移除自定义样式标记
  document.body.classList.remove('custom-highlight-theme')

  // 动态创建 link 元素加载 CSS
  const link = document.createElement('link')
  link.id = THEME_LINK_ID
  link.rel = 'stylesheet'
  link.type = 'text/css'

  // 使用 CDN 加载 PrismJS 主题
  const themeName = themeConfig.cssPath.replace('prismjs/themes/', '').replace('.min.css', '')
  link.href = `https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/${themeName}.min.css`

  document.head.appendChild(link)

  console.log(`[HighlightLoader] 加载主题: ${themeConfig.name}, 背景色: ${bgColor}`)
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

/**
 * 将当前高亮的 sidebar 项滚动到可视区域顶部
 */
function scrollSidebarToActive() {
  const sidebar = document.querySelector('.vp-sidebar')
  if (!sidebar) return

  // 获取当前 URL 的 hash
  const hash = window.location.hash

  // 查找与当前 hash 匹配的 sidebar 项，或最深的 active 项
  let targetItem = null

  if (hash) {
    // 如果有 hash，查找链接匹配该 hash 的项
    const hashLink = `${window.location.pathname}${hash}`
    targetItem = sidebar.querySelector(`.vp-sidebar-item.active a[href="${hashLink}"]`)
    if (targetItem) {
      targetItem = targetItem.closest('.vp-sidebar-item')
    }
  }

  // 如果没找到匹配 hash 的项，使用最后一个（最深的）active 项
  if (!targetItem) {
    const activeItems = sidebar.querySelectorAll('.vp-sidebar-item.active')
    if (activeItems.length > 0) {
      // 使用最后一个 active 项（最深的嵌套项）
      targetItem = activeItems[activeItems.length - 1]
    }
  }

  if (!targetItem) return

  // 计算高亮项相对于 sidebar 容器的位置
  const sidebarRect = sidebar.getBoundingClientRect()
  const targetRect = targetItem.getBoundingClientRect()

  // 如果高亮项不在可视区域内（或不在顶部区域），则滚动
  const isAboveView = targetRect.top < sidebarRect.top
  const isBelowView = targetRect.bottom > sidebarRect.bottom
  const isNotNearTop = targetRect.top > sidebarRect.top + 50

  if (isAboveView || isBelowView || isNotNearTop) {
    // 计算滚动位置：让高亮项位于 sidebar 顶部附近
    const offsetTop = targetItem.offsetTop - sidebar.offsetTop - 10

    sidebar.scrollTo({
      top: offsetTop,
      behavior: 'smooth'
    })
  }
}

/**
 * 监听路由变化，自动滚动 sidebar 到高亮项
 */
function setupSidebarScroll() {
  const route = useRoute()
  const router = useRouter()

  // 防抖处理，避免频繁滚动
  let scrollTimeout = null

  const handleScroll = () => {
    if (scrollTimeout) {
      clearTimeout(scrollTimeout)
    }
    scrollTimeout = setTimeout(() => {
      nextTick(() => {
        scrollSidebarToActive()
      })
    }, 100)
  }

  // 监听路由 hash 变化（文章滚动时 hash 会跟随改变）
  watch(
    () => route.hash,
    handleScroll,
    { immediate: false }
  )

  // 监听路由路径变化（页面切换）
  watch(
    () => route.path,
    handleScroll,
    { immediate: false }
  )

  // 页面首次加载时，滚动到当前高亮项
  onMounted(() => {
    nextTick(() => {
      // 等待 sidebar 渲染完成后滚动
      setTimeout(scrollSidebarToActive, 200)
    })
  })
}

export default defineClientConfig({
  layouts: {
    Layout,
    Preview,
  },
  setup() {
    // 初始化默认主题的功能
    setupDarkMode()
    setupHeaders()
    setupSidebarItems()

    // 初始化 sidebar 自动滚动
    setupSidebarScroll()

    // 禁止浏览器翻译（网站本身就是中文）
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('translate', 'no')

      // 初始化代码高亮主题
      const config = getSiteConfig()
      loadThemeCSS(config.highlightTheme || 'default')

      // 监听配置变更事件
      window.addEventListener('site-config-changed', handleConfigChange)
    }
  }
})