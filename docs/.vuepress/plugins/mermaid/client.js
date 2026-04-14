/**
 * Mermaid 客户端配置
 * 动态加载 Mermaid 库并初始化
 */
import { defineClientConfig, usePageData } from 'vuepress/client'
import { onMounted, watch, ref } from 'vue'
import { useRouter } from 'vue-router'

// 全局 mermaid 实例
let mermaidInstance = null
let mermaidLoaded = false

/**
 * 渲染页面中的 mermaid 图表
 */
async function renderMermaid() {
  const mermaidModule = await import('mermaid').catch(() => null)
  if (!mermaidModule) return

  const mermaid = mermaidModule.default
  mermaidInstance = mermaid

  // 初始化 mermaid（只执行一次）
  if (!mermaidLoaded) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true
      },
      themeVariables: {
        fontSize: '14px',
        // 节点样式 - 与 SVG 图片一致
        primaryColor: '#eeeeee',
        primaryBorderColor: '#999999',
        primaryTextColor: '#333333',
        // 节点边框宽度
        nodeBorderWidth: '2px',
        // 线条样式
        lineColor: '#333333',
        // 菱形节点样式
        secondaryColor: '#eeeeee',
        secondaryBorderColor: '#999999',
        secondaryTextColor: '#333333'
      }
    })
    mermaidLoaded = true
    window.mermaid = mermaid
  }

  // 查找所有未渲染的 mermaid pre 元素
  const mermaidEls = document.querySelectorAll('pre.mermaid')

  for (const el of mermaidEls) {
    // 跳过已经渲染过的元素
    if (el.dataset.rendered === 'true') continue

    // 获取 code 元素中的内容
    const codeEl = el.querySelector('code')
    const code = codeEl ? codeEl.textContent : el.textContent

    if (code && code.trim()) {
      try {
        // 创建新的 div 容器
        const div = document.createElement('div')
        div.className = 'mermaid'
        div.style.textAlign = 'center'
        div.textContent = code.trim()

        // 标记为已渲染
        div.dataset.rendered = 'true'

        // 替换原来的 pre 元素
        el.parentNode.replaceChild(div, el)

        // 渲染
        await mermaid.run({ nodes: [div] })

        // 渲染后确保居中
        const svg = div.querySelector('svg')
        if (svg) {
          svg.style.display = 'inline-block'
          svg.style.margin = '0 auto'
        }
      } catch (err) {
        console.error('Mermaid render error:', err)
      }
    }
  }
}

export default defineClientConfig({
  enhance({ app }) {
    // 预加载 mermaid 库
    if (typeof window !== 'undefined') {
      import('mermaid').then((mermaidModule) => {
        const mermaid = mermaidModule.default
        window.mermaid = mermaid

        mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true
          },
          themeVariables: {
            fontSize: '14px',
            // 节点样式 - 与 SVG 图片一致
            primaryColor: '#eeeeee',
            primaryBorderColor: '#999999',
            primaryTextColor: '#333333',
            // 节点边框宽度
            nodeBorderWidth: '2px',
            // 线条样式
            lineColor: '#333333',
            // 菱形节点样式
            secondaryColor: '#eeeeee',
            secondaryBorderColor: '#999999',
            secondaryTextColor: '#333333'
          }
        })
        mermaidLoaded = true
        mermaidInstance = mermaid
      }).catch(err => {
        console.error('Failed to load mermaid:', err)
      })
    }
  },

  setup() {
    // 使用 Vue Router 监听路由变化
    const router = useRouter()
    const isInitialized = ref(false)
    const pageData = usePageData()

    // 组件挂载时渲染
    onMounted(() => {
      // 延迟渲染，确保 DOM 已加载
      setTimeout(renderMermaid, 100)
      setTimeout(renderMermaid, 300)
    })

    // 监听路由变化，每次路由切换后重新渲染
    watch(
      () => router.currentRoute.value.path,
      (newPath, oldPath) => {
        // 路由变化后延迟渲染
        if (oldPath !== undefined) {
          setTimeout(renderMermaid, 100)
          setTimeout(renderMermaid, 300)
          setTimeout(renderMermaid, 500)
        }
      },
      { immediate: false }
    )

    // 监听页面数据变化（热更新会触发此变化），重新渲染 mermaid 图表
    watch(
      () => pageData.value,
      () => {
        setTimeout(renderMermaid, 100)
        setTimeout(renderMermaid, 300)
      },
      { deep: true }
    )
  }
})