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
        htmlLabels: true,
        // 大幅减小节点间距和层级间距，使图表更紧凑
        nodeSpacing: 20,
        rankSpacing: 25,
        // 使用贝塞尔曲线，连线更紧凑
        curve: 'basis'
      },
      themeVariables: {
        // 默认字体从 14px 减小到 11px，节点会更小
        fontSize: '11px',
        // 节点样式 - 与 SVG 图片一致
        primaryColor: '#eeeeee',
        primaryBorderColor: '#999999',
        primaryTextColor: '#333333',
        // 节点边框宽度从 2px 减小到 1px
        nodeBorderWidth: '1px',
        // 线条样式
        lineColor: '#333333',
        // 线条宽度变细
        edgeLineWidth: '1px',
        // 菱形节点样式
        secondaryColor: '#eeeeee',
        secondaryBorderColor: '#999999',
        secondaryTextColor: '#333333',
        // 减小节点内部 padding
        nodePadding: 8
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

    // 从 data-mermaid 属性获取原始代码（避免浏览器解析 HTML 标签）
    const code = el.dataset.mermaid ? decodeURIComponent(el.dataset.mermaid) : el.textContent

    // 提取尺寸修饰符类名（mermaid-small, mermaid-compact, mermaid-tiny）
    const sizeClasses = el.className.split(' ').filter(cls => cls.startsWith('mermaid-') && cls !== 'mermaid')

    if (code && code.trim()) {
      try {
        // 创建新的 div 容器，保留尺寸修饰符
        const div = document.createElement('div')
        div.className = 'mermaid ' + sizeClasses.join(' ')
        div.style.textAlign = 'center'
        div.textContent = code.trim()

        // 标记为已渲染
        div.dataset.rendered = 'true'

        // 替换原来的 pre 元素
        el.parentNode.replaceChild(div, el)

        // 渲染
        await mermaid.run({ nodes: [div] })

        // 渲染后确保居中，并根据缩放调整容器大小
        const svg = div.querySelector('svg')
        if (svg) {
          svg.style.display = 'inline-block'
          svg.style.margin = '0 auto'

          // 如果有尺寸修饰符，调整容器大小以避免裁剪
          const sizeClass = sizeClasses.find(cls => cls.startsWith('mermaid-'))
          if (sizeClass) {
            // 获取缩放比例
            const scales = {
              'mermaid-small': 0.85,
              'mermaid-compact': 0.75,
              'mermaid-tiny': 0.6
            }
            const scale = scales[sizeClass] || 1

            // 获取 SVG 的原始尺寸
            const viewBox = svg.getAttribute('viewBox')
            if (viewBox) {
              const parts = viewBox.split(' ')
              const width = parseFloat(parts[2])
              const height = parseFloat(parts[3])

              // 设置容器宽高为缩放后的尺寸
              div.style.width = `${width * scale}px`
              div.style.height = `${height * scale}px`
              div.style.overflow = 'hidden'
              div.style.display = 'inline-block'
              div.style.textAlign = 'left'

              // SVG 使用 transform scale
              svg.style.transform = `scale(${scale})`
              svg.style.transformOrigin = 'top left'
              // SVG 保持原始尺寸
              svg.style.width = `${width}px`
              svg.style.height = `${height}px`
            }
          }
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
            htmlLabels: true,
            // 大幅减小节点间距和层级间距，使图表更紧凑
            nodeSpacing: 20,
            rankSpacing: 25,
            // 使用贝塞尔曲线，连线更紧凑
            curve: 'basis'
          },
          themeVariables: {
            // 默认字体从 14px 减小到 11px，节点会更小
            fontSize: '11px',
            // 节点样式 - 与 SVG 图片一致
            primaryColor: '#eeeeee',
            primaryBorderColor: '#999999',
            primaryTextColor: '#333333',
            // 节点边框宽度从 2px 减小到 1px
            nodeBorderWidth: '1px',
            // 线条样式
            lineColor: '#333333',
            // 线条宽度变细
            edgeLineWidth: '1px',
            // 菱形节点样式
            secondaryColor: '#eeeeee',
            secondaryBorderColor: '#999999',
            secondaryTextColor: '#333333',
            // 减小节点内部 padding
            nodePadding: 8
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