/**
 * nn-arch 客户端配置
 * 动态加载 nn-arch 库并渲染 SVG
 */
import { defineClientConfig, usePageData } from 'vuepress/client'
import { onMounted, watch, ref } from 'vue'
import { useRouter } from 'vue-router'

// nn-arch 模块缓存
let nnArchModule = null
let nnArchLoaded = false

/**
 * 渲染页面中的 nn-arch 图表
 */
async function renderNNArch() {
  // 尝试加载 nn-arch 模块
  if (!nnArchLoaded) {
    try {
      nnArchModule = await import('@icyfenix-dmla/nn-arch')
      nnArchLoaded = true
    } catch (err) {
      console.error('Failed to load nn-arch:', err)
      // 显示加载失败提示
      const containers = document.querySelectorAll('.nn-arch-container')
      for (const container of containers) {
        if (container.dataset.rendered !== 'true') {
          container.innerHTML = `<div class="nn-arch-error">无法加载神经网络可视化库，请刷新页面重试</div>`
          container.dataset.rendered = 'true'
        }
      }
      return
    }
  }

  const NNArch = nnArchModule.default || nnArchModule

  // 查找所有未渲染的 nn-arch 容器
  const containers = document.querySelectorAll('.nn-arch-container')

  for (const container of containers) {
    // 跳过已渲染的元素
    if (container.dataset.rendered === 'true') continue

    // 获取 YAML 内容
    const yaml = container.dataset.nnArch ? decodeURIComponent(container.dataset.nnArch) : ''

    // 获取尺寸参数
    const width = container.dataset.width ? parseInt(container.dataset.width, 10) : null
    const height = container.dataset.height ? parseInt(container.dataset.height, 10) : null

    if (!yaml.trim()) {
      container.innerHTML = `<div class="nn-arch-error">未提供神经网络架构定义</div>`
      container.dataset.rendered = 'true'
      continue
    }

    try {
      // 调用 nn-arch 生成 SVG
      const svg = NNArch.generateFromYaml(yaml)

      // 创建 SVG 容器
      const svgWrapper = document.createElement('div')
      svgWrapper.className = 'nn-arch-svg-wrapper'
      svgWrapper.innerHTML = svg

      // 获取 SVG 元素并应用尺寸
      const svgElement = svgWrapper.querySelector('svg')
      if (svgElement) {
        // 设置 SVG 样式
        svgElement.style.display = 'block'
        svgElement.style.margin = '0 auto'

        if (width || height) {
          // 获取原始尺寸
          const viewBox = svgElement.getAttribute('viewBox')
          if (viewBox) {
            const parts = viewBox.split(' ')
            const origWidth = parseFloat(parts[2])
            const origHeight = parseFloat(parts[3])
            const ratio = origHeight / origWidth

            if (width && height) {
              // 同时指定宽高
              svgElement.style.width = `${width}px`
              svgElement.style.height = `${height}px`
            } else if (width) {
              // 仅指定宽度，高度按比例计算
              svgElement.style.width = `${width}px`
              svgElement.style.height = `${width * ratio}px`
            } else if (height) {
              // 仅指定高度，宽度按比例计算
              svgElement.style.width = `${height / ratio}px`
              svgElement.style.height = `${height}px`
            }
          }
        } else {
          // 默认尺寸：100% 宽度，高度自适应
          svgElement.style.width = '100%'
          svgElement.style.maxWidth = '100%'
          svgElement.style.height = 'auto'
        }
      }

      // 替换 loading 内容
      container.innerHTML = ''
      container.appendChild(svgWrapper)
      container.dataset.rendered = 'true'

    } catch (err) {
      console.error('nn-arch render error:', err)
      container.innerHTML = `<div class="nn-arch-error">渲染失败: ${err.message || 'YAML 格式错误'}</div>`
      container.dataset.rendered = 'true'
    }
  }
}

export default defineClientConfig({
  enhance({ app }) {
    // 预加载 nn-arch 库
    if (typeof window !== 'undefined') {
      import('@icyfenix-dmla/nn-arch').then((module) => {
        nnArchModule = module
        nnArchLoaded = true
      }).catch(err => {
        console.error('Failed to preload nn-arch:', err)
      })
    }
  },

  setup() {
    const router = useRouter()
    const isInitialized = ref(false)
    const pageData = usePageData()

    // 组件挂载时渲染
    onMounted(() => {
      setTimeout(renderNNArch, 100)
      setTimeout(renderNNArch, 300)
    })

    // 监听路由变化
    watch(
      () => router.currentRoute.value.path,
      (newPath, oldPath) => {
        if (oldPath !== undefined) {
          setTimeout(renderNNArch, 100)
          setTimeout(renderNNArch, 300)
          setTimeout(renderNNArch, 500)
        }
      },
      { immediate: false }
    )

    // 监听页面数据变化（热更新）
    watch(
      () => pageData.value,
      () => {
        setTimeout(renderNNArch, 100)
        setTimeout(renderNNArch, 300)
      },
      { deep: true }
    )
  }
})