/**
 * Math 插件客户端配置
 * 加载 KaTeX CSS 样式和自定义公式编号样式
 * 使用 VuePress 客户端生命周期处理公式编号
 */
import { onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import 'katex/dist/katex.min.css'

// 将数字转换为圆圈数字
const toCircleNumber = (num) => {
  // Unicode 圆圈数字范围
  // ①-⑨ (1-9): U+2460-U+2468
  // ⑩-⑳ (10-20): U+2469-U+2473
  // ㉑-㉟ (21-35): U+2474-U+2488
  // ㊱-㊽ (36-50): U+2489-U+249D
  if (num >= 1 && num <= 50) {
    return String.fromCodePoint(0x245F + num)
  }
  // 超过50时使用 CSS 样式生成圆圈
  return num
}

// 公式编号样式内容（在 onMounted 中插入）
const equationStylesContent = `
  /* 编号公式容器 */
  .equation-numbered {
    display: flex !important;
    align-items: center;
    justify-content: center;
    position: relative;
    margin: 1em 0;
    width: 100%;
    /* 锚点跳转时预留顶部空间 */
    scroll-margin-top: 0px;
  }

  /* 公式内容区域 */
  .equation-numbered .equation-content {
    flex: 1;
    text-align: center;
  }

  /* 公式内容区域内的 KaTeX */
  .equation-numbered .equation-content .katex-display {
    margin: 0 !important;
  }

  /* 公式编号 - 圆圈数字样式 */
  .equation-numbered .equation-number {
    margin-left: 2em;
    color: var(--c-text-lighter);
    font-size: 1em;
    white-space: nowrap;
  }

  /* 超过50的数字使用CSS圆圈 */
  .equation-number-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5em;
    height: 1.5em;
    border: 1px solid var(--c-text-lighter);
    border-radius: 50%;
    font-size: 0.7em;
  }

  /* 公式引用链接 - 圆圈数字样式 */
  .equation-reference {
    color: var(--c-brand);
    text-decoration: none;
    cursor: pointer;
    font-size: inherit;
    vertical-align: baseline;
    position: relative;
  }

  .equation-reference:hover {
    text-decoration: none;
    opacity: 0.8;
  }

  /* 公式预览提示框 */
  .equation-preview {
    position: fixed;
    z-index: 1000;
    max-width: 600px;
    min-width: 200px;
    padding: 12px 16px;
    background: var(--vp-c-bg, #fff);
    border: 1px solid var(--vp-c-border, #e4e4e7);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 14px;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s, visibility 0.2s;
    overflow-x: auto;
  }

  .equation-preview.visible {
    opacity: 1;
    visibility: visible;
  }

  .equation-preview .katex-display {
    margin: 0 !important;
  }

  .equation-preview-header {
    font-size: 12px;
    color: var(--c-text-mute, #71717a);
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--vp-c-border, #e4e4e7);
  }

  /* 未知的公式引用 */
  .equation-reference-unknown {
    color: var(--c-danger);
    font-style: italic;
    font-size: inherit;
    vertical-align: baseline;
  }
`

// 样式元素引用（在 onMounted 中创建）
let equationStyles = null

// 创建公式预览提示框
let previewElement = null

const createPreviewElement = () => {
  if (previewElement) return previewElement

  previewElement = document.createElement('div')
  previewElement.className = 'equation-preview'
  previewElement.innerHTML = '<div class="equation-preview-header">公式预览</div><div class="equation-preview-content"></div>'
  document.body.appendChild(previewElement)
  return previewElement
}

// 显示公式预览
const showEquationPreview = (refElement, label) => {
  const preview = createPreviewElement()
  const equationElement = document.getElementById(`eq-${label}`)

  if (!equationElement) return

  // 获取公式内容
  const contentDiv = equationElement.querySelector('.equation-content')
  if (!contentDiv) return

  // 设置预览内容
  const previewContent = preview.querySelector('.equation-preview-content')
  previewContent.innerHTML = contentDiv.innerHTML

  // 更新标题显示公式编号（圆圈数字）
  const number = equationElement.dataset.equationNumber
  const circleNum = parseInt(number) <= 50 ? toCircleNumber(parseInt(number)) : number
  preview.querySelector('.equation-preview-header').textContent = `公式 ${circleNum}`

  // 计算位置
  const rect = refElement.getBoundingClientRect()
  const previewRect = preview.getBoundingClientRect()

  // 默认显示在引用上方
  let top = rect.top - previewRect.height - 10
  let left = rect.left + rect.width / 2 - previewRect.width / 2

  // 如果上方空间不足，显示在下方
  if (top < 10) {
    top = rect.bottom + 10
  }

  // 确保不超出屏幕左右边界
  if (left < 10) {
    left = 10
  } else if (left + previewRect.width > window.innerWidth - 10) {
    left = window.innerWidth - previewRect.width - 10
  }

  // 使用 fixed 定位，相对于视口
  preview.style.top = `${top}px`
  preview.style.left = `${left}px`

  preview.classList.add('visible')
}

// 隐藏公式预览
const hideEquationPreview = () => {
  if (previewElement) {
    previewElement.classList.remove('visible')
  }
}

// 将服务端渲染的括号数字转换为圆圈数字
const convertNumbersToCircles = () => {
  // 处理公式编号
  const numberDivs = document.querySelectorAll('.equation-number')
  for (const div of numberDivs) {
    const text = div.textContent
    const match = text.match(/^\((\d+)\)$/)
    if (match) {
      const num = parseInt(match[1])
      if (num <= 50) {
        div.textContent = toCircleNumber(num)
      } else {
        // 超过50使用CSS圆圈
        const circleSpan = document.createElement('span')
        circleSpan.className = 'equation-number-circle'
        circleSpan.textContent = num
        div.textContent = ''
        div.appendChild(circleSpan)
      }
    }
  }

  // 处理公式引用链接
  const refLinks = document.querySelectorAll('.equation-reference')
  for (const link of refLinks) {
    const text = link.textContent
    const match = text.match(/^\((\d+)\)$/)
    if (match) {
      const num = parseInt(match[1])
      if (num <= 50) {
        link.textContent = toCircleNumber(num)
      } else {
        const circleSpan = document.createElement('span')
        circleSpan.className = 'equation-number-circle'
        circleSpan.textContent = num
        link.textContent = ''
        link.appendChild(circleSpan)
      }
      // 添加鼠标悬停预览功能
      const label = link.getAttribute('href')?.match(/#eq-(.+)$/)?.[1]
      if (label) {
        link.dataset.label = label.trim()
        link.addEventListener('mouseenter', () => {
          showEquationPreview(link, label.trim())
        })
        link.addEventListener('mouseleave', () => {
          hideEquationPreview()
        })
        link.addEventListener('click', (e) => {
          e.preventDefault()
          hideEquationPreview()
          const target = document.getElementById(`eq-${label.trim()}`)
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        })
      }
    }
  }
}

// 处理公式编号（使用 TreeWalker 查找注释节点）
const processEquationNumbers = () => {
  // 首先转换服务端渲染的括号数字为圆圈数字
  convertNumbersToCircles()

  let equationCounter = 0
  const equationLabels = new Map()

  // 首先从已有的 .equation-numbered 元素中注册标签（服务端可能已创建）
  const existingNumbered = document.querySelectorAll('.equation-numbered')
  for (const el of existingNumbered) {
    const label = el.dataset.label
    const number = parseInt(el.dataset.equationNumber)
    if (label && number) {
      equationLabels.set(label, number)
      // 确保 counter 不重复
      if (number > equationCounter) {
        equationCounter = number
      }
    }
  }

  // 第一遍：处理 equation:label 标记（处理未被服务端处理的公式）
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_COMMENT,
    null,
    false
  )

  const commentsToRemove = []
  const replacements = []

  while (walker.nextNode()) {
    const comment = walker.currentNode
    const value = comment.nodeValue.trim()

    // 处理 equation:label 标记
    if (value.startsWith('equation:label=')) {
      const label = value.replace('equation:label=', '').trim()
      // 如果标签已注册，跳过（服务端已处理）
      if (equationLabels.has(label)) {
        commentsToRemove.push(comment)
        continue
      }
      equationCounter++
      equationLabels.set(label, equationCounter)

      // 查找下一个兄弟元素（应该是公式）
      const nextElement = comment.nextSibling
      if (nextElement && nextElement.classList && nextElement.classList.contains('katex-display')) {
        replacements.push({
          comment,
          target: nextElement,
          label,
          number: equationCounter
        })
      }
      commentsToRemove.push(comment)
    }

    // 处理 end-equation 标记
    if (value === 'end-equation') {
      commentsToRemove.push(comment)
    }
  }

  // 执行替换
  for (const { comment, target, label, number } of replacements) {
    // 包装公式
    const wrapper = document.createElement('div')
    wrapper.className = 'katex-display equation-numbered'
    wrapper.id = `eq-${label}`
    wrapper.dataset.equationNumber = number
    wrapper.dataset.label = label

    const contentDiv = document.createElement('div')
    contentDiv.className = 'equation-content'

    // 如果公式内部有嵌套的 katex-display，只取内部内容
    const innerKatex = target.querySelector('.katex-display')
    if (innerKatex) {
      contentDiv.innerHTML = innerKatex.innerHTML
    } else {
      contentDiv.innerHTML = target.innerHTML
    }

    const numberDiv = document.createElement('div')
    numberDiv.className = 'equation-number'
    // 使用圆圈数字
    if (number <= 50) {
      numberDiv.textContent = toCircleNumber(number)
    } else {
      // 超过50使用CSS圆圈
      const circleSpan = document.createElement('span')
      circleSpan.className = 'equation-number-circle'
      circleSpan.textContent = number
      numberDiv.appendChild(circleSpan)
    }

    wrapper.appendChild(contentDiv)
    wrapper.appendChild(numberDiv)

    // 替换原公式
    target.replaceWith(wrapper)
  }

  // 移除注释节点
  for (const comment of commentsToRemove) {
    comment.remove()
  }

  // 第二遍：处理 eqref 引用（先收集再批量处理，避免 TreeWalker 遍历中断）
  const walker2 = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_COMMENT,
    null,
    false
  )

  const eqrefComments = []
  while (walker2.nextNode()) {
    const comment = walker2.currentNode
    const value = comment.nodeValue.trim()
    if (value.startsWith('eqref:')) {
      eqrefComments.push({ comment, value })
    }
  }

  // 批量处理 eqref 注释
  for (const { comment, value } of eqrefComments) {
    const label = value.replace('eqref:', '').trim()
    const equationNumber = equationLabels.get(label)

    // 创建引用链接
    const refElement = document.createElement('a')
    if (equationNumber) {
      refElement.className = 'equation-reference'
      refElement.href = `#eq-${label}`
      // 使用圆圈数字
      if (equationNumber <= 50) {
        refElement.textContent = toCircleNumber(equationNumber)
      } else {
        const circleSpan = document.createElement('span')
        circleSpan.className = 'equation-number-circle'
        circleSpan.textContent = equationNumber
        refElement.appendChild(circleSpan)
      }
      refElement.dataset.label = label

      // 鼠标悬停显示公式预览
      refElement.addEventListener('mouseenter', () => {
        showEquationPreview(refElement, label)
      })

      refElement.addEventListener('mouseleave', () => {
        hideEquationPreview()
      })

      // 点击时手动滚动到公式位置
      refElement.addEventListener('click', (e) => {
        e.preventDefault()
        hideEquationPreview()
        const target = document.getElementById(`eq-${label}`)
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }
      })
    } else {
      refElement.className = 'equation-reference-unknown'
      refElement.textContent = `${label}?`
    }

    // 替换注释
    comment.replaceWith(refElement)
  }

  // 第三遍：修复服务端创建的 unknown 引用（现在标签已注册）
  const unknownRefs = document.querySelectorAll('.equation-reference-unknown')
  for (const unknownEl of unknownRefs) {
    // 从内容中提取标签，格式为 "label?"
    const content = unknownEl.textContent
    const labelMatch = content.match(/^([^?]+)\?$/)
    if (labelMatch) {
      const label = labelMatch[1]
      const equationNumber = equationLabels.get(label)

      if (equationNumber) {
        // 创建正确的引用链接
        const refElement = document.createElement('a')
        refElement.className = 'equation-reference'
        refElement.href = `#eq-${label}`
        // 使用圆圈数字
        if (equationNumber <= 50) {
          refElement.textContent = toCircleNumber(equationNumber)
        } else {
          const circleSpan = document.createElement('span')
          circleSpan.className = 'equation-number-circle'
          circleSpan.textContent = equationNumber
          refElement.appendChild(circleSpan)
        }
        refElement.dataset.label = label

        // 鼠标悬停显示公式预览
        refElement.addEventListener('mouseenter', () => {
          showEquationPreview(refElement, label)
        })

        refElement.addEventListener('mouseleave', () => {
          hideEquationPreview()
        })

        // 点击时手动滚动到公式位置
        refElement.addEventListener('click', (e) => {
          e.preventDefault()
          hideEquationPreview()
          const target = document.getElementById(`eq-${label}`)
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        })

        // 替换 unknown 元素
        unknownEl.replaceWith(refElement)
      }
    }
  }
}

// MutationObserver 监听内容变化（处理 VuePress HMR 热更新）
let mutationObserver = null
let debounceTimer = null

// 设置 MutationObserver 监听内容变化
const setupMutationObserver = () => {
  if (mutationObserver) return // 避免重复创建

  mutationObserver = new MutationObserver((mutations) => {
    // 检查是否有内容变化
    const hasContentChange = mutations.some(mutation => {
      // 检查是否是内容区域的变化
      const target = mutation.target
      // VuePress 内容容器通常是 .theme-default-content
      if (target.classList && target.classList.contains('theme-default-content')) {
        return true
      }
      // 检查父元素是否是内容区域
      if (target.parentElement && target.parentElement.classList) {
        return target.parentElement.classList.contains('theme-default-content')
      }
      return false
    })

    if (hasContentChange) {
      // 使用防抖避免频繁处理
      clearTimeout(debounceTimer)
      debounceTimer = setTimeout(processEquationNumbers, 150)
    }
  })

  // 监听整个文档的变化
  mutationObserver.observe(document.body, {
    childList: true,
    subtree: true
  })
}

// 插入公式编号样式（仅在浏览器环境中执行）
const insertEquationStyles = () => {
  if (equationStyles) return // 避免重复插入
  equationStyles = document.createElement('style')
  equationStyles.textContent = equationStylesContent
  document.head.appendChild(equationStyles)
}

export default {
  setup() {
    const route = useRoute()

    // 页面加载时处理
    onMounted(() => {
      // 插入样式（仅在浏览器环境中）
      insertEquationStyles()
      // 等待 DOM 完全渲染
      setTimeout(processEquationNumbers, 100)
      // 设置 MutationObserver 监听后续变化
      setupMutationObserver()
    })

    // 路由变化时重新处理
    watch(
      () => route.path,
      () => {
        setTimeout(processEquationNumbers, 200)
      }
    )
  }
}