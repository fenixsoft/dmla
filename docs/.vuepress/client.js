import { defineClientConfig } from 'vuepress/client'
import { onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import './styles/custom.css'
import HomeHero from './components/HomeHero.vue'
import ChatDemo from './components/ChatDemo.vue'

// Sidebar 配置（从 config.js 同步）
const sidebarConfig = [
  {
    text: '前置数学基础',
    collapsible: false,
    children: [
      {
        text: '线性代数',
        collapsible: false,
        children: [
          { text: '引言', link: '/linear/introduction' },
          { text: '向量基础', link: '/linear/vectors' },
          { text: '矩阵基础', link: '/linear/matrices' },
          { text: '数据处理实践', link: '/linear/numpy' },
          { text: '应用场景', link: '/linear/applications' },
        ]
      },
      {
        text: '微积分',
        collapsible: false,
        children: [
          { text: '引言：变化与累积', link: '/calculus/01-introduction' },
          { text: '极限、导数与微分', link: '/calculus/02-derivative' },
          { text: '多元函数与优化基础', link: '/calculus/03-gradient' },
          { text: '微积分计算实践', link: '/calculus/04-numpy-practice' },
          { text: '应用场景', link: '/calculus/05-applications' },
        ]
      }
    ]
  }
]

export default defineClientConfig({
  setup() {
    if (typeof window === 'undefined') return // SSR 安全守卫，仅在客户端执行

    const router = useRouter()

    /**
     * 自动检测并标记真正的图注段落。
     * CSS 的 :only-child 伪类只计算元素节点，不计算文本节点，
     * 会导致正文中的斜体内联元素被误判为图注。
     * 此函数通过比较 <p> 与 <em> 的文本内容来精确判断：
     * 只有当 <p> 的全部文本都在 <em> 内时，才认定为图注。
     */
    function applyFigureCaptionStyles() {
      nextTick(() => {
        document.querySelectorAll('.theme-default-content p > em:only-child').forEach(em => {
          const p = em.parentElement
          // 真正的图注：<p> 的全部文本内容都在 <em> 内
          if (p.textContent.trim() === em.textContent.trim()) {
            em.classList.add('figure-caption')
          }
        })
      })
    }

    onMounted(applyFigureCaptionStyles)
    router.afterEach(applyFigureCaptionStyles)
  },
  enhance({ app }) {
    // 手动注册 HomeHero 组件，确保 VuePress 上下文正确传递
    // 避免 registerComponentsPlugin 自动注册导致的 HMR 上下文问题
    app.component('HomeHero', HomeHero)
    app.component('ChatDemo', ChatDemo)

    // 注入 sidebar 配置到全局属性
    app.provide('sidebarConfig', sidebarConfig)
  }
})