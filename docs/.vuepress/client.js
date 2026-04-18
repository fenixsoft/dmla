import { defineClientConfig } from 'vuepress/client'
import './styles/custom.css'
import HomeHero from './components/HomeHero.vue'

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
  enhance({ app }) {
    // 手动注册 HomeHero 组件，确保 VuePress 上下文正确传递
    // 避免 registerComponentsPlugin 自动注册导致的 HMR 上下文问题
    app.component('HomeHero', HomeHero)

    // 注入 sidebar 配置到全局属性
    app.provide('sidebarConfig', sidebarConfig)
  }
})