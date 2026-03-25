/**
 * Mermaid 客户端配置
 * 动态加载 Mermaid 库并初始化
 */
import { defineClientConfig } from 'vuepress/client'

export default defineClientConfig({
  enhance({ app }) {
    // 动态导入 mermaid
    if (typeof window !== 'undefined') {
      import('mermaid').then((mermaid) => {
        mermaid.default.initialize({
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true
          }
        })
      })
    }
  },

  setup() {
    // 页面加载后重新渲染 mermaid 图表
    if (typeof window !== 'undefined') {
      import('mermaid').then((mermaid) => {
        setTimeout(() => {
          mermaid.default.run()
        }, 100)
      })
    }
  }
})