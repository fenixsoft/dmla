import { viteBundler } from '@vuepress/bundler-vite'
import { getDirname, path } from 'vuepress/utils'
import mermaidPlugin from './plugins/mermaid/index.js'
import runnableCodePlugin from './plugins/runnable-code/index.js'
import mathPlugin from './plugins/math/index.js'
import emphasisFixPlugin from './plugins/emphasis-fix/index.js'
import wordCountPlugin from './plugins/word-count/index.js'
import commentsPlugin from './plugins/comments/index.js'
import ideaspacesTheme from './theme/index.js'

const __dirname = getDirname(import.meta.url)

export default {
  // 站点配置
  lang: 'zh-CN',
  title: 'IdeaSpaces',
  description: '交互式知识管理平台',

  // 部署配置 - GitHub Pages 子目录
  // base: '/ideaspaces/',

  // 使用自定义主题
  theme: ideaspacesTheme({
    // 导航栏
    navbar: [
      { text: '首页', link: '/' },
      { text: '机器学习数学基础', link: '/linear-algebra-vectors-matrices/' },
      { text: 'Python', link: '/python/' },
    ],

    // 侧边栏 - 使用简单的链接列表，让 VuePress 自动添加页面标题
    sidebar: {
      '/linear-algebra-vectors-matrices/': [
        {
          text: '向量与矩阵运算基础',
          collapsible: true,
          children: [
            { text: '概述', link: '/linear-algebra-vectors-matrices/' },
            { text: '1. 引言', link: '/linear-algebra-vectors-matrices/01-introduction' },
            { text: '2. 向量基础', link: '/linear-algebra-vectors-matrices/02-vectors' },
            { text: '3. 矩阵基础', link: '/linear-algebra-vectors-matrices/03-matrices' },
            { text: '4. NumPy 实践', link: '/linear-algebra-vectors-matrices/04-numpy' },
            { text: '5. 应用场景', link: '/linear-algebra-vectors-matrices/05-applications' },
            { text: '6. 总结与练习', link: '/linear-algebra-vectors-matrices/06-summary' },
            { text: '实验与练习', link: '/linear-algebra-vectors-matrices/experiments' }
          ]
        }
      ],
      '/python/': [
        {
          text: 'Python 教程',
          collapsible: true,
          children: [
            { text: 'Python 知识库', link: '/python/' },
            { text: '装饰器详解', link: '/python/decorators' }
          ]
        }
      ]
    }
  }),

  // 插件配置
  plugins: [
    // 修复中文括号后粗体标记问题
    emphasisFixPlugin,
    // Mermaid 流程图
    mermaidPlugin,
    // 可运行代码块
    runnableCodePlugin,
    // LaTeX 数学公式
    mathPlugin,
    // 字数统计
    wordCountPlugin,
    // GitHub Issues 评论系统
    commentsPlugin({
      repo: 'fenixsoft/ideaspaces',
      clientId: process.env.GITHUB_CLIENT_ID || ''
    })
  ],

  // 打包器配置
  bundler: viteBundler(),

  // 开发服务器配置
  devServer: {
    port: 8080,
    open: false
  }
}