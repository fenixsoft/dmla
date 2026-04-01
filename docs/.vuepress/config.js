import { viteBundler } from '@vuepress/bundler-vite'
import { gitPlugin } from '@vuepress/plugin-git'
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
  description: '思维实验室',

  // 部署配置 - GitHub Pages 子目录
  // base: '/ideaspaces/',

  // 使用自定义主题
  theme: ideaspacesTheme({
    // 禁用颜色模式切换按钮
    colorModeSwitch: false,

    // 导航栏
    navbar: [
      { text: '首页', link: '/' },
      { text: '线性代数', link: '/linear/introduction' },
      { text: '微积分', link: '/calculus/' },
    ],

    // 侧边栏 - 使用简单的链接列表，让 VuePress 自动添加页面标题
    sidebar: {
      '/linear/': [
        {
          text: '向量与矩阵运算基础',
          collapsible: true,
          children: [
            { text: '引言', link: '/linear/introduction' },
            { text: '向量基础', link: '/linear/vectors' },
            { text: '矩阵基础', link: '/linear/matrices' },
            { text: '数据处理实践', link: '/linear/numpy' },
            { text: '应用场景', link: '/linear/applications' },
          ]
        }
      ],
      '/calculus/': [
        {
          text: '微积分基础',
          collapsible: true,
          children: [
            { text: '概述', link: '/calculus/' },
            { text: '引言：微积分是变化与累积的语言', link: '/calculus/01-introduction' },
            { text: '基础概念：极限、导数与微分', link: '/calculus/02-derivative' },
            { text: '进阶概念：多元函数与优化基础', link: '/calculus/03-gradient' },
            { text: 'NumPy实践：微积分计算', link: '/calculus/04-numpy-practice' },
            { text: '应用场景：微积分在机器学习中的实践', link: '/calculus/05-applications' },
          ]
        }
      ],
    }
  }),

  // 插件配置
  plugins: [
    // Git 信息（更新时间）
    gitPlugin(),
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