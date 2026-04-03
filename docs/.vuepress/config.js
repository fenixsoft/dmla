import { viteBundler } from '@vuepress/bundler-vite'
import { gitPlugin } from '@vuepress/plugin-git'
import { getDirname, path } from 'vuepress/utils'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import mermaidPlugin from './plugins/mermaid/index.js'
import runnableCodePlugin from './plugins/runnable-code/index.js'
import mathPlugin from './plugins/math/index.js'
import emphasisFixPlugin from './plugins/emphasis-fix/index.js'
import wordCountPlugin from './plugins/word-count/index.js'
import commentsPlugin from './plugins/comments/index.js'
import ideaspacesTheme from './theme/index.js'
import { searchProPlugin } from 'vuepress-plugin-search-pro'
import { searchVersionFixPlugin } from './plugins/search-version-fix/index.js'

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
      { text: '目录', link: '/SUMMARY' },
      { text: 'GitHub', link: 'https://github.com/fenixsoft/ideaspaces' },
    ],

    // 侧边栏 - 统一显示所有菜单，默认全部展开
    sidebar: [
      {
        text: '目录',
        collapsible: false,
        link: 'SUMMARY'
      },
      {
        text: '前言',
        collapsible: false,
        children: [
          { text: '关于作者', link: '/introduction/about-me' },
          { text: '关于本项目' },
        ]
      },
      {
        text: '机器学习的数学基础',
        collapsible: false,
        children: [
          {
            text: '线性代数',
            collapsible: false,
            children: [
              { text: '引言：机器学习的语言', link: '/linear/introduction' },
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
              { text: '引言：变化与累积', link: '/calculus/introduction' },
              { text: '极限、导数与微分', link: '/calculus/derivative' },
              { text: '多元函数与复合基础', link: '/calculus/gradient' },
              { text: '微积分计算实践', link: '/calculus/numpy' },
            ]
          }
        ]
      },  
    ]
  }),

  // 插件配置
  plugins: [
    // 自动注册 components 目录下的组件
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, './components'),
    }),
    // 搜索功能
    searchProPlugin({
      // 禁用索引热度排序
      hotReload: true,
    }),
    // 修复搜索版本字段兼容性问题
    searchVersionFixPlugin,
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

  // 打包器配置 - 抑制 Sass if-function 弃用警告
  bundler: viteBundler({
    viteOptions: {
      css: {
        preprocessorOptions: {
          scss: {
            silenceDeprecations: ['if-function']
          }
        }
      }
    }
  }),

  // 开发服务器配置
  devServer: {
    port: 8080,
    open: false
  }
}