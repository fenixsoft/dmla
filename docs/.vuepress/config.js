import { viteBundler } from '@vuepress/bundler-vite'
import { defaultTheme } from '@vuepress/theme-default'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import { getDirname, path } from 'vuepress/utils'
import mermaidPlugin from './plugins/mermaid/index.js'

const __dirname = getDirname(import.meta.url)

export default {
  // 站点配置
  lang: 'zh-CN',
  title: 'IdeaSpaces',
  description: '交互式知识管理平台',

  // 主题配置
  theme: defaultTheme({
    logo: '/logo.png',
    repo: 'https://github.com/username/ideaspaces',
    repoLabel: 'GitHub',
    editLink: false,
    lastUpdated: false,
    contributors: false,

    // 导航栏
    navbar: [
      { text: '首页', link: '/' },
      { text: 'Python', link: '/python/' },
      { text: '设计文档', link: '/arch/design' }
    ],

    // 侧边栏
    sidebar: {
      '/python/': [
        {
          text: 'Python 教程',
          collapsible: true,
          children: [
            '/python/',
            '/python/decorators'
          ]
        }
      ],
      '/arch/': [
        {
          text: '设计文档',
          children: [
            '/arch/design'
          ]
        }
      ]
    }
  }),

  // 插件配置
  plugins: [
    // 自动注册组件
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, './components')
    }),

    // Mermaid 流程图
    mermaidPlugin
  ],

  // 打包器配置
  bundler: viteBundler(),

  // 开发服务器配置
  devServer: {
    port: 8080,
    open: false
  }
}