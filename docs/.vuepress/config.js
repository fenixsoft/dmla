import { viteBundler } from '@vuepress/bundler-vite'
import { getDirname, path } from 'vuepress/utils'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import { mediumZoomPlugin } from '@vuepress/plugin-medium-zoom'
import mermaidPlugin from './plugins/mermaid/index.js'
import runnableCodePlugin from './plugins/runnable-code/index.js'
import mathPlugin from './plugins/math/index.js'
import emphasisFixPlugin from './plugins/emphasis-fix/index.js'
import wordCountPlugin from './plugins/word-count/index.js'
import dmlaTheme from './theme/index.js'
import { searchProPlugin } from 'vuepress-plugin-search-pro'
import { searchVersionFixPlugin } from './plugins/search-version-fix/index.js'

const __dirname = getDirname(import.meta.url)

export default {
  // 站点配置
  lang: 'zh-CN',
  title: '设计机器学习应用系统',
  description: '',

  // 禁止浏览器翻译（网站本身就是中文）
  head: [
    ['meta', { name: 'google', content: 'notranslate' }],
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  // 部署配置 - GitHub Pages 子目录
  // base: '/dmla/',

  // 使用自定义主题
  theme: dmlaTheme({
    logo: '/logo_min_size.png',

    // 禁用颜色模式切换按钮
    colorModeSwitch: false,

    // 配置主题插件
    themePlugins: {
      // 禁用默认的 medium-zoom，使用自定义配置
      mediumZoom: false
    },

    // 导航栏
    navbar: [
      { text: '首页', link: '/' },
      { text: '讨论区', link: '/boards' },
    ],

    // 侧边栏 - 统一显示所有菜单，默认全部展开
    sidebar: [
      {
        text: '目录',
        collapsible: false,
        link: 'contents'
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
        text: '机器学习数学基础',
        collapsible: false,
        children: [
          {
            text: '线性代数',
            collapsible: false,
            children: [
              { text: '引言：机器学习的语言', link: '/maths/linear/introduction' },
              { text: '向量基础', link: '/maths/linear/vectors' },
              { text: '矩阵基础', link: '/maths/linear/matrices' },
              { text: '数据处理实践', link: '/maths/linear/numpy' },
              { text: '应用场景', link: '/maths/linear/applications' },
            ]
          },
          {
            text: '微积分',
            collapsible: false,
            children: [
              { text: '引言：变化与累积', link: '/maths/calculus/introduction' },
              { text: '极限、导数与微分', link: '/maths/calculus/derivative' },
              { text: '多元函数与复合函数求导', link: '/maths/calculus/gradient' },
              { text: '微积分计算实践', link: '/maths/calculus/numpy' },
            ]
          },
          {
            text: '统计与概率',
            collapsible: false,
            children: [
              { text: '引言：概率性思维', link: '/maths/probability/introduction' },
              { text: '概率基础', link: '/maths/probability/probability-basics' },
              { text: '统计推断', link: '/maths/probability/statistical-inference' },
              { text: '概率统计实践', link: '/maths/probability/numpy-practice' },
            ]
          }
        ]
      }, 
      {
        text: '经典统计学习方法',
        collapsible: false,
        children: [
          {
            text: '线性模型',
            collapsible: false,
            children: [
              { text: '线性回归', link: '/statistical-learning/linear-models/linear-regression' },
              { text: '逻辑回归', link: '/statistical-learning/linear-models/logistic-regression' },
              { text: '正则化与广义线性模型', link: '/statistical-learning/linear-models/regularization-glm' },
            ]
          },
          {
            text: '贝叶斯方法',
            collapsible: false,
            children: [
              { text: '朴素贝叶斯', link: '/statistical-learning/bayesian-methods/naive-bayes' },
              { text: '贝叶斯网络', link: '/statistical-learning/bayesian-methods/bayesian-network' },
              { text: 'EM 算法', link: '/statistical-learning/bayesian-methods/em-algorithm' },
            ]
          },
          {
            text: '支持向量机',
            collapsible: false,
            children: [
              { text: '支持向量机', link: '/statistical-learning/support-vector-machines/svm-max-margin' },
              { text: '核技巧', link: '/statistical-learning/support-vector-machines/kernel-methods' },
            ]
          },
          {
            text: '决策树与集成',
            collapsible: false,
            children: [
              { text: '决策树', link: '/statistical-learning/decision-tree-ensemble/decision-tree' },
              { text: '随机森林', link: '/statistical-learning/decision-tree-ensemble/random-forest' },
              { text: '提升方法', link: '/statistical-learning/decision-tree-ensemble/boosting' },
            ]
          },
          {
            text: '无监督学习',
            collapsible: false,
            children: [
              { text: '聚类', link: '/statistical-learning/unsupervised-learning/clustering' },
              { text: '降维', link: '/statistical-learning/unsupervised-learning/dimensionality-reduction' },
            ]
          },
        ]
      },
      {
        text: '深度学习（2006-2017 TF架构前）',
        collapsible: false,
      },
      {
        text: '大语言模型（2017- SOTA）',
        collapsible: false,
      },
      {
        text: 'AI Infra & 应用',
        collapsible: false,
      },
      {
        text: '经典论文阅读',
        collapsible: false,
      },
    ]
  }),

  // 插件配置
  plugins: [
    // 自动注册 components 目录下的组件，排除 HomeHero（已手动注册）
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, './components'),
      excludes: ['HomeHero.vue'],
    }),
    // 图片缩放，排除带有 data-no-zoom 属性的图片
    mediumZoomPlugin({
      selector: '[vp-content] > img:not([data-no-zoom]), [vp-content] :not(a) > img:not([data-no-zoom])'
    }),
    // 搜索功能
    searchProPlugin({
      // 开发时禁用搜索索引更新后的自动刷新，避免编辑 markdown 时页面 reload
      hotReload: false,
    }),
    // 修复搜索版本字段兼容性问题
    searchVersionFixPlugin,
    // Git 信息由 defaultTheme 内置提供，无需额外配置
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
    // 评论系统由 Comments.vue 组件直接集成 Giscus
  ],

  // 打包器配置 - 抑制 Sass if-function 弃用警告
  bundler: viteBundler({
    viteOptions: {
      resolve: {
        alias: {
          // 强制所有 VueUse 导入指向根目录，避免重复加载
          '@vueuse/core': path.resolve(__dirname, '../../node_modules/@vueuse/core/index.mjs'),
          '@vueuse/shared': path.resolve(__dirname, '../../node_modules/@vueuse/shared/index.mjs')
        }
      },
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