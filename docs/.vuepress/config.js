import { viteBundler } from '@vuepress/bundler-vite'
import { getDirname, path } from 'vuepress/utils'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import { mediumZoomPlugin } from '@vuepress/plugin-medium-zoom'
import mermaidPlugin from './plugins/mermaid/index.js'
import runnableCodePlugin from './plugins/runnable-code/index.js'
import mathPlugin from './plugins/math/index.js'
import emphasisFixPlugin from './plugins/emphasis-fix/index.js'
import wordCountPlugin from './plugins/word-count/index.js'
import nnArchPlugin from './plugins/nn-arch/index.js'
import dmlaTheme from './theme/index.js'
import { searchProPlugin } from 'vuepress-plugin-search-pro'
import { searchVersionFixPlugin } from './plugins/search-version-fix/index.js'

const __dirname = getDirname(import.meta.url)

export default {
  // 站点配置
  lang: 'zh-CN',
  title: '设计机器学习应用系统',
  description: '',

  // Markdown 渲染配置 - 启用 GFM breaks，让换行符渲染为 <br>
  // 解决 blockquote 中多行内容被合并到同一行的问题
  markdown: {
    breaks: true,
  },

  // 禁止浏览器翻译（网站本身就是中文）
  head: [
    ['meta', { name: 'google', content: 'notranslate' }],
    ['meta', { name: 'og:image', content: 'https://ai.icyfenix.cn/logo_min_size.png' }],
    ['meta', { name: 'og:description', content: '面向工程师从零建立机器学习的系统性理解' }],
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
          { text: '关于本文档', link: '/introduction/about-dmla' },
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
        text: '神经网络与深度学习',
        collapsible: false,
        children: [
          {
            text: '神经网络结构',
            collapsible: false,
            children: [
              { text: '神经网络基础原理', link: '/deep-learning/neural-network-structure/idea-origin' },
              { text: '线性感知机', link: '/deep-learning/neural-network-structure/perceptron' },
              { text: '多层感知机', link: '/deep-learning/neural-network-structure/mlp' },
              { text: '前向传播', link: '/deep-learning/neural-network-structure/forward-propagation' },
              { text: '反向传播', link: '/deep-learning/neural-network-structure/backpropagation' },
              { text: '激活函数与损失函数', link: '/deep-learning/neural-network-structure/activation-loss-functions' },
            ]
          },
          {
            text: '优化神经网络',
            collapsible: false,
            children: [
              { text: '梯度下降', link: '/deep-learning/neural-network-optimization/gradient-descent' },
              { text: '自适应优化器', link: '/deep-learning/neural-network-optimization/adaptive-optimizers' },
            ]
          },
          {
            text: '深层网络稳定性',
            collapsible: false,
            children: [
              { text: '权重初始化', link: '/deep-learning/neural-network-stability/weight-initialization' },
              { text: 'Dropout 正则化', link: '/deep-learning/neural-network-stability/dropout' },
              { text: '批归一化', link: '/deep-learning/neural-network-stability/batch-normalization' },
            ]
          },
          {
            text: '卷积神经网络',
            collapsible: false,
            children: [
              { text: 'CNN 基础原理', link: '/deep-learning/convolutional-neural-network/cnn-basics' },
              { text: 'AlexNet 与 CNN 复兴', link: '/deep-learning/convolutional-neural-network/alexnet' },
              { text: 'VGG 与 GoogLeNet', link: '/deep-learning/convolutional-neural-network/vgg-inception' },
              { text: 'ResNet 残差网络', link: '/deep-learning/convolutional-neural-network/resnet' },
              { text: '工程实训：AlexNet 图像分类实验', link: '/deep-learning/convolutional-neural-network/alexnet-experiment' },
            ]
          },
          {
            text: '生成式模型',
            collapsible: false,
            children: [
              { text: '变分自编码器', link: '/deep-learning/generative-models/vae' },
              { text: '生成式对抗网络', link: '/deep-learning/generative-models/gan' },
              { text: '工程实训：DCGAN 图像生成实验', link: '/deep-learning/generative-models/gan-experiment' },
            ]
          },
          {
            text: '序列模型',
            collapsible: false,
            children: [
              { text: '词嵌入与表示学习', link: '/deep-learning/sequence-models/word-embedding' },
              { text: 'RNN 基础原理', link: '/deep-learning/sequence-models/rnn-basics' },
              { text: 'LSTM 与 GRU 门控机制', link: '/deep-learning/sequence-models/lstm-gru' },
              { text: 'Seq2Seq 序列映射', link: '/deep-learning/sequence-models/seq2seq' },
              { text: '工程实训：LSTM 古诗词生成实验', link: '/deep-learning/sequence-models/lstm-experiment' },
            ]
          },
        ]
      },
      {
        text: '语言模型的奇点',
        collapsible: false,
        children: [
          {
            text: 'Transformer 架构',
            collapsible: false,
            children: [
              { text: 'Transformer 基础原理', link: '/language-models/architecture-basics/transformer-architecture' },
              { text: 'Transformer 演进与变体', link: '/language-models/architecture-basics/architecture-evolution' },
              { text: '语言模型与分词', link: '/language-models/architecture-basics/language-model-tokenization' },
              { text: '工程实训：Transformer 模型训练实验', link: '/language-models/architecture-basics/llm-pretrain-experiment' },
            ]
          },
          {
            text: '预训练与微调',
            collapsible: false,
            children: [
              { text: '预训练数据工程', link: '/language-models/pretraining/pretraining-data' },
              { text: '缩放定律', link: '/language-models/pretraining/scaling-laws' },
              { text: '分布式训练基础设施', link: '/language-models/pretraining/distributed-training' },
              { text: '监督微调', link: 'language-models/pretraining/supervised-finetuning' },
              { text: '工程实训：SFT 模型对话实验', link: '/language-models/pretraining/llm-sft-experiment' },
            ]
          },
          {
            text: '对齐训练',
            collapsible: false,
            children: [
              { text: '人类反馈强化学习', link: '/language-models/alignment/rlhf' },
              { text: '对齐方法的演进', link: '/language-models/alignment/alignment-new-paradigms' },
              { text: '工程实训：DPO 对齐训练实验', link: '/language-models/alignment/llm-dpo-experiment' },
            ]
            
          },
          {
            text: '推理能力',
            collapsible: false,
            children: [
              { text: '思维链与推理模型', link: '/language-models/reasoning/chain-of-thought' },
              { text: '推理缩放定律', link: '/language-models/reasoning/test-time-compute' },
              { text: '推理效率优化', link: '/language-models/reasoning/inference-efficiency' },
              // { text: '工程实训：LLM 推理效率优化实验', link: '/language-models/reasoning/llm-reasoning-experiment' },
            ]
          },
          {
            text: '模态融合与安全',
            collapsible: false,
            children: [
              // { text: '多模态大模型', link: '/language-models/frontier/multimodal-llm' },
              // { text: '模型评估与安全', link: '/language-models/frontier/evaluation-safety' },
            ]
          },
        ]
      },
      {
        text: 'AI 基础设施与工程化',
        collapsible: false,
          children: [
            // {
            //   text: '模型服务化',
            //   collapsible: false,
            //   children: [
            //     { text: '推理服务架构原理', link: '/ai-infra-engineering/model-serving/inference-service-architecture' },
            //     { text: '请求调度与批处理', link: '/ai-infra-engineering/model-serving/request-scheduling' },
            //     { text: 'GPU 资源管理', link: '/ai-infra-engineering/model-serving/gpu-resource-management' },
            //     { text: '工程实训：部署 LLM 推理服务', link: '/ai-infra-engineering/model-serving/llm-inference-experiment' },
            //   ]
            // },
            // {
            //   text: '工程化实践',
            //   collapsible: false,
            //   children: [
            //     { text: '数据版本管理', link: '/ai-infra-engineering/mlops/data-versioning' },
            //     { text: '特征存储', link: '/ai-infra-engineering/mlops/feature-store' },
            //     { text: '数据质量监控', link: '/ai-infra-engineering/mlops/data-quality-monitoring' },
            //     { text: '实验追踪', link: '/ai-infra-engineering/mlops/experiment-tracking' },
            //     { text: '模型注册与生命周期', link: '/ai-infra-engineering/mlops/model-registry-lifecycle' },
            //     { text: '自动化调参', link: '/ai-infra-engineering/mlops/hyperparameter-optimization' },
            //     { text: '模型性能监控', link: '/ai-infra-engineering/mlops/model-performance-monitoring' },
            //     { text: '漂移检测', link: '/ai-infra-engineering/mlops/drift-detection' },
            //     { text: '在线实验与渐进发布', link: '/ai-infra-engineering/mlops/online-experimentation' },
            //   ]
            // },
          ]
      },
      {
        text: '检索与 Agent 系统',
        collapsible: false,
          children: [
            // {
            //   text: '向量检索与 RAG',
            //   collapsible: false,
            //   children: [
            //     { text: '嵌入与向量检索', link: '/ai-infra-engineering/vector-retrieval-rag/embedding-and-indexing' },
            //     { text: '检索质量评估与优化', link: '/ai-infra-engineering/vector-retrieval-rag/retrieval-quality' },
            //     { text: '文档处理流水线', link: '/ai-infra-engineering/vector-retrieval-rag/document-processing-pipeline' },
            //     { text: '检索增强生成', link: '/ai-infra-engineering/vector-retrieval-rag/retrieval-augmented-generation' },
            //     { text: '工程实训：构建知识库问答系统', link: '/ai-infra-engineering/vector-retrieval-rag/rag-experiment' },
            //   ]
            // },
            // {
            //   text: 'Agent 系统',
            //   collapsible: false,
            //   children: [
            //     { text: '从 LLM 到 Agent', link: '/ai-infra-engineering/agent-systems/llm-to-agent' },
            //     { text: '工具调用', link: '/ai-infra-engineering/agent-systems/tool-use' },
            //     { text: '规划与推理', link: '/ai-infra-engineering/agent-systems/planning-reasoning' },
            //     { text: '记忆系统', link: '/ai-infra-engineering/agent-systems/memory-systems' },
            //     { text: '协作模式', link: '/ai-infra-engineering/agent-systems/collaboration-patterns' },
            //     { text: '通信协议', link: '/ai-infra-engineering/agent-systems/communication-protocols' },
            //     { text: '编排与容错', link: '/ai-infra-engineering/agent-systems/orchestration-fault-tolerance' },
            //     { text: '工程实训：构建自主 Agent', link: '/ai-infra-engineering/agent-systems/agent-experiment' },
            //     { text: '工程实训：构建多智能体协作系统', link: '/ai-infra-engineering/agent-systems/multi-agent-experiment' },
            //   ]
            // },
          ]
      },
      {
        text: '附录',
        collapsible: false,
        children: [
              { text: '构建沙箱环境', link: '/sandbox' },
              // { text: '临时格式测试页面', link: '/test' },
        ]
      },
    ]
  }),

  // 插件配置
  plugins: [
    // 自动注册 components 目录下的组件，排除 HomeHero（已手动注册）
    registerComponentsPlugin({
      componentsDir: path.resolve(__dirname, './components'),
      excludes: ['HomeHero.vue', 'ChatDemo.vue'],
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
    // 神经网络架构可视化
    nnArchPlugin,
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