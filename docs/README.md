---
heroImage: /logo.png
# heroText: 
# tagline: 
prev: false
next: false
badges:
  - src: https://img.shields.io/github/stars/fenixsoft/dmla
    alt: GitHub Repo stars
    href: https://github.com/fenixsoft/dmla
  - src: https://img.shields.io/github/followers/fenixsoft
    alt: GitHub followers
    href: https://github.com/fenixsoft
  - src: https://img.shields.io/github/forks/fenixsoft/dmla
    alt: GitHub forks
    href: https://github.com/fenixsoft/dmla/forks
informations:
  - src: https://visitor-badge.laobi.icu/badge?page_id=fenixsoft/dmla
    alt: visitors
  - src: https://img.shields.io/github/last-commit/fenixsoft/dmla
    alt: GitHub last commit
  - src: https://img.shields.io/github/actions/workflow/status/fenixsoft/dmla/deploy.yml
    alt: GitHub Actions Workflow Status
  - src: https://img.shields.io/npm/v/%40icyfenix-dmla%2Fcli?color=aquamarine
    alt: NPM Version
  - src: https://img.shields.io/badge/license-creative%20commons%20by%204.0-red
    alt: License
  - src: https://img.shields.io/badge/author-icyfenix-chocolate
    alt: Author
actions:
  - text: 开始阅读
    link: /introduction/about-dmla.html
    type: primary
  - text: 文档目录
    link: /contents.html
    type: secondary
  # - text: 构建沙箱
  #   link: /sandbox.html
  #   type: secondary
# features:
#   - title: 🧑‍💻 开发者与智能
#     details: 以开发者向人工智能转型的视角撰写，成体系的机器学习文章集
#   - title: 🏅 实践驱动
#     details: 由案例和代码驱动，通过动手实践理解人工智能的原理
#   - title: 💾 代码沙箱
#     details: 部署沙箱 Docker 后，文章中的 Python 代码支持在线修改与执行，支持 GPU 加速
footer: CC-BY-NC-SA-4.0 Licensed | Copyright © 2026
---

<HomeHero />

## 文档简介

这是一本讨论"如何将机器学习应用于生产系统"的文档，旨在为软件开发者在人工智能时代提供梳理知识框架的导航地图。从目的角度看，它与我的另一部文档《[凤凰架构](https://icyfenix.cn/)》是一样的。不过，这部文档的阅读门槛相对凤凰架构更高，建议开始前先阅读文章《[关于设计机器学习应用系统](./introduction/about-dmla.md)》，了解文档的主旨与读者画像。

笔者出于以下目的，撰写这部文档：

- 笔者是一名软件工程师，也是机器学习方向的 PhD。身处软件业正由"硬编码的逻辑"转向"数据驱动的智能"的时点，笔者希望借文档为引，系统性地整理机器学习相关知识，查缺补漏，将它们都融入自己的知识框架。
- 笔者认为技术人员成长有一捷径，不仅要去看、去读、去想、去用，更要去说、去写。将自己认为掌握了的知识叙述出来，能够说得清晰有条理，讲得理直气壮；能够让他人听得明白，释去心中疑惑；能够把自己的观点交予别人的审视，乃至质疑，在此过程中，会挖掘出很多潜藏在"已知"背后的"未知"。未有知而不行者，知而不行，只是未知。

## 快速开始

- **在线阅读**：本文档在线阅读地址为 [https://ai.icyfenix.cn](https://ai.icyfenix.cn) 。
网站由 GitHub Pages 提供网站存储空间；由 GitHub Actions 提供的持续集成服务实时把 Git 仓库的 Markdown 文档编译同步至网站，并推送至 CDN 提供国内的访问加速；由 GitHub Discussions 与 [Giscus](https://github.com/giscus/giscus) 提供讨论服务。

- **代码沙箱**：本文档内包含大量可在网页中运行的代码实践，用于演示机器学习算法以及进行模型训练。对于第一部分数学基础和第二部分经典统计学习算法，只需纯 CPU 环境即可运行；对于深度学习及之后的内容，部分模型训练实验需要 GPU 支持（目前基于 PyTorch with CUDA 12.8）。更多详情请参见[构建沙箱环境](sandbox.md)。
  ```shell
  npx @icyfenix-dmla/install@latest
  ```

- **离线运行**：
  - 部署离线站点：文档基于 [VuePress](https://vuepress.vuejs.org/zh/) 构建，如你希望在企业内部搭建文档站点，请使用如下命令：

  ``` shell
  # 克隆获取源码
  git clone https://github.com/fenixsoft/dmla.git && cd dmla

  # 安装工程依赖
  npm install

  # 运行网站，文档地址默认为 http://localhost:8080，沙箱地址默认为 http://localhost:3001
  npm run local
  ```
  
  - 本地运行时，如需使用运行文档中的代码，仍然需要有 Docker 镜像的支持，镜像可使用前面沙箱一键安装直接拉取，也可以使用如下命令在本机构建：
  ``` shell
  npm run build:sandbox:all
  ```

- **二次演绎、传播和发行**：本文档中所有的内容，如引用其他资料，均在文档中明确列出资料来源，一切权利归属原作者。除此以外的所有内容，包括但不限于文字、图片、表格，等等，均属笔者原创，这些原创内容，笔者声明以[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）](http://creativecommons.org/licenses/by-nc-sa/4.0/) 发行，只要遵循许可协议条款中署名、非商业性使用、相同方式共享的条件，你可以在任何地方、以任何形式、向任何人使用、修改、演绎、传播本文档中任何部分的内容。详细可见本文档的"协议"一节。


## 协议

- 本作品采用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）](http://creativecommons.org/licenses/by-nc-sa/4.0/)许可。遵循许可的前提下，你可以自由地共享，包括在任何媒介上以任何形式复制、发行本作品，亦可以自由地演绎、修改、转换或以本作品为基础进行二次创作。但要求你：
  - **署名**：应在使用本文档的全部或部分内容的时候，注明原作者及来源信息。
  - **非商业性使用**：不得用于商业出版或其他任何带有商业性质的行为。如需商业性使用，请联系作者。
  - **相同方式共享的条件**：在本文档基础上演绎、修改的作品，应当继续以知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）许可。

## 备案

网站备案信息：[粤 ICP 备 18088957 号](http://beian.miit.gov.cn)