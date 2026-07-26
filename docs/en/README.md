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
  - text: Start Reading
    link: /en/introduction/about-dmla.html
    type: primary
  - text: Table of Contents
    link: /en/contents.html
    type: secondary
  # - text: Build Sandbox
  #   link: /en/appendixes/sandbox.html
  #   type: secondary
# features:
#   - title: Developer & Intelligence
#     details: Written from the perspective of developers transitioning to AI, a systematic collection of machine learning articles
#   - title: Practice-Driven
#     details: Driven by case studies and code, understand AI principles through hands-on practice
#   - title: Code Sandbox
#     details: After deploying the sandbox Docker, Python code in the articles supports online editing and execution, with GPU acceleration
footer: CC-BY-NC-SA-4.0 Licensed | Copyright &copy; 2026
---

<HomeHero />

## About This Document

This document explores how to apply machine learning to production systems, offering software developers a roadmap for organizing their knowledge in the AI era. In spirit, it is similar to my other work, [The Fenix Projects](https://icyfenix.cn/). However, this document has a steeper learning curve than Phoenix Architecture. It is recommended to read [About Designing Machine Learning Applications](./introduction/about-dmla.md) first to understand the document's purpose and target audience.

I wrote this document with the following goals in mind:

- I am a software engineer and a PhD in machine learning. At a time when the software industry is shifting from "hard-coded logic" to "data-driven intelligence," I hope to use this document as a guide to systematically organize my knowledge of machine learning, fill in the gaps, and integrate it all into my own knowledge framework.
- I believe there is a shortcut to technical growth: not only reading, studying, thinking, and applying, but also articulating and writing. Expressing what you believe you have mastered -- stating it clearly and coherently, explaining it with confidence, making others understand and resolving their doubts, and subjecting your views to the scrutiny and even challenge of others. In this process, you uncover much of the "unknown" lurking behind the "known." There is no true knowledge without action; knowledge without action is not yet true knowledge.

## Quick Start

- **Read Online**: The online reading address for this document is [https://ai.icyfenix.cn/en](https://ai.icyfenix.cn/en). The website is hosted on GitHub Pages for storage, uses GitHub Actions for continuous integration to compile and sync Markdown documents from the Git repository to the site in real time, pushes to CDN for accelerated access in China, and uses GitHub Discussions with [Giscus](https://github.com/giscus/giscus) for discussion services.

- **Code Sandbox**: This document contains numerous interactive code examples that run directly in your browser for demonstrating machine learning algorithms and model training. For Part 1 (Mathematical Foundations) and Part 2 (Classic Statistical Learning Algorithms), a pure CPU environment is sufficient. For deep learning and beyond, some model training experiments require GPU support (currently based on PyTorch with CUDA 13.0). For more details, see [Building the Sandbox Environment](./appendixes/sandbox.md).

  ``` shell
  # Install and run the sandbox environment
  npx @icyfenix-dmla/install@latest
  ```

- **Run Offline**:
  - Deploy an offline site: The document is built with [VuePress](https://vuepress.vuejs.org/). If you want to host the documentation site within your organization, use the following commands:

  ``` shell
  # Clone the repository
  git clone https://github.com/fenixsoft/dmla.git && cd dmla

  # Install project dependencies
  npm install

  # Run the website; default address is http://localhost:8080, sandbox address defaults to http://localhost:3001
  npm run local
  ```

  - For local runtime, running the code examples in the document still requires a Docker image. You can pull the image directly using the one-click sandbox installation command, or build it locally with the following command:

  ``` shell
  npm run build:sandbox:all
  ```

- **Secondary Creation, Distribution, and Publishing**: All content in this document cites sources where applicable, with rights belonging to the original authors. All other content, including but not limited to text, images, tables, etc., is original work by the author. This original content is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/). As long as you follow the license terms of attribution, non-commercial use, and share-alike, you may use, modify, adapt, and distribute any part of this document anywhere, in any form, to anyone. See the "License" section of this document for details.

## License

- This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/). Under this license, you are free to share -- copy and redistribute the material in any medium or format -- and adapt -- remix, transform, and build upon the material, provided you meet the following conditions:
  - **Attribution**: You must give appropriate credit to the original author and source.
  - **Non-Commercial**: You may not use the material for commercial purposes. For commercial use, please contact the author.
  - **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## Filing

Website ICP filing: [Guangdong ICP 18088957](http://beian.miit.gov.cn)
