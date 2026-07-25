<p align="center">
	<br/>
  <a href="https://ai.icyfenix.cn" target="_blank">
    <img src="https://github.com/fenixsoft/dmla/blob/main/docs/.vuepress/public/logo.png?raw=true" alt="logo">
  </a>
  <div align="center">
    <img src="https://img.shields.io/github/stars/fenixsoft/dmla" alt="GitHub Repo stars">
    <img src="https://img.shields.io/github/followers/fenixsoft" alt="GitHub followers">
    <img src="https://img.shields.io/github/forks/fenixsoft/dmla" alt="GitHub forks">
  </div>
  <div align="center">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=fenixsoft/dmla" alt="visitors">
    <img src="https://img.shields.io/github/last-commit/fenixsoft/dmla" alt="GitHub last commit">
    <img src="https://img.shields.io/github/actions/workflow/status/fenixsoft/dmla/deploy.yml" alt="GitHub Actions Workflow Status">
    <img src="https://img.shields.io/npm/v/%40icyfenix-dmla%2Fcli?color=aquamarine" alt="NPM Version">
    <img src="https://img.shields.io/badge/license-creative%20commons%20by%204.0-red" alt="License">
    <img src="https://img.shields.io/badge/author-icyfenix-chocolate" alt="Author">
</div>
</p>

## Introduction

This is a book on "how to apply machine learning to production systems," designed to serve as a navigational map for software developers to organize their knowledge framework in the age of artificial intelligence. In terms of purpose, it is similar to my other work, [The Fenix Project](https://icyfenix.cn/). However, this book has a higher barrier to entry. Before diving in, it is recommended to read the article [On Designing Machine Learning Application](https://ai.icyfenix.cn/introduction/about-dmla.html) to understand the book's scope and intended audience.

I wrote this book with the following goals in mind:

- I am a software engineer and a PhD candidate in machine learning. At a time when the software industry is shifting from "hardcoded logic" to "data-driven intelligence," I hope to use this book as a guide to systematically organize my machine learning knowledge, fill in the gaps, and integrate everything into my own knowledge framework.
- I believe there is a shortcut to growth for technical professionals: it is not only about seeing, reading, thinking, and applying — it is equally about speaking and writing. Articulating what you believe you have mastered, explaining it clearly and coherently, presenting it with confidence, helping others understand it and dispelling their doubts, and subjecting your views to scrutiny and even challenge — in this process, you will uncover much of the "unknown" hidden behind what you thought was "known." There is no knowing without doing; to know and not to do is simply not yet to know.

## Screenshot
<p align="center">
  <div align="center">
    <img src="https://github.com/fenixsoft/dmla/blob/main/docs/.vuepress/public/homepage.png?raw=true" >
    <a href="https://ai.icyfenix.cn" target="_blank">Designing Machine Learning Application Systems</a>
  </div>
</p>

## Quick Start

- **Read Online**: The online reading address is [https://ai.icyfenix.cn](https://ai.icyfenix.cn).
  The site is hosted on GitHub Pages; continuous integration powered by GitHub Actions compiles the Markdown documents from the Git repository and syncs them to the website in real time, pushing them to a CDN for accelerated access within China. Discussion services are provided by GitHub Discussions and [Giscus](https://github.com/giscus/giscus).

- **Code Sandbox**: This book contains a large number of code exercises that can be run directly in the browser, used to demonstrate machine learning algorithms and to perform model training. For Part I (Mathematical Foundations) and Part II (Classical Statistical Learning Algorithms), a pure CPU environment is sufficient. For deep learning and beyond, some model training experiments require GPU support (currently based on PyTorch with CUDA 12.8). For more details, see [Setting Up the Sandbox Environment](https://ai.icyfenix.cn/appendixes/sandbox.html).
  ```shell
  npx @icyfenix-dmla/install@latest
  ```

- **Offline Use**:
  - Deploy an offline site: The documentation is built with [VuePress](https://vuepress.vuejs.org/). To set up a documentation site within your organization, use the following commands:

  ``` shell
  # Clone the source code
  git clone https://github.com/fenixsoft/dmla.git && cd dmla

  # Install project dependencies
  npm install

  # Start the website; the docs are served at http://localhost:8080 by default,
  # and the sandbox at http://localhost:3001
  npm run local
  ```

  - When running locally, executing the code in the documentation still requires Docker image support. You can pull images directly using the one-click sandbox installer mentioned above, or build them locally with:
  ``` shell
  npm run build:sandbox:all
  ```

- **Redistribution, Adaptation, and Publication**: For all content in this book that references other sources, the original sources are explicitly credited within the text, and all rights belong to the original authors. All other content, including but not limited to text, images, and tables, is original work by the author. The author releases these original works under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/). As long as you comply with the license terms — attribution, non-commercial use, and share-alike — you are free to use, modify, adapt, and redistribute any part of this book anywhere, in any form, to anyone. See the "License" section below for details.

## License

- This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/). Under this license, you are free to share — copy and redistribute the material in any medium or format — as well as to adapt — remix, transform, and build upon the material. However, you must:
  - **Attribution**: When using all or part of this book, you must credit the original author and provide source information.
  - **Non-Commercial**: You may not use the material for commercial publication or any other commercial purposes. For commercial use, please contact the author.
  - **Share-Alike**: If you remix, transform, or build upon this work, you must distribute your contributions under the same Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
