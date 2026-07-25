# 翻译进度追踪

> 最后更新: 2026-07-25

| 相对路径 | 状态 | 翻译代理 | 校审代理 | 备注 |
|---------|------|---------|---------|------|
| introduction/about-dmla.md | 校审完成 | ✓ | ✓ | 已修复链接 |
| introduction/about-me.md | 校审完成 | ✓ | ✓ | 已修复图片路径 |
| maths/linear/vectors.md | 校审完成 | ✓ | ✓ | 已修复链接路径（6 处 `../../../en/` → `../../`）|
| maths/linear/matrices.md | 校审完成 | ✓ | ✓ | 无需修复 |
| maths/calculus/derivative.md | 校审完成 | ✓ | ✓ | 已修复图片路径、英文 `neither zero nor yet zero` 术语误译 |
| maths/calculus/gradient.md | 校审完成 | ✓ | ✓ | 无需修复 |
| maths/probability/probability-basics.md | 校审完成 | ✓ | ✓ | 已修复残留中文「思维方式」未译 |
| maths/probability/statistical-inference.md | 校审完成 | ✓ | ✓ | 修复锚点链接 `#maximum-likelihood-estimation-mle`→`#maximum-likelihood-estimation`；两个英文用词优化 |
| statistical-learning/linear-models/linear-regression.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/linear-models/logistic-regression.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/linear-models/regularization-glm.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/bayesian-methods/naive-bayes.md | 校审完成 | ✓ | ✓ | 修复图片路径引用（引用源目录 assets）；发现 `bayesian-network.md` 断链（英文目录下缺失该文件） |
| statistical-learning/bayesian-methods/bayesian-network.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/bayesian-methods/em-algorithm.md | 校审完成 | ✓ | ✓ | 发现 `vae.md` 断链（英文目录下缺失该文件）|
| statistical-learning/support-vector-machines/kernel-methods.md | 校审完成 | ✓ | ✓ | 修复 typo: `of-dimensional`→`of dimensional`；修复可读性: `encountered as`→`used as`；deep-learning 词袋模型链接目标文件尚未翻译（断链） |
| statistical-learning/support-vector-machines/svm-max-margin.md | 校审完成 | ✓ | ✓ | 修复 CN 练习题编号（全部为1→1/2/3/4）、图片路径 `./assets/`→`assets/` 统一、长句断句 |
| statistical-learning/decision-tree-ensemble/decision-tree.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/decision-tree-ensemble/random-forest.md | 校审完成 | ✓ | ✓ | 共享模块 `random_forest_classifier.py` 需从本文重新提取获得英文注释 |
| statistical-learning/decision-tree-ensemble/boosting.md | 校审完成 | ✓ | ✓ | 无需修复 |
| statistical-learning/unsupervised-learning/clustering.md | 校审完成 | ✓ | ✓ | 注释译英、图片路径、链接锚点均已正确修复；修复 matplotlib features 列表缺失的 `•` 项目符号（2 处）；练习题为英译"1."→"1."和"2."（原文 ZH 两题均为"1."）；全文术语、代码、LaTeX、Mermaid、路径、完整性、可读性均校验通过 |
| statistical-learning/unsupervised-learning/dimensionality-reduction.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、路径、完整性、可读性均校验通过，无需修复；深层学习链接（word-embedding、vae）系已知断链 |
| deep-learning/neural-network-structure/idea-origin.md | 校审完成 | ✓ | ✓ | 修复残留中文「低谷」未译；术语、路径、完整性、可读性均校验通过 |
| deep-learning/neural-network-structure/perceptron.md | 校审完成 | ✓ | ✓ | 修复练习题编号（`1.`→`2.`）；术语、路径、代码、LaTeX、完整性、可读性均校验通过 |
| deep-learning/neural-network-structure/mlp.md | 校审完成 | ✓ | ✓ | 修复残留中文「脉络」未译；术语、路径、代码、LaTeX、完整性、可读性均校验通过 |
| deep-learning/neural-network-structure/forward-propagation.md | 校审完成 | ✓ | ✓ | 修复 Mermaid 残留中文标签（"输入 x"→"Input x" 等 4 图共 8 处）；术语、路径、代码、LaTeX、完整性、可读性均校验通过 |
| deep-learning/neural-network-structure/backpropagation.md | 校审完成 | ✓ | ✓ | 术语、路径、代码、LaTeX、Mermaid、完整性、可读性均校验通过，无需修复 |
| deep-learning/neural-network-structure/activation-loss-functions.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、图片路径、完整性、可读性均校验通过，无需修复；`weight-initialization.md`、`word-embedding.md` 系已知断链 |
| deep-learning/neural-network-stability/weight-initialization.md | 校审完成 | ✓ | ✓ | 修复 ReLU 锚点 `#relu-function`→`#relu-and-its-variants`；术语、代码、LaTeX、锚点链接、完整性、可读性均校验通过；发现原文已有断链 2 处（详见报告）|
| deep-learning/neural-network-stability/batch-normalization.md | 校审完成 | ✓ | ✓ | 修复代码中 `BN's` 未转义导致语法错误（单引号字符串内含撇号）；术语、路径、LaTeX、完整性、可读性均校验通过 |
| deep-learning/neural-network-stability/dropout.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、锚点链接、完整性、可读性均校验通过，无需修复；`batch-normalization.md` 断链已由本次翻译修复 |
| deep-learning/neural-network-optimization/gradient-descent.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、图片路径、完整性、可读性均校验通过，无需修复。修复锚点链接 `#learning-rate-decay`→`#learning-rate-selection-strategy`（该锚点在中文原文中亦不指向任何标题）|
| deep-learning/neural-network-optimization/adaptive-optimizers.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、链接、完整性、可读性均校验通过，无需修复 |
| deep-learning/convolutional-neural-network/cnn-basics.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、Mermaid、图片路径、完整性、可读性均校验通过，无需修复 |
| README.md | 校审完成 | ✓ | ✓ | 修复 `PhD candidate`→`PhD` 事实错误；优化多处英译流畅性：`navigational map for structuring their knowledge framework in the era of AI`→`roadmap for organizing their knowledge in the AI era`、`higher reading threshold`→`steeper learning curve`、`runnable code examples that can be executed in the web page`→`interactive code examples that run directly in your browser` |
