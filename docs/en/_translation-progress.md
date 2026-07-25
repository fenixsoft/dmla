# 翻译进度追踪

> 最后更新: 2026-07-25（llm-to-agent.md 翻译完成）

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
| deep-learning/convolutional-neural-network/alexnet.md | 校审完成 | ✓ | ✓ | 修复内部链接路径错误 2 处（`../../../deep-learning/...`→`../neural-network-structure/...`、`../../neural-network-stability/...`→`../neural-network-stability/...`）；术语、代码、LaTeX、nn-arch 图、Mermaid、完整性、可读性均校验通过 |
| deep-learning/convolutional-neural-network/vgg-inception.md | 校审完成 | ✓ | ✓ | LaTeX/Markdown 不变；nn-arch 图表中文标签译英；锚点 `#cnn-架构设计原则`→`#cnn-architecture-design-principles`；内部 `.md` 链接保留原相对路径不变；`\text{}` 中文译英；校审：术语、nn-arch 图、代码、LaTeX、完整性、可读性均校验通过，无需修复 |
| deep-learning/convolutional-neural-network/resnet.md | 校审完成 | ✓ | ✓ | 修复英译可读性：`substantially brought about by`→`largely due to`；链接锚点 `#cnn-architecture-design-principles` 已确认存在于英文版 `cnn-basics.md`；术语、LaTeX、nn-arch 图、完整性、可读性均校验通过 |
| README.md | 校审完成 | ✓ | ✓ | 修复 `PhD candidate`→`PhD` 事实错误；优化多处英译流畅性：`navigational map for structuring their knowledge framework in the era of AI`→`roadmap for organizing their knowledge in the AI era`、`higher reading threshold`→`steeper learning curve`、`runnable code examples that can be executed in the web page`→`interactive code examples that run directly in your browser` |
| deep-learning/convolutional-neural-network/alexnet-experiment.md | 校审完成 | ✓ | ✓ | 修复锚点 `#network-architecture`→`#network-structure`（EN heading 为 `Network Structure`）；发现 `sandbox.md` 断链（英文目录下缺失该文件）；术语、代码、LaTeX、图片路径、完整性、可读性均校验通过 |
| deep-learning/generative-models/vae.md | 校审完成 | ✓ | ✓ | Mermaid 标签、术语、代码、LaTeX、完整性均校验通过。修复可读性: `key of`→`key to`、`technical guarantee`→`what makes ... possible`。已知 `sandbox.md` 断链（英文目录下缺失该文件，详见 alexnet-experiment 报告） |
| deep-learning/generative-models/gan.md | 校审完成 | ✓ | ✓ | 术语、LaTeX、Mermaid、nn-arch、图片路径、内部链接、完整性均校验通过。第 100 行原文"第 2、4 步是关键"→EN 为"steps 2 and 5"（步骤 4 为重新采样噪声，步骤 5 为计算生成器损失，英译可能纠正了原文笔误，已标记请用户确认）；「Montreal's famous The Three Brewers bar」定冠词冗余；无可读性问题 |
| deep-learning/generative-models/gan-experiment.md | 校审完成 | ✓ | ✓ | 修复文件末尾多余反引号；术语、代码（注释与字符串）、LaTeX、图片路径、内部链接锚点、完整性、可读性均校验通过。`sandbox.md` 系已知断链（英文目录下缺失该文件，详见 alexnet-experiment 报告）|
| deep-learning/sequence-models/rnn-basics.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、nn-arch 图、链接路径、完整性、可读性均校验通过，无需修复 |
| deep-learning/sequence-models/word-embedding.md | 校审完成 | ✓ | ✓ | 修复断链锚点 `#dot-product-and-projection`→`#inner-product-and-projection`（英文版 vectors.md 标题为 `Inner Product and Projection`）；术语、代码、LaTeX、路径、完整性、可读性均校验通过 |
| deep-learning/sequence-models/lstm-gru.md | 校审完成 | ✓ | ✓ | Mermaid 标签、术语、代码（注释及字符串）、LaTeX、完整性和可读性均校验通过，无需修复。锚点 `#gradient-propagation-and-limitations` 已验证存在于英文版 `rnn-basics.md`。`pack_padded_sequence` 中文注释笔误 `pack_packed_sequence` 已在英译中纠正 |
| deep-learning/sequence-models/seq2seq.md | 校审完成 | ✓ | ✓ | Mermaid 标签、术语、代码、LaTeX、nn-arch 图、链接路径、完整性、可读性均校验通过，无需修复 |
| deep-learning/sequence-models/lstm-experiment.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、完整性、可读性均校验通过；`sandbox.md` 系已知断链（英文目录下缺失该文件） |
| language-models/architecture-basics/transformer-architecture.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、Mermaid、nn-arch 图、图片路径、内部链接锚点、完整性、可读性均校验通过，无需修复 |
| language-models/architecture-basics/architecture-evolution.md | 校审完成 | ✓ | ✓ | 锚点译英；内部 `.md` 链接保留原相对路径；图片 `./assets/`→`../../../language-models/architecture-basics/assets/`；Mermaid 标签译英；练习题内容译英。校审：术语、LaTeX、Mermaid、nn-arch 图、图片路径、内部链接锚点、完整性、可读性均校验通过，无需修复 |
| language-models/architecture-basics/language-model-tokenization.md | 校审完成 | ✓ | ✓ | 修复字节级 BPE 段残文乱码（`["learning"]`→`["学", "习"]`，移除译者随想式旁白）；术语、锚点链接、代码、LaTeX、完整性、可读性均校验通过 |
| language-models/architecture-basics/llm-pretrain-experiment.md | 校审完成 | ✓ | ✓ | 链接锚点译英、内部 `.md` 链接路径、代码注释与字符串、术语、LaTeX、完整性、可读性均校验通过，无需修复 |
| language-models/alignment/rlhf.md | 校审完成 | ✓ | ✓ | 术语、代码、LaTeX、Mermaid、完整性、可读性均校验通过，无需修复。发现 1 处断链：`probability-numpy.md`（EN appendixes 目录尚为空）系已知未翻译文件；`alignment-new-paradigms.md` 的断链已随该文件翻译完成而修复 |
| language-models/alignment/alignment-new-paradigms.md | 校审完成 | ✓ | ✓ | 术语、Mermaid、链接路径、LaTeX、完整性、可读性均校验通过，无需修复 |
| language-models/alignment/llm-dpo-experiment.md | 校审完成 | ✓ | ✓ | 锚点译英（`#数据管理`→`#data-management`）；内部 `.md` 链接保留原相对路径；LaTeX 不变；代码注释译英；`::: info` `::: details` 标题译英；JSON 示例内容译英；表格内容译英；术语、代码、LaTeX、内部链接、完整性、可读性均校验通过，无需修复；`sandbox.md` 系已知断链（EN 目录下缺失该文件）|
| language-models/reasoning/test-time-compute.md | 校审完成 | ✓ | ✓ | Mermaid 标签、术语、图片路径、链接锚点、LaTeX、完整性、可读性均校验通过。发现 `probability-numpy.md` 断链（EN `appendixes/numpy/` 目录为空，系已知未翻译文件）|
| language-models/reasoning/chain-of-thought.md | 校审完成 | ✓ | ✓ | 术语、图片路径、链接锚点、LaTeX、完整性、可读性均校验通过，无需修复 |
| language-models/reasoning/inference-efficiency.md | 校审完成 | ✓ | ✓ | 术语、Mermaid 标签、链接锚点（6 处英文锚点均已验证存在）、LaTeX、代码、完整性、可读性均校验通过。发现 3 处可优化：① line 94 "离线再平衡"→"online rebalancing"（建议保持 EN "online"，因描述内容为迁移运行中请求，属在线操作）；② line 143 "verifies them" 丢失 ZH "逐个" nuance，建议补为 "verifies them one by one"；③ line 148 "relative to the efficiency difference with" 略显冗长，建议简化为 "relative to the traditional approach"；④（中英原文共有 bug）line 39 公式参数 128×128 中 n_head 应为 64 而非 128，但计算结果 10.7GB 正确 |
| language-models/reasoning/llm-reasoning-experiment.md | 校审完成 | ✓ | ✓ | 修复术语 `inference saturation` → `reasoning decay`（6 处）与原文 `推理衰减模型` 一致；修复阶段三概述锚点 `#speculative-decoding` → 无锚点（该段涵盖量化/KV Cache/投机解码三个方向）；代码、LaTeX、表格、`::: info`、完整性、可读性均校验通过 |
| language-models/reasoning/reasoning-reliability.md | 校审完成 | ✓ | ✓ | 内部 `.md` 链接保留原相对路径；锚点译英；图片路径 `./assets/`→`../../../language-models/reasoning/assets/`；代码译英；术语、LaTeX、完整性、可读性均校验通过；修复残留中文「思路」「偶然的错误」共 2 处未译 |
| language-models/frontier/multimodal-llm.md | 校审完成 | ✓ | ✓ | 术语、LaTeX、Mermaid、图片路径、链接锚点、完整性、可读性均校验通过，无需修复 |
| language-models/frontier/vlm-training-experiment.md | 校审完成 | ✓ | ✓ | nn-arch 图标签译英、图片路径指向正确、锚点 `#training-multimodal-models` 已校验存在、代码注释与字符串译英、术语一致、完整性及可读性均校验通过，无需修复 |
| language-models/pretraining/pretraining-data.md | 校审完成 | ✓ | ✓ | 修复残留中文 5 处：`随处可见`→`that are everywhere`（Wikipedia 段）、`大量`→`large amounts of`（毒性过滤段）、`解题`→`problem-solving`（数据污染段）、`普遍`→`universally`（数据污染段）、`倾向于`→`tending to`（合成数据段）；术语、代码、LaTeX、Mermaid、锚点链接、完整性、可读性均校验通过 |
| language-models/pretraining/scaling-laws.md | 校审完成 | ✓ | | 修复 4 处断链链接（`supervised-finetuning.md`→`../../../language-models/pretraining/supervised-finetuning.md`、`../alignment/rlhf.md`→`../../../language-models/alignment/rlhf.md`、`../../appendixes/numpy/probability-numpy.md#monte-carlo-method`→`../../../appendixes/numpy/probability-numpy.md#蒙特卡洛方法`、`../reasoning/test-time-compute.md`→`../../../language-models/reasoning/test-time-compute.md`）；修复 over-training 术语注法使其对英语读者更自然；术语、图片路径、LaTeX、完整性、可读性均校验通过 |
| language-models/pretraining/distributed-training.md | 校审完成 | ✓ | ✓ | 术语、Mermaid、链接路径、图片路径、LaTeX、完整性、可读性均校验通过，无需修复 |
| language-models/pretraining/llm-sft-experiment.md | 校审完成 | ✓ | ✓ | 术语、代码（注释与字符串）、LaTeX、链接路径、完整性、可读性均校验通过；`sandbox.md` 系已知断链（英文目录下缺失该文件，详见 alexnet-experiment 报告）|
| language-models/pretraining/supervised-finetuning.md | 校审完成 | ✓ | ✓ | Mermaid、链接路径、术语、代码、LaTeX、完整性均校验通过，无需修复。修复可读性 3 处：`training effect`→`training effectiveness`；`someone who already knows English learning a British accent`→`someone who already knows English and is learning a British accent`；`is not unfamiliar to us`→`is familiar` |
| language-models/frontier/evaluation-safety.md | 校审完成 | ✓ | ✓ | 修复 Causal Tracing Mermaid 输入字符串遗漏 `___` 填空占位符导致的示例自洽性 bug（正文与 Mermaid 示例不一致）；术语、LaTeX、图片路径、链接锚点、完整性、可读性均校验通过 |
| ai-infra-engineering/model-serving/inference-service-architecture.md | 校审完成 | ✓ | ✓ | 术语、Mermaid 标签、链接锚点（7 处英文锚点均已验证存在）、LaTeX、代码（注释与字符串译英）、完整性、可读性均校验通过。`gpu-resource-management.md` 系已知断链（中文源目录存在对应文件，英文目录尚未翻译）|
| ai-infra-engineering/model-serving/request-scheduling.md | 校审完成 | ✓ | ✓ | 术语、Mermaid 标签、LaTeX、链接锚点（6 处英文锚点均已验证存在）、图片路径（2 图指向源目录 assets）、完整性、可读性均校验通过，无需修复 |
| ai-infra-engineering/model-serving/gpu-resource-management.md | 校审完成 | ✓ | ✓ | 术语、LaTeX、链接锚点（3 处英文锚点均已验证存在）、图片路径、完整性、可读性均校验通过；建议优化 `Slow food running`→`Slow food runners`/`Slow food delivery`；修复 `inference-service-architecture.md` 中已知断链 |
| ai-infra-engineering/model-serving/llm-inference-experiment.md | 校审完成 | ✓ | ✓ | 锚点译英（`#并发性能与显存调优`→`#concurrent-performance-and-memory-tuning`、`#流式输出与-kv-cache-实验`→`#streaming-output-and-kv-cache-experiment`、`inference-service-architecture.md#流式输出与-server-sent-events`→`#streaming-output-and-server-sent-events`、`request-scheduling.md#前缀缓存`→`#prefix-caching`）均已验证存在于目标文件；代码注释与字符串译英；术语、代码、LaTeX、锚点链接、完整性、可读性均校验通过，无需修复；仅发现 1 处细微语义丢失：行尾"认证鉴权"→"authentication"（缺 authorization）|
| ai-infra-engineering/mlops/data-versioning.md | 校审完成 | ✓ | ✓ | Mermaid 标签、锚点、术语、代码、完整性、可读性均校验通过；L33 残留中文"采集"已修正为 `sensor readings` |
| ai-infra-engineering/mlops/model-lifecycle.md | 校审完成 | ✓ | ✓ | 术语、Mermaid 标签、锚点链接、图片路径 `assets/timeline.png`→`../../ai-infra-engineering/mlops/assets/timeline.png`、代码（无代码块）、LaTeX（无）、完整性、可读性均校验通过，无需修复 |
| ai-infra-engineering/mlops/model-performance-monitoring.md | 校审完成 | ✓ | ✓ | 修复图片路径 `../../ai-infra-engineering/mlops/assets/`→`../../../ai-infra-engineering/mlops/assets/`（4 处路径少一层 ..，指向不存在的 en/ 子目录）；修复冗余"PSI index"→"PSI"（PSI 中的 I 已表示 Index）；术语、代码、LaTeX、Mermaid 标签、链接锚点、完整性、可读性均校验通过 |
| ai-infra-engineering/mlops/hyperparameter-optimization.md | 校审完成 | ✓ | ✓ | 内部 `.md` 链接、锚点译英、Mermaid 标签、代码注释、术语、LaTeX、完整性、可读性均校验通过；修复图片路径 `./assets/rs-tpe.png`→`../../ai-infra-engineering/mlops/assets/rs-tpe.png`（EN 目录下无 assets 子目录，图片在源目录）|
| ai-infra-engineering/mlops/drift-detection.md | 校审完成 | ✓ | ✓ | 修复 3 处锚点链接：`#label-delay-issue`→`#label-delay-problem`、`#statistical-test-methods`→`#statistical-testing-methods`、`#autoencoder-principles`→`#autoencoder-fundamentals`；术语、图片路径、完整性、可读性均校验通过 |
| agent-applications/vector-retrieval-rag/rag-experiment.md | 翻译完成 | ✓ | | 内部 `.md` 链接保留原相对路径；锚点译英；代码注释与字符串译英；术语、LaTeX、完整性均校验通过 |
| agent-applications/vector-retrieval-rag/retrieval-quality.md | 翻译完成 | ✓ | | 内部 `.md` 链接保留原相对路径；锚点译英；Mermaid 标签译英；代码注释与字符串译英；术语、LaTeX、完整性均校验通过 |
| agent-applications/vector-retrieval-rag/embedding-and-indexing.md | 翻译完成 | ✓ | | 内部 `.md` 链接保留原相对路径；锚点译英；Mermaid 标签译英；代码注释译英；术语、LaTeX、完整性均校验通过 |
| agent-applications/agent-systems/llm-to-agent.md | 翻译完成 | ✓ | | 内部 `.md` 链接保留原相对路径；锚点译英；Mermaid 标签译英；代码注释译英；术语、LaTeX、完整性均校验通过 |
| agent-applications/vector-retrieval-rag/retrieval-augmented-generation.md | 翻译完成 | ✓ | | 内部 `.md` 链接保留原相对路径；锚点译英；代码注释译英；术语、LaTeX、完整性均校验通过 |