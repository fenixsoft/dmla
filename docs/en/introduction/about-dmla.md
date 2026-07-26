# About Designing Machine Learning Applications

In my outline, "[Designing Machine Learning Applications](https://ai.icyfenix.cn/en)" (DMLA) will span about 700,000 Chinese characters, making it the longest among the nine computer books I have published, and also the most technically intensive. Learning energy is a finite resource, so I hope you will take a moment before diving in to read this article and decide whether this document is what you need, and whether its approach suits where you are right now.

DMLA's core purpose is to help engineers build a systematic understanding of machine learning from the ground up. It begins with the prerequisite mathematical foundations, surveys classical statistical learning methods, then focuses on neural network models -- especially convolutional models, generative models, sequence models derived from deep learning, and language models based on the Transformer architecture -- and finally applies theoretical knowledge to software systems. DMLA aims to help you understand the mechanisms and principles of machine learning, so you can train models yourself, implement algorithms, and know how to put that knowledge into production.

The software industry today is being reshaped by AI, undergoing changes more dramatic than the first and second software crises. A large number of software practitioners are flooding into the AI track. This is driven by real supply-and-demand dynamics, genuine curiosity and legitimate career needs, and undoubtedly some anxiety in these turbulent times. Not every software developer needs to learn Transformer internals, understand attention mechanisms, derive backpropagation, or compute gradients by hand. To that end, I define four levels of collaboration between programmers and AI. Please take a moment to assess which category you fall into now, and which one you hope to reach in the future.

- Level 1: **AI as a pure black-box tool**

  You know libraries like LangChain, can call APIs, pass prompts to models, retrieve results, and let an agent framework drive the workflow forward. You treat AI as a more advanced search engine and rule engine, iterating by tweaking prompts through trial and error.

  There is almost no moat at this stage. Anyone can call the same API; anyone can write the same prompts. You can follow a three-hour tutorial to build a chatbot, or skim a "Seven Days from Zero to Mastery" guide to orchestrate a workflow. So can anyone else -- even AI itself. Developers who follow procedures, write boilerplate code, stitch together libraries from documentation, and make no design decisions or trade-offs are being rapidly eroded by AI in this era.

- Level 2: **Concepts are sufficient; common phenomena can be explained with correct causality**

  You know that sampling temperature affects the diversity of model output, but cannot articulate how the temperature parameter influences results during the Softmax process. You know that in tool-use, the model can be made to produce specific parameter structures for invoking tools, but cannot explain how the model performs constrained decoding to ensure the output precisely matches the tool's structural requirements.

  At this stage, you have accumulated considerable experience using AI. You can explain strange phenomena in terms of mechanisms rather than mysticism, make sound judgments and technology choices, and balance trade-offs based on real-world conditions. But you only have experiential knowledge -- lacking principled understanding, you cannot tell whether combining certain components will cause problems in production. AI technology changes daily; every new concept is genuinely new knowledge for you, requiring careful placement into your existing knowledge framework through accumulated experience.

- Level 3: **Mechanistic intuition; architectural decisions map to engineering consequences**

  You know why Transformer uses RMSNorm instead of LayerNorm, and why removing centering does not affect numerical stability. You know how YaRN improves extrapolation compared to standard RoPE. You understand how to design attention sparsity patterns in resource-constrained environments, what context window size to choose, and how to anchor global attention tokens to build a viable system at a reasonable cost. You understand how each decision affects production in concrete ways. On the other hand, you have never derived backpropagation by hand, computed gradient magnitudes manually, reproduced papers, or conducted rigorous theoretical proofs.

  At this stage, you can control most aspects of the system yourself. Principled knowledge of AI not only guides design decisions in practice but also significantly reduces the [cognitive load](https://icyfenix.cn/tricks/2021/geekbang.html#%E7%A8%8B%E5%BA%8F%E5%91%98%E7%9A%84%E4%BB%B7%E5%80%BC%E8%A7%82) of mastering new knowledge -- which ultimately becomes your own moat.

- Level 4: **Research and innovation; advancing model architectures and methods**

  At this stage, you follow SOTA, discuss new papers on arXiv, and gather information through thinking, experimentation, papers, and conversations with others -- not from books or tutorials. Your profile is typically a PhD student in a machine-learning-related field, or a practitioner working on foundation model research. I imagine the only intersection this document might have with you is if you are a PhD outside the ML field looking to fill in some engineering practice gaps.

DMLA is positioned at the transition zone from Level 2 to Level 3, targeting practitioners with a background in science, engineering, and software engineering. In writing, I assume you have the learning aptitude of a typical science-and-engineering graduate. You may have forgotten the mathematics, but you must have learned it at some point. You should not be afraid to write and read code -- most articles in DMLA are accompanied by code that runs directly in the page to aid explanation.

- If you are not a developer -- a product manager, executive, or someone outside the industry -- and simply want to understand what AI is and what it can do, perhaps *[The Boundary of Intelligence](https://book.douban.com/subject/30379536/)* (智慧的疆界) would be a better fit.
- If you want to build a working demo, there are plenty of quick-start tutorials online that promise mastery in hours or days. DMLA focuses on production-grade systems, which requires patience.
- If you are already a PhD or researcher in this field, DMLA's content may be too basic for you, and its code-heavy narrative style might not suit your needs. This is written for engineers, not for scientists.

By now, I believe you have enough information to decide whether to continue reading this document. If you are ready, let us officially begin this journey of discussion and code practice around machine learning.
