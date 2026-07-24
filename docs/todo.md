- FEAT：AI全文校验
    - 全文代码可执行性检测
    - 全文知识性错误检测
    - 全文语法/笔误/错别字检测
- FEAT：提供英文语言版本
    - 英文语言切换菜单
    - 文章英文翻译
    - 代码注释英文翻译
    - 图片英文翻译
- FEAT：Word文档/PDF文档生成
    - 脚注：多臂老虎机（Multi-Armed Bandit）--> 地址
- FEAT：文章转可视化（视频、PPT）
- FEAT：自动微信公众号发布

- TODO：优化GPU镜像，并回归测试
    1. CUDA 开发包（2,189 MB） 是最大可优化项。如果用 nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 替代 -devel- 镜像，可节省约 2.2 GB，但这会导致 flashinfer JIT 编译失败（需要 nvcc）。可以考虑在构建时编译好
    flashinfer 内核再切换到运行时镜像。
    2. vLLM（3,110 MB） 是单个最大的 Python 包。可以评估是否必需完整安装，或者用 --no-deps 排除部分间接依赖。
    3. 移除  NVIDIA DALI 244 MB
    