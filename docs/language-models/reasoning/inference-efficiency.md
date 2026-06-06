# 推理效率优化 —— 让推理跑得更快

在上一章中，我们探讨了 Test-Time Compute Scaling——推理阶段投入更多算力可以获得更好的答案。但现实世界对推理提出了另一个约束：**速度与成本**。一个需要"思考 30 秒"才能回答的模型，在实时对话场景中几乎不可用；一个需要 8 张 H100 才能运行的推理服务，其部署成本令大多数团队望而却步。

推理效率优化就是在"答得好"和"答得快"之间寻找平衡。从 KV Cache 的内存优化，到 Prefill-Decode 分离的架构创新，到投机解码的"猜答案"策略，再到模型本身的轻量化，这些技术共同构成了大模型推理的工程体系。本文将系统梳理推理效率优化的核心技术与思路。

## 推理瓶颈分析

### 自回归生成的计算特征

分析自回归生成中 Prefill（预填充）与 Decode（解码）两阶段的计算模式差异：Prefill 阶段输入 token 可并行处理，属于计算密集型；Decode 阶段逐 token 生成，每步依赖上一步输出，属于访存密集型。这一根本差异是后续所有优化技术的出发点。

### 显存与算力的矛盾

总结推理服务的核心矛盾：Decode 阶段 GPU 算力利用率极低（常低于 5%），但显存已被 KV Cache 占满，无法增加 batch size 来提升算力利用率。这个"显存墙"问题驱动了后续的优化方向。

## PagedAttention：分页管理 KV Cache

介绍 vLLM 提出的 PagedAttention 机制，借鉴操作系统虚拟内存的分页思想，将 KV Cache 按固定大小的 block 管理，消除显存碎片，支持跨请求共享（如 system prompt 的 KV Cache 复用）。对比传统连续分配方式下的显存浪费。


## Prefill-Decode 分离架构

### 为什么分离 Prefill 与 Decode

深入分析 Prefill 与 Decode 的资源需求差异：Prefill 需要高算力（大量矩阵乘法），Decode 需要高显存带宽（逐 token 读取 KV Cache）。将两者混合在同一 GPU 上执行，导致资源利用不均衡——Prefill 时算力满载但显存带宽闲置，Decode 时显存带宽满载但算力闲置。

### PD 分离的基本架构

介绍 Prefill-Decode 分离架构：Prefill 实例专注处理输入 prompt（高算力 GPU，如 H100），Decode 实例专注生成 token（高带宽 GPU 或更经济的配置），中间通过高速网络传输 KV Cache。分析 Splitwise、DistServe 等系统的设计思路。

### KV Cache 传输与调度

讨论 PD 分离中的工程挑战：KV Cache 跨节点传输的延迟与带宽需求、Prefill 与 Decode 实例的负载均衡调度、请求路由策略（如何决定哪个 Decode 实例接收新生成的 KV Cache）。

### Mooncake 与 Disaggregated Serving

以 Mooncake 为例介绍更极致的分离架构——利用闲置 GPU 资源构成"KV Cache 池"，实现请求级弹性调度。讨论分离架构在多租户场景下的优势与挑战。

## 投机解码

### 投机解码的基本思想

介绍投机解码的核心思路：用一个小模型（draft model）快速生成多个候选 token，大模型（target model）一次前向传播同时验证所有候选，接受正确的、拒绝错误的。以"先猜后验"替代"逐个生成"，在不改变输出分布的前提下加速推理。

### Speculative Sampling 的理论保证

说明投机解码如何保证输出分布与原始自回归采样完全一致——通过修正采样概率，使得接受/拒绝机制不改变目标分布。这是投机解码区别于其他近似加速方法的关键优势。

### Draft Model 的选择与训练

讨论 draft model 的选择标准：足够小以保证快速生成、足够准以提高接受率。介绍 draft model 的训练策略，以及 Medusa 等无需独立 draft model 的方案（在目标模型上添加多个预测头）。

### 推理加速比分析

分析投机解码的加速比取决于接受率与推测长度：接受率越高、推测长度越长，加速比越大。讨论不同任务类型下接受率的差异（代码生成接受率高，开放对话接受率低），以及实际系统中的加速效果。

## 模型轻量化

### 知识蒸馏：从小模型学大模型

介绍知识蒸馏在 LLM 中的应用：如何将大模型的推理能力迁移到小模型，蒸馏数据的选择（用大模型的 CoT 输出作为小模型的训练数据），以及蒸馏模型与原始小模型的性能差距。

### 剪枝与稀疏化

介绍 LLM 剪枝的挑战——传统结构化剪枝在 LLM 上效果有限的原因，以及非结构化稀疏（如 SparseGPT）与半结构化稀疏（如 2:4 稀疏模式，NVIDIA Ampere+ 架构原生支持）的实践。

## 推理框架与部署实践

### 主流推理框架对比

对比 vLLM、TensorRT-LLM、SGLang、LMDeploy 等推理框架的设计理念与适用场景：vLLM 的 PagedAttention 与连续批处理、TensorRT-LLM 的算子融合与内核优化、SGLang 的 RadixAttention 与编程模型。

### Continuous Batching 与调度策略

介绍 Continuous Batching（连续批处理）如何替代传统 Static Batching：请求完成即可释放资源并加入新请求，显著提升吞吐量。分析不同调度策略（FCFS、优先级调度、长短请求分离）对延迟与吞吐的影响。

### 推理服务的关键指标

定义推理服务的核心指标：Time to First Token（TTFT）、Tokens Per Second（TPS）、吞吐量（Throughput）、并发数（Concurrency）。讨论这些指标之间的 trade-off 关系，以及不同应用场景（实时对话 vs 批量处理）的优化目标差异。

## 小结

## 练习题

## 参考资料
