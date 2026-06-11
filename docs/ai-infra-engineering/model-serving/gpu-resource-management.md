# GPU 资源管理

## 核心问题

GPU 是 LLM 推理服务中最昂贵也最稀缺的资源。一块 A100 80GB 价值数万元，运行 70B 模型时可能只能同时处理不到 10 个请求。如何让每块 GPU 的每一 MB 显存、每一个 Tensor Core 都被充分利用，是推理服务成本控制的关键。本文从 GPU 的硬件架构出发，分析显存、算力和带宽三大资源的特征与瓶颈，讨论显存管理、算力调度、多实例共享以及量化等资源优化技术。

## 第一章：GPU 硬件架构与资源特征

### 1.1 GPU 的三大资源：显存、算力、带宽

- **显存**（HBM，High Bandwidth Memory）：存储模型权重和 KV Cache。A100 80GB、H100 80GB，容量固定且有限。显存决定了能部署多大的模型、能同时处理多少请求
- **算力**（FLOPS，Floating Point Operations Per Second）：GPU 的计算能力。A100 的 FP16 算力为 312 TFLOPS，H100 为 990 TFLOPS。算力决定了每秒能处理多少 token
- **带宽**（Memory Bandwidth）：GPU 与显存之间的数据传输速率。A100 的 HBM2e 带宽为 2 TB/s，H100 为 3.35 TB/s。带宽决定了 Decode 阶段能多快读取 KV Cache
- 三者之间的关系：算力的发挥依赖带宽（数据必须从显存加载到计算单元），显存容量限制了能存储的数据量。三者必须匹配，任何一项成为瓶颈都会拖累其他两项

### 1.2 算术强度与 Roofline 模型

- **算术强度**（Arithmetic Intensity）：每字节显存访问对应的浮点运算次数，单位为 FLOP/Byte。算术强度高的是计算密集型（算力瓶颈），低的是访存密集型（带宽瓶颈）
- **Roofline 模型**：以算术强度为横轴、性能（FLOPS）为纵轴，画出硬件的性能上限曲线。曲线分为两段：左侧上升段（带宽瓶颈区，性能 = 带宽 × 算术强度）和右侧平台段（算力瓶颈区，性能 = 峰值算力）
- LLM 推理在 Roofline 上的位置：Prefill 阶段的算术强度高（矩阵乘法的计算量大），位于平台段（算力瓶颈）；Decode 阶段的算术强度极低（每步只生成一个 token），位于上升段（带宽瓶颈）
- Roofline 模型对推理优化的指导：Decode 阶段优化带宽比优化算力更有效，Prefill 阶段则相反

> **可视化建议**：
> - Roofline 模型图：标注 Prefill 和 Decode 在曲线上的位置
> - 对比表格：A100 与 H100 在显存、算力、带宽上的参数对比

```python runnable
# 绘制 Roofline 模型，展示 Prefill 和 Decode 的性能瓶颈
import numpy as np

# A100 参数
peak_flops = 312e12     # 312 TFLOPS (FP16)
bandwidth = 2e12        # 2 TB/s (HBM2e)

# 算术强度范围
arithmetic_intensity = np.logspace(-1, 3, 1000)  # FLOP/Byte

# Roofline 性能上限
roofline = np.minimum(bandwidth * arithmetic_intensity, peak_flops)

# Prefill 和 Decode 的典型算术强度
prefill_intensity = 100   # FLOP/Byte（计算密集）
decode_intensity = 1      # FLOP/Byte（访存密集）

# 它们的实际性能
prefill_perf = min(bandwidth * prefill_intensity, peak_flops)
decode_perf = min(bandwidth * decode_intensity, peak_flops)

print("A100 Roofline 分析")
print(f"峰值算力: {peak_flops/1e12:.0f} TFLOPS")
print(f"峰值带宽: {bandwidth/1e12:.1f} TB/s")
print(f"拐点算术强度: {peak_flops/bandwidth:.0f} FLOP/Byte")
print()
print(f"Prefill (算术强度={prefill_intensity}): 性能={prefill_perf/1e12:.1f} TFLOPS, 瓶颈={'算力' if bandwidth*prefill_intensity >= peak_flops else '带宽'}")
print(f"Decode (算术强度={decode_intensity}): 性能={decode_perf/1e12:.1f} TFLOPS, 瓶颈={'算力' if bandwidth*decode_intensity >= peak_flops else '带宽'}")
print()
print(f"Decode 阶段算力利用率: {decode_perf/peak_flops*100:.1f}%")
print(f"Prefill 阶段算力利用率: {prefill_perf/peak_flops*100:.1f}%")
```

### 1.3 GPU 架构演进对推理的影响

- 从 A100 到 H100 的演进：算力提升 3 倍（312→990 TFLOPS），带宽提升 1.7 倍（2→3.35 TB/s），显存容量不变（80GB）
- 算力增长快于带宽增长，意味着 Decode 阶段（带宽瓶颈）从 A100 到 H100 的加速比远小于 Prefill 阶段（算力瓶颈）。这进一步验证了 PD 分离架构的必要性
- NVLink 与 NVSwitch：节点内 GPU 互连带宽（NVLink 4.0 900 GB/s）远高于节点间网络（InfiniBand 400 Gb/s），决定了张量并行（节点内）和流水线并行（节点间）的性能边界

## 第二章：显存管理

### 2.1 显存预算分析

- GPU 显存被三部分瓜分：模型权重、KV Cache、运行时临时缓冲区
- 以 LLaMA-2 70B（float16）为例：模型权重约 140 GB（需要 2×A100 80GB 的张量并行）、每块 GPU 上的权重约 70 GB、剩余约 10 GB 给 KV Cache 和运行时、单个请求的 KV Cache 约 10 GB，意味着每块 GPU 只能容纳 1 个请求的 KV Cache
- 显存预算决定了并发上限：并发数 = (可用显存 - 模型权重 - 运行时) / 单请求 KV Cache 大小

### 2.2 KV Cache 显存优化

- KV Cache 的显存占用公式（回顾）：$M_{\text{KV}} = 2 \times n_{\text{layer}} \times d_{\text{head}} \times n_{\text{head}} \times n_{\text{max}} \times b \times sizeof(\text{dtype})$
- 降低 KV Cache 显存占用的三个方向：
  - 降低精度：从 float16 到 float8，显存减半。KV Cache 量化对生成质量的影响小于权重量化，因为注意力计算对精度的敏感度较低
  - 降低序列长度上限：限制最大序列长度 $n_{\max}$，超出则截断。简单但限制了长文本处理能力
  - 压缩 KV Cache：通过低秩近似或选择性丢弃不重要的注意力向量来减少 KV Cache 大小。如 Scissorhands（保留"重要"token 的 KV Cache，丢弃其余）和 GECKO（基于注意力权重的自适应压缩）
- 框架实例：vLLM 支持的 KV Cache 量化（FP8 E5M2 格式），在 A100 上将 KV Cache 显存减半，生成质量损失可忽略

### 2.3 显存碎片与内存池

- GPU 显存分配器（如 PyTorch 的 CUDACachingAllocator）使用缓存分配策略：释放的内存块不立即归还给 CUDA，而是缓存在池中供后续分配使用
- 碎片化问题：不同大小的分配请求可能将连续的显存空间切割成不连续的碎片块，即使总空闲显存充足，也无法分配大的连续块
- PagedAttention 的 Block 分配器从根本上解决了 KV Cache 的碎片问题（固定大小的 Block 不会产生外部碎片），但非 KV Cache 的显存分配（如临时张量）仍可能面临碎片化
- 显存池的监控与调优：跟踪显存使用率、碎片率、分配/释放频率，及时发现显存泄漏和碎片化问题

> **可视化建议**：
> - 饼图：GPU 显存预算分配（模型权重 vs KV Cache vs 运行时）
> - 对比图：不同 KV Cache 优化策略下的并发容量

```python runnable
# 分析不同模型在 A100 80GB 上的显存预算
import numpy as np

models = {
    "LLaMA-2 7B": {"params_B": 7, "layers": 32, "heads": 32, "head_dim": 128, "tp": 1},
    "LLaMA-2 13B": {"params_B": 13, "layers": 40, "heads": 40, "head_dim": 128, "tp": 1},
    "LLaMA-2 70B": {"params_B": 70, "layers": 80, "heads": 64, "head_dim": 128, "tp": 4},
}

gpu_memory_gb = 80        # A100 80GB
max_seq_len = 4096
dtype_size = 2            # float16
runtime_overhead_gb = 2   # 运行时临时缓冲区

print(f"{'模型':<15} | {'权重(GB)':<10} | {'单请求KV(GB)':<12} | {'可用KV(GB)':<10} | {'最大并发'}")
print("-" * 70)

for name, m in models.items():
    weight_gb = m["params_B"] * dtype_size  # float16 每参数 2 字节
    weight_per_gpu = weight_gb / m["tp"]

    # 单请求 KV Cache
    kv_per_request = (2 * m["layers"] * m["head_dim"] * m["heads"]
                       * max_seq_len * dtype_size) / (1024**3)

    available_kv = gpu_memory_gb - weight_per_gpu - runtime_overhead_gb
    max_concurrency = int(available_kv / kv_per_request) if kv_per_request > 0 else 0

    print(f"{name:<15} | {weight_per_gpu:>7.1f}   | {kv_per_request:>9.2f}    | {available_kv:>7.1f}   | {max_concurrency}")
```

## 第三章：算力调度与利用率优化

### 3.1 GPU 利用率的度量

- SM 占用率（SM Occupancy）：GPU 流多处理器（Streaming Multiprocessor）上活跃线程束（Warp）的比例，反映了 GPU 计算单元的利用程度
- 算力利用率（FLOPS Utilization）：实际 FLOPS / 峰值 FLOPS，是更直观的效率指标。LLM Decode 阶段的算力利用率通常只有 1-5%
- 显存带宽利用率（Bandwidth Utilization）：实际带宽 / 峰值带宽。Decode 阶段的带宽利用率通常在 30-60%，说明带宽也没有被充分利用
- 利用率低不等于效率低：某些场景下利用率低是不可避免的（如批量大小为 1 的 Decode），关键是在约束条件下尽可能提高利用率

### 3.2 Kernel 优化与算子融合

- GPU Kernel 是在 GPU 上执行的最小计算单元。LLM 推理涉及大量小 Kernel（LayerNorm、Attention、MLP、Activation），每个 Kernel 的启动开销和数据搬运开销累积起来不可忽视
- 算子融合（Operator Fusion）：将多个连续的 Kernel 合并为一个，减少中间结果的显存写入和读取。例如将 LayerNorm + Attention + Residual 融合为一个 Kernel
- 框架实例：FlashAttention 是算子融合的典型代表，将 Attention 的分块计算与 Softmax 融合，避免了 $N \times N$ 注意力矩阵的显存写入，同时利用 SRAM 的局部性减少了 HBM 访问
- TensorRT-LLM 的 Kernel 优化：针对 NVIDIA GPU 的 Tensor Core 专门优化的 GEMM Kernel，融合了多种算子，在 H100 上可以达到 50% 以上的算力利用率

### 3.3 CUDA Stream 与并行执行

- CUDA Stream 是 GPU 上的任务队列，同一 Stream 内的操作串行执行，不同 Stream 的操作可以并行执行
- LLM 推理中 CUDA Stream 的使用：将 KV Cache 的传输（CPU→GPU 或 GPU→GPU）与计算放在不同 Stream 上，实现计算与传输的并行
- 多 Stream 的调度挑战：Stream 之间的同步点设计。计算 Stream 必须等待数据传输完成才能开始，过多的同步点会抵消并行收益

> **可视化建议**：
> - Roofline 图：标注不同 Kernel 优化后的性能位置
> - 时序图：CUDA Stream 的并行执行时序

## 第四章：多实例 GPU 共享

### 4.1 时间分片与空间分片

- **时间分片**（Time-Slicing）：多个推理实例轮流使用同一块 GPU，每个实例获得固定的时间片。NVIDIA 的 MPS（Multi-Process Service）是一种时间分片实现
- **空间分片**：将一块 GPU 的计算单元和显存划分为多个独立分区，每个分区运行一个推理实例。NVIDIA MIG（Multi-Instance GPU）是空间分片的硬件支持
- 时间分片的问题：上下文切换开销（切换实例时需要保存/恢复 GPU 状态），且不同实例可能互相干扰（一个实例的长时间计算会延迟其他实例）
- 空间分片的优势：硬件级隔离，互不干扰。A100 支持 7 个 MIG 实例（每个约 10 GB 显存），H100 支持更多

### 4.2 MIG 的应用场景

- 用 MIG 将一块 A100 80GB 划分为多个实例，分别运行不同大小的模型。例如划分 2 个 40GB 实例，一个运行 7B 模型，另一个运行 13B 模型
- MIG 的局限：划分后的实例显存和算力都受限，不适合大模型（70B+）的推理。且 MIG 实例之间无法通过 NVLink 通信，不能用于张量并行
- MIG 的最佳实践：用于多个小模型的并发推理，或开发测试环境中的多用户隔离

### 4.3 推理框架的多模型调度

- 不使用 MIG 的多模型共享：推理框架（如 Triton Inference Server）在同一块 GPU 上部署多个模型，通过调度器控制各模型的执行顺序
- 动态批处理的多模型扩展：将不同模型的请求分开批处理，但共享 GPU 的执行时间
- 模型热切换：在不重启服务的情况下加载/卸载模型。模型权重从磁盘加载到显存需要数十秒，通过权重预加载到 CPU 内存可以加速到数秒

> **可视化建议**：
> - 对比图：时间分片 vs MIG 空间分片的资源分配方式
> - 示意图：A100 的 MIG 分区配置示例

## 第五章：量化与精度优化

### 5.1 量化的基本原理

- 量化的核心思想：用更低精度的数值格式表示模型权重和/或激活值，减少显存占用和计算量。从 float32 到 float16 已经是一种量化（2 倍压缩），但推理优化的量化目标是 int8（4 倍压缩）甚至 int4（8 倍压缩）
- 量化的分类：**训练后量化**（Post-Training Quantization，PTQ）不需要重新训练，直接量化已有模型；**量化感知训练**（Quantization-Aware Training，QAT）在训练过程中模拟量化误差，训练后量化的精度损失更小
- 量化对推理的收益：显存占用减少（更多请求可以同时处理）、计算速度提升（int8 的 Tensor Core 吞吐量是 float16 的 2 倍）、带宽需求降低（每字节传输更多参数）

### 5.2 权重量化：从 GPTQ 到 AWQ

- **GPTQ**（2022 年，弗兰塔·伊莱亚斯（Elias Frantar）等人）：基于近似二阶信息的训练后量化方法，逐层量化权重矩阵，通过 Hessian 矩阵补偿量化误差。在 4-bit 量化下仍能保留原始模型 99% 的性能
- **AWQ**（2023 年，林智杰（Ji Lin）等人）：激活感知权重量化，发现权重中只有约 1% 的通道对激活值敏感，对这些"重要通道"保持较高精度，其余通道激进量化。比 GPTQ 更快（无需逐行计算 Hessian），且精度相当
- **GGUF**（2023 年，Georgi Gerganov）：为 CPU 推理优化的量化格式，支持多种量化精度（2-bit 到 8-bit 混合），使得 70B 模型可以在消费级硬件上运行。GGUF 的核心创新是按重要性对不同层使用不同精度

### 5.3 KV Cache 量化

- KV Cache 量化与权重量化的区别：权重是静态的（量化后不再变化），KV Cache 是动态的（每个请求都不同，且随生成过程增长）
- KV Cache 量化的挑战：量化精度对注意力计算的影响。注意力分数 $\text{softmax}(QK^T/\sqrt{d})$ 对 Key 的精度敏感，量化可能导致注意力分布偏移
- 实践中的 KV Cache 量化：FP8 E5M2 格式是当前最主流的 KV Cache 量化方案。与权重 FP8 量化不同，KV Cache 的 FP8 使用 E5M2（5 位指数、2 位尾数）而非 E4M3，因为 KV Cache 的数值范围更大、需要更多指数位
- 框架实例：vLLM 的 KV Cache FP8 量化，在 A100 上将 KV Cache 显存减半，吞吐量提升 30-50%，而生成质量的影响在大多数任务上可忽略

### 5.4 量化的精度-效率权衡

- 量化不是免费的午餐：精度越低，显存和速度收益越大，但生成质量损失也越大。4-bit 量化在简单任务上损失可忽略，但在数学推理和代码生成等需要精确计算的任务上可能有明显退化
- 混合精度量化：对模型的不同层使用不同的量化精度。Attention 层的精度对生成质量影响更大，可以保持较高精度；MLP 层的精度影响较小，可以更激进地量化
- 量化评估方法：Perplexity（困惑度）评估量化对语言建模能力的影响、下游任务基准评估（如 MMLU、HumanEval）量化对推理和代码能力的影响、人工评估量化对对话质量的影响

> **可视化建议**：
> - 对比图：不同量化方案（float16、int8、int4）的显存占用与性能损失
> - 表格：各量化方案在不同模型规模上的精度评估结果

```python runnable
# 模拟不同量化方案对显存和并发的影响
import numpy as np

model_params_B = 70  # 70B 参数模型
gpu_memory_gb = 80   # A100 80GB
tp = 4               # 张量并行度

quantization_configs = [
    {"name": "FP16", "weight_bytes": 2, "kv_bytes": 2, "perf_retention": 1.00},
    {"name": "FP16 + KV FP8", "weight_bytes": 2, "kv_bytes": 1, "perf_retention": 0.995},
    {"name": "INT8 Weight + KV FP8", "weight_bytes": 1, "kv_bytes": 1, "perf_retention": 0.99},
    {"name": "INT4 Weight + KV FP8", "weight_bytes": 0.5, "kv_bytes": 1, "perf_retention": 0.97},
]

# KV Cache 参数（70B 模型）
layers = 80
heads = 64
head_dim = 128
max_seq = 4096

print(f"模型: LLaMA-2 70B, GPU: A100 80GB × {tp}")
print(f"{'配置':<25} | {'权重/卡(GB)':<12} | {'KV/请求(GB)':<12} | {'最大并发':<8} | {'性能保留'}")
print("-" * 75)

for cfg in quantization_configs:
    weight_per_gpu = model_params_B * cfg["weight_bytes"] / tp
    kv_per_request = (2 * layers * head_dim * heads * max_seq * cfg["kv_bytes"]) / (1024**3)
    available = gpu_memory_gb - weight_per_gpu - 2  # 减去运行时开销
    max_concurrency = max(0, int(available / kv_per_request))

    print(f"{cfg['name']:<25} | {weight_per_gpu:>8.1f}     | {kv_per_request:>8.2f}     | {max_concurrency:>6}   | {cfg['perf_retention']:.1%}")
```

## 本章小结

- GPU 的三大资源（显存、算力、带宽）各有瓶颈，Roofline 模型帮助识别 Prefill 阶段是算力瓶颈、Decode 阶段是带宽瓶颈
- 显存管理是推理服务的命脉。模型权重占去大部分显存后，留给 KV Cache 的空间直接决定了并发上限。PagedAttention 和 KV Cache 量化是扩展并发数的两大利器
- 算力利用率的提升依赖 Kernel 优化和算子融合。FlashAttention 是最成功的案例，TensorRT-LLM 的 Kernel 优化则代表了厂商级优化的极致
- 多实例 GPU 共享通过时间分片或空间分片（MIG）提升 GPU 利用率，适合多小模型并发场景
- 量化是降低资源需求最直接的手段。权重量化（GPTQ、AWQ）减少模型占用的显存和带宽，KV Cache 量化减少推理时的显存占用，两者通常组合使用

## 练习题

1. 使用 Roofline 模型分析：在 H100 上（峰值算力 990 TFLOPS FP16，带宽 3.35 TB/s），LLM Decode 阶段（算术强度约 1 FLOP/Byte）的理论性能是多少？相比 A100（312 TFLOPS，2 TB/s）提升多少倍？这说明了什么问题？

   <details>
   <summary>参考答案</summary>

   Decode 阶段的算术强度为 1 FLOP/Byte，远低于拐点（A100 拐点 = 312/2 = 156 FLOP/Byte，H100 拐点 = 990/3.35 = 296 FLOP/Byte），因此 Decode 在两款 GPU 上都受带宽瓶颈。

   A100 Decode 性能 = 2 TB/s × 1 FLOP/Byte = 2 TFLOPS
   H100 Decode 性能 = 3.35 TB/s × 1 FLOP/Byte = 3.35 TFLOPS

   提升倍数 = 3.35 / 2 = 1.675 倍。而 H100 的峰值算力是 A100 的 990/312 = 3.17 倍。

   这说明对于 Decode 阶段（带宽瓶颈），从 A100 升级到 H100 的加速比远低于峰值算力的提升比例。算力的增长（3.17 倍）远快于带宽的增长（1.675 倍），这意味着新一代 GPU 对 Decode 阶段的性价比提升有限。这也是 PD 分离架构有吸引力的重要原因——Prefill 阶段能充分利用新 GPU 的算力增长。

   </details>

2. 一个 13B 模型（float16）部署在单块 A100 80GB 上。请计算：(1) 模型权重占多少显存？(2) 剩余显存可支持多少并发的 KV Cache（假设最大序列长度 4096，运行时开销 2 GB）？(3) 如果使用 INT4 权重量化 + KV Cache FP8 量化，并发数可以提升到多少？

   <details>
   <summary>参考答案</summary>

   (1) 模型权重：13B × 2 字节 = 26 GB

   (2) 剩余显存：80 - 26 - 2 = 52 GB
   单请求 KV Cache（13B 模型，40 层，40 头，每头 128 维，float16）：
   $2 \times 40 \times 128 \times 40 \times 4096 \times 2$ = $2 \times 40 \times 5120 \times 4096 \times 2$ ≈ 3.33 GB
   并发数 = 52 / 3.33 ≈ 15 个请求

   (3) INT4 权重量化：权重 = 13B × 0.5 字节 = 6.5 GB
   KV Cache FP8：单请求 KV Cache ≈ 3.33 / 2 = 1.67 GB
   剩余显存：80 - 6.5 - 2 = 71.5 GB
   并发数 = 71.5 / 1.67 ≈ 42 个请求

   量化后并发数从 15 提升到 42，提升约 2.8 倍。

   </details>

3. 分析为什么 KV Cache 量化使用 FP8 E5M2 格式而非 E4M3 格式。提示：考虑 KV Cache 中数值的分布特征。

   <details>
   <summary>参考答案</summary>

   FP8 有两种格式：E4M3（4 位指数、3 位尾数，精度更高但范围更小）和 E5M2（5 位指数、2 位尾数，精度更低但范围更大）。

   KV Cache 中的数值分布特征：Key 和 Value 向量是 Transformer 各层的输出，经过 LayerNorm 等归一化后，大部分数值集中在较小范围内，但少数异常值（Outlier）可能非常大。这些异常值在注意力计算中起关键作用（决定了注意力的分布），如果被截断会导致严重的精度损失。

   E5M2 格式可表示的数值范围更大（最大约 57344，而 E4M3 最大约 448），能容纳 KV Cache 中的异常值而不溢出。虽然尾数位数少了一位导致精度略低，但对于注意力计算来说，数值范围的正确性比精度更重要——一个被截断到最大值的大数值对注意力分布的影响远大于一个精度稍低的正常数值。

   相比之下，权重是静态的，在量化时可以逐通道分析数值范围并选择合适的缩放因子，因此权重 FP8 量化通常使用 E4M3（精度更高，范围够用）。

   </details>
