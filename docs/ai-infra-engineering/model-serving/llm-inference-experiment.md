# 工程实训：部署 LLM 推理服务

## 核心问题

通过前三章的学习，我们了解了推理服务的架构原理、请求调度与批处理机制、GPU 资源管理策略。本次实训将动手部署一个完整的 LLM 推理服务，从模型加载到 API 服务到性能调优，亲身体验推理服务从零到可用的全过程。实训覆盖模型量化部署、推理服务启动与 API 调用、性能基准测试、流式输出与多轮对话、显存管理与调度策略调优等核心环节。

## 第一章：实训概述与环境准备

### 1.1 实训目标

- 理解 LLM 推理服务的完整部署流程，从模型选择到服务上线
- 掌握推理服务的核心性能指标（TTFT、TPS、吞吐量、并发数）及其测量方法
- 体验量化、批处理、KV Cache 管理等技术对推理性能的实际影响
- 通过动手实验验证前三章讨论的理论知识

### 1.2 实训环境

- 硬件要求：至少一块 NVIDIA GPU（推荐 A100 或 3090 以上，显存 24GB+）
- 软件环境：Python 3.10+、PyTorch 2.0+、vLLM 推理框架
- 模型选择：Qwen2.5-1.5B 或类似小规模模型（确保单卡可部署），用于演示推理服务的基本原理。如有大显存 GPU，可额外尝试 7B 模型对比

### 1.3 环境搭建

- 安装 vLLM 及其依赖
- 下载模型权重（从 Hugging Face 或 ModelScope）
- 验证 GPU 环境与模型加载

```python runnable gpu
# 验证 GPU 环境与 PyTorch CUDA 支持
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  显存总量: {props.total_mem / 1024**3:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  当前显存占用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  当前显存缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("CUDA 不可用，请检查 GPU 驱动和 CUDA 安装")
```

## 第二章：模型加载与推理服务启动

### 2.1 模型加载与显存分析

- 使用 Transformers 加载模型，观察显存占用的三个部分：模型权重、KV Cache 预分配、运行时缓冲区
- 对比不同精度（float32、float16、bfloat16）下的显存占用差异
- 计算模型的理论显存需求与实际显存占用的差异，理解 PyTorch 显存管理器的缓存策略

```python runnable gpu
# 加载模型并分析显存占用
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen2.5-0.5B"  # 使用小模型确保可运行

# 记录加载前的显存状态
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    before_mem = torch.cuda.memory_allocated() / 1024**3

# 加载模型（float16 精度）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 记录加载后的显存状态
if torch.cuda.is_available():
    after_mem = torch.cuda.memory_allocated() / 1024**3
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"加载前显存: {before_mem:.2f} GB")
    print(f"加载后显存: {after_mem:.2f} GB")
    print(f"峰值显存: {peak_mem:.2f} GB")
    print(f"模型权重理论大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.3f} GB (float16)")
```

### 2.2 单次推理与延迟测量

- 执行一次完整的推理（Prefill + Decode），测量各阶段耗时
- 理解 TTFT、TPS 的实际含义：TTFT = Prefill 时间，TPS = 输出 token 数 / Decode 时间
- 观察不同输入长度对 TTFT 的影响、不同输出长度对 TPS 的影响

```python runnable gpu
# 单次推理延迟测量
import torch
import time

def measure_inference_latency(model, tokenizer, prompt, max_new_tokens=50):
    """测量推理的 TTFT 和 TPS"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # 测量完整生成时间
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    torch.cuda.synchronize()
    end_time = time.time()

    output_length = outputs.shape[1] - input_length
    total_time = end_time - start_time

    # 近似 TTFT（Prefill 时间约占总时间的输入长度/总长度比例）
    # 精确测量需要逐 token 生成，此处为简化演示
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"输入长度: {input_length} tokens")
    print(f"输出长度: {output_length} tokens")
    print(f"总时间: {total_time*1000:.0f} ms")
    print(f"TPS (近似): {output_length / total_time:.1f} tokens/s")
    print(f"生成内容: {result[:100]}...")

    return {
        "input_length": input_length,
        "output_length": output_length,
        "total_time_ms": total_time * 1000,
        "tps": output_length / total_time,
    }

# 测试不同长度的提示
short_prompt = "什么是人工智能？"
long_prompt = "请详细解释深度学习的发展历程，从感知机开始，到多层感知机、卷积神经网络、循环神经网络，再到 Transformer 架构。每个阶段请说明其核心创新和代表性工作。"

print("=== 短提示推理 ===")
measure_inference_latency(model, tokenizer, short_prompt, max_new_tokens=50)

print("\n=== 长提示推理 ===")
measure_inference_latency(model, tokenizer, long_prompt, max_new_tokens=50)
```

### 2.3 启动 vLLM 推理服务

- vLLM 的命令行启动方式与参数配置
- 关键参数解读：`--tensor-parallel-size`（张量并行度）、`--gpu-memory-utilization`（GPU 显存利用率上限）、`--max-model-len`（最大序列长度）、`--quantization`（量化方案）
- 通过 OpenAI 兼容 API 发送推理请求

> **可视化建议**：
> - 架构图：vLLM 服务的启动流程与组件关系

## 第三章：性能基准测试

### 3.1 基准测试方法论

- 推理服务基准测试的核心指标：TTFT（P50/P95/P99）、TPS（每请求）、吞吐量（系统总 token/s）、并发数
- 基准测试的变量：并发请求数、输入长度分布、输出长度分布、批处理策略
- 测试数据集的选择：ShareGPT（真实对话长度分布）、随机固定长度（控制变量）、自定义分布（模拟特定业务场景）

### 3.2 不同并发下的性能特征

- 逐步增加并发请求数（1、4、8、16、32...），记录各指标的变化
- 预期观察：低并发时吞吐量随并发线性增长，高并发时吞吐量增速放缓（显存瓶颈），延迟随并发逐渐增长
- 找到"甜点"并发数：吞吐量仍在快速增长但延迟增长尚可接受的区间

```python runnable gpu
# 模拟不同并发下的性能特征
import numpy as np

# 模拟参数（基于 vLLM 在 A100 上运行 7B 模型的典型数据）
model_params = "7B"
gpu = "A100 80GB"
single_request_tps = 40       # 单请求 TPS (token/s)
max_kv_cache_gb = 20          # 可用于 KV Cache 的显存
kv_per_request_gb = 0.8       # 单请求 KV Cache (4096 token, float16)
decode_time_per_step_ms = 12  # 单步 Decode 时间 (ms)

max_concurrency = int(max_kv_cache_gb / kv_per_request_gb)

print(f"模型: LLaMA-2 {model_params}, GPU: {gpu}")
print(f"最大并发数（显存约束）: {max_concurrency}")
print()

concurrency_levels = [1, 2, 4, 8, 16, 24, max_concurrency]
print(f"{'并发数':<8} | {'吞吐量(token/s)':<16} | {'单请求TPS':<12} | {'GPU利用率':<10} | {'延迟倍增'}")
print("-" * 65)

for c in concurrency_levels:
    if c > max_concurrency:
        break
    # 批处理效率：吞吐量增长亚线性
    batch_efficiency = min(0.95, 0.85 * np.log2(c + 1) / np.log2(max_concurrency + 1))
    throughput = single_request_tps * c * batch_efficiency
    per_request_tps = throughput / c
    gpu_util = min(95, throughput / (single_request_tps * max_concurrency * 0.9) * 100)
    latency_mult = single_request_tps / per_request_tps

    print(f"  {c:<6} | {throughput:>12.0f}     | {per_request_tps:>8.1f}   | {gpu_util:>7.1f}%  | {latency_mult:>6.1f}x")
```

### 3.3 Prefill 与 Decode 的延迟分解

- 分别测量 Prefill 和 Decode 的延迟，验证 Prefill 延迟与输入长度成正比、Decode 延迟与输出长度成正比
- 观察批量大小对 Prefill 和 Decode 延迟的不同影响：Prefill 的延迟增长较快（计算密集），Decode 的延迟增长较慢（访存密集）
- 当一个大的 Prefill 请求与多个 Decode 请求混合执行时的延迟干扰

### 3.4 量化对性能的影响

- 对比 float16 与 INT4/INT8 量化的推理性能：吞吐量、TTFT、TPS
- 量化对生成质量的影响：在标准基准上比较量化前后的输出一致性
- 量化带来的并发数提升：显存节省使得更多请求可以同时处理

> **可视化建议**：
> - 折线图：吞吐量和延迟随并发数变化的曲线
> - 柱状图：不同量化配置下的性能对比

## 第四章：流式输出与多轮对话

### 4.1 实现流式输出

- 使用 vLLM 的 OpenAI 兼容 API 获取流式响应
- SSE 协议的请求与响应格式
- 流式输出中的 token 级延迟测量：每个 token 从请求到到达客户端的时间

```python runnable gpu
# 演示流式生成（使用 Transformers 的 TextIteratorStreamer）
import torch
from threading import Thread

def simulate_streaming_generation(model, tokenizer, prompt, max_new_tokens=30):
    """模拟流式输出过程，逐 token 生成并打印"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 逐 token 生成
    generated_tokens = []
    current_text = ""

    torch.cuda.synchronize()
    start_time = time.time()
    first_token_time = None

    with torch.no_grad():
        # 逐 token 生成以模拟流式输出
        past_key_values = None
        current_ids = inputs["input_ids"]

        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(input_ids=current_ids, use_cache=True)
            else:
                outputs = model(input_ids=current_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_tokens.append(token_text)
            current_text += token_text
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if first_token_time is None:
                first_token_time = time.time()
                ttft = (first_token_time - start_time) * 1000

            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = (time.time() - start_time) * 1000
    decode_time = total_time - ttft
    output_len = len(generated_tokens)

    print(f"提示: {prompt}")
    print(f"TTFT: {ttft:.0f} ms")
    print(f"Decode 总时间: {decode_time:.0f} ms")
    print(f"输出长度: {output_len} tokens")
    print(f"TPS: {output_len / (decode_time / 1000):.1f} tokens/s")
    print(f"\n生成内容: {current_text}")

simulate_streaming_generation(model, tokenizer, "请用三句话介绍大语言模型的推理过程")
```

### 4.2 多轮对话的 KV Cache 复用

- 多轮对话场景下，前几轮的 KV Cache 可以直接复用，只需对新增的输入做 Prefill
- 测量有无 KV Cache 复用时各轮对话的 TTFT 差异
- 观察 KV Cache 复用带来的显存节省：多轮对话的显存增长仅由新 token 贡献

### 4.3 前缀缓存实验

- 构造多个共享 system prompt 的请求，观察前缀缓存开启/关闭时的 Prefill 时间差异
- 测量前缀缓存的命中率：请求中与前缀匹配的 token 比例
- 前缀缓存对 TTFT 的优化效果随 system prompt 长度增长而愈加显著

## 第五章：调度策略调优实验

### 5.1 批处理策略对比

- 对比静态批处理与连续批处理的性能差异
- 构造混合长度的请求负载（短请求 50-100 token + 长请求 500-1000 token），测量两种批处理策略下短请求的延迟
- 预期观察：静态批处理下短请求被长请求拖慢（延迟膨胀），连续批处理下短请求可以提前完成

### 5.2 显存管理策略调优

- 调整 vLLM 的 `--gpu-memory-utilization` 参数（0.8、0.9、0.95），观察对并发数和 OOM 风险的影响
- 对比 Swap 和 Recomputation 抢占策略的性能差异
- 构造超量并发场景（并发数超过显存容纳上限），观察抢占行为

```python runnable gpu
# 模拟不同 gpu-memory-utilization 设置下的并发容量
import numpy as np

model_weight_gb = 14     # 7B float16 模型权重约 14 GB
gpu_total_gb = 80        # A100 80GB
runtime_overhead_gb = 3  # 运行时临时缓冲区
kv_per_request_gb = 0.8  # 单请求 KV Cache

utilizations = [0.80, 0.85, 0.90, 0.95, 0.98]

print(f"模型: 7B (float16), GPU: A100 80GB")
print(f"模型权重: {model_weight_gb} GB")
print()
print(f"{'利用率':<8} | {'可用显存(GB)':<14} | {'KV Cache空间(GB)':<16} | {'最大并发':<8} | {'OOM风险'}")
print("-" * 70)

for util in utilizations:
    available = gpu_total_gb * util
    kv_space = available - model_weight_gb - runtime_overhead_gb
    max_conc = int(kv_space / kv_per_request_gb)
    oom_risk = "低" if util < 0.9 else ("中" if util < 0.95 else "高")

    print(f"  {util:.0%}   | {available:>10.1f}    | {kv_space:>12.1f}     | {max_conc:>6}   | {oom_risk}")
```

### 5.3 请求优先级与调度实验

- 设计一个优先级调度实验：同时发送高优先级和低优先级请求，观察高优先级请求的延迟是否优于低优先级
- 测试 FCFS 与优先级调度在混合负载下的延迟分布差异
- 构造极端场景：大量低优先级长请求占据 GPU 后，高优先级短请求的等待时间

## 第六章：综合实验——从模型到生产级推理服务

### 6.1 设计目标

- 部署一个支持多用户并发的 LLM 推理服务
- SLO 目标：TTFT P99 < 1s，TPS > 20 token/s，支持 10 并发
- 在限定硬件资源下（单块 GPU）达到最优的延迟-吞吐量权衡

### 6.2 部署方案设计

- 模型选择与量化策略：根据硬件资源和性能目标选择合适的模型与精度
- vLLM 参数调优：`--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs`（最大并发序列数）
- API 网关配置：限流策略、超时设置、认证鉴权

### 6.3 负载测试与 SLO 验证

- 使用渐进式负载测试：从 1 并发逐步增加到目标并发，记录各指标
- SLO 达标验证：各指标是否满足设计目标
- 瓶颈分析：如果 SLO 未达标，定位瓶颈是显存、算力、还是调度策略

### 6.4 调优迭代

- 根据瓶颈分析结果调整部署方案：量化降低显存压力、调整最大序列长度、优化批处理策略
- 验证调优效果：重新运行负载测试，对比调优前后的性能指标
- 最终方案确认：达到 SLO 的最优配置

> **可视化建议**：
> - 调优前后的性能对比图
> - 完整的部署架构图（包含 API 网关、vLLM 服务、GPU 资源）

## 本章小结

- 本次实训覆盖了 LLM 推理服务部署的完整流程：环境搭建 → 模型加载 → 性能测试 → 流式输出 → 调度调优 → 综合部署
- 核心性能指标（TTFT、TPS、吞吐量、并发数）的测量与分析是推理服务优化的基础
- 量化、批处理策略、显存管理是性能调优的三大杠杆
- 从模型到生产级推理服务，需要经过多轮负载测试与调优迭代，每轮聚焦一个瓶颈点

## 练习题

1. 在本次实训中，如果你需要将 7B 模型的推理服务部署到一块 RTX 3090（24GB 显存）上，请设计完整的部署方案，包括量化策略、最大序列长度、预期并发数，并说明你的设计理由。

   <details>
   <summary>参考答案</summary>

   显存预算分析：7B float16 模型权重约 14 GB，运行时约 2 GB，剩余约 8 GB。单请求 KV Cache（4096 token）约 0.8 GB，最多支持约 10 并发。但如果使用 INT4 量化，权重降至 3.5 GB，剩余约 18.5 GB，配合 KV Cache FP8（0.4 GB/请求），可支持约 46 并发。

   推荐方案：INT4 权重量化 + KV Cache FP8 量化，最大序列长度 2048（限制序列长度以降低单请求 KV Cache），预期并发 20-30 个请求。

   设计理由：RTX 3090 的显存有限（24 GB vs A100 的 80 GB），必须通过量化压缩模型权重和 KV Cache。限制最大序列长度到 2048 是因为对话场景中大部分请求的输入+输出不超过 2048 token，这样可以大幅提升并发能力。如果需要支持更长的序列，可以适当降低并发数。

   </details>

2. 设计一个实验方案，验证连续批处理相对于静态批处理的优势。请说明：(1) 测试负载的构造方法（请求长度分布）；(2) 对比的指标；(3) 预期的实验结果。

   <details>
   <summary>参考答案</summary>

   (1) 测试负载构造：使用混合长度分布，80% 的请求生成 50-100 token（短请求），20% 的请求生成 500-1000 token（长请求）。并发数设置为 GPU 可容纳最大并发的 50%，确保有足够的请求填充批量。这种分布模拟了真实场景中大部分对话请求较短、少数请求较长的特点。

   (2) 对比指标：短请求的 P50/P99 延迟、系统总吞吐量、GPU 算力利用率（观察静态批处理中短请求完成后的空闲时间）。

   (3) 预期结果：静态批处理下，短请求的 P99 延迟远高于连续批处理（因为短请求必须等待同一批量中最长的请求完成）。连续批处理的吞吐量更高（短请求完成后立即让出位置给新请求，GPU 持续满载）。GPU 算力利用率方面，静态批处理在短请求完成后出现明显的空闲周期，而连续批处理保持稳定的高利用率。P99 延迟的差距最为显著——静态批处理下短请求的 P99 延迟可能达到连续批处理的 5-10 倍。

   </details>

3. 某推理服务在运行过程中出现间歇性的延迟飙升（P99 延迟从 500ms 突增到 5s），但 GPU 利用率始终在 60% 左右。请根据本章和前三章的知识，分析可能的原因并给出排查思路。

   <details>
   <summary>参考答案</summary>

   GPU 利用率 60% 且延迟飙升，说明 GPU 并非持续满载，问题可能出现在以下方面：

   (1) **Prefill 长请求阻塞 Decode**：一个长输入的 Prefill 请求可能暂时占用大量 GPU 算力，导致正在进行的 Decode 请求被暂停。排查方法：检查延迟飙升时是否有特别长的输入请求，对比 Prefill 请求的输入长度与延迟飙升的时间是否吻合。解决方案：PD 分离，或限制单次 Prefill 的最大 chunk 大小（Chunked Prefill）。

   (2) **KV Cache 碎片化**：虽然 vLLM 使用 PagedAttention 避免了碎片，但如果 Block 分配器在高并发时来不及回收已完成的 Block，新的请求可能需要等待 Block 回收后才能进入批量。排查方法：监控 Block 分配器的等待队列长度。解决方案：调整 Block 大小或预分配更多 Block。

   (3) **抢占与 Swap 开销**：当显存不足时触发抢占，被抢占请求的 KV Cache 通过 Swap 到 CPU 内存，恢复时的 PCIe 传输导致延迟飙升。排查方法：检查抢占事件频率和 Swap 操作的时间。解决方案：降低并发上限或增加 KV Cache 量化以减少显存压力。

   (4) **调度器开销**：连续批处理的 iteration-level 调度在极端情况下（大量请求同时完成或到达）可能导致调度延迟。排查方法：测量调度器的决策时间。解决方案：限制每次调度的请求数量或降低调度频率。

   排查思路优先级：先检查 Prefill 干扰（最常见），再检查抢占/Swap（次常见），最后检查调度器和 Block 分配器。

   </details>
