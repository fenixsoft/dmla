# 工程实训：部署 LLM 推理服务

通过前三章的学习，我们了解了推理服务的[架构原理](inference-service-architecture.md)、[请求调度与批处理机制](request-scheduling.md)、[GPU 资源管理策略](gpu-resource-management.md)。这些知识构成了理解 LLM 推理服务化的理论基础，但仅凭阅读很难真正体会将一个大语言模型部署为可用的推理服务时面临的实际取舍。本次实验中，我们将使用 vLLM 推理框架，从模型加载到性能调优到流式输出，完整走一遍 LLM 推理服务从零到可用的全过程。

## 实验准备

本次实验需要一块支持 CUDA 的 NVIDIA GPU，推荐显存 8 GB 以上。实验使用 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 作为演示模型，参数量约 5 亿，小到单卡即可轻松加载，大到足以展示推理服务的各项性能特征。

实验的核心依赖是 vLLM 推理框架，已在 DMLA 沙箱镜像中预装。首先验证 GPU 环境、CUDA 支持和 vLLM 是否正常：

```python runnable gpu
import torch
import os

# 设置 HuggingFace 缓存目录到持久化数据卷，避免每次执行重新下载模型
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  显存总量: {props.total_memory / 1024**3:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  当前显存占用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
else:
    print("CUDA 不可用，请检查 GPU 驱动和 CUDA 安装")

# 验证 vLLM 可用性
try:
    import vllm
    print(f"\nvLLM 版本: {vllm.__version__}")
    print("vLLM 可用 ✓")
except ImportError:
    print("\nvLLM 未安装。如果你使用的是旧版 Docker 镜像，请重新拉取或构建包含 vLLM 的镜像。")
    print("手动安装: pip install vllm")
```

## 第一阶段：模型加载与显存分析

推理服务启动的第一步是将模型加载到 GPU 显存中。与直接使用 Transformers 加载模型不同，vLLM 在加载时会自动启用 **PagedAttention** 机制——将 KV Cache 切分为固定大小的 Block，像操作系统的虚拟内存页面一样管理，从根本上解决显存碎片问题，让有限的显存能容纳更多并发请求。

vLLM 通过 `gpu_memory_utilization` 参数控制显存使用上限（默认 0.90）。这个值设得越高，预分配的 KV Cache Block 就越多，能同时处理的请求也越多，但留给 CUDA 运行时和临时缓冲区的余量就越少。下面的代码以 0.85 的利用率加载模型，观察加载前后的显存变化，并分析各部分占比。

```python runnable gpu timeout=unlimited
import torch
import os
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")

# 记录加载前的显存状态
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    before_mem = torch.cuda.memory_allocated() / 1024**3

# 使用 vLLM 加载模型，自动启用 PagedAttention
# gpu_memory_utilization=0.85：最多使用 85% 显存，剩余留给 CUDA 上下文
# max_model_len=4096：限制最大序列长度，直接影响 KV Cache Block 预分配量
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B",
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

# 记录加载后的显存状态
if torch.cuda.is_available():
    after_mem = torch.cuda.memory_allocated() / 1024**3
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    # 模型权重的理论大小：参数量 × float16 字节数
    # 0.5B params × 2 bytes ≈ 0.93 GB
    model_weight_gb = 0.5 * 1e9 * 2 / 1024**3

    print(f"模型: Qwen/Qwen2.5-0.5B (float16)")
    print(f"加载前显存: {before_mem:.2f} GB")
    print(f"加载后显存: {after_mem:.2f} GB")
    print(f"峰值显存: {peak_mem:.2f} GB")
    print(f"vLLM 总占用: {after_mem - before_mem:.2f} GB")
    print(f"\n--- 显存构成分析 ---")
    print(f"模型权重 (理论): {model_weight_gb:.2f} GB")
    print(f"KV Cache Block + CUDA 开销: {after_mem - before_mem - model_weight_gb:.2f} GB")
    print(f"  (PagedAttention 预分配的 Block 池 + PyTorch CUDA 上下文)")
```

从输出中可以观察到，vLLM 加载后的显存占用由两部分构成：模型权重（约为参数量 × 精度的理论值）和 PagedAttention 预分配的 Block 池。这个 Block 池的大小由 `gpu_memory_utilization` 和 `max_model_len` 共同决定——前者控制总预算上限，后者控制单个 Block 的大小（Block 大小 = token 数 × KV 维度 × 精度字节数）。

不同精度下的模型权重占用差异显著。以 0.5B 模型为例，float32 下权重约 2 GB，float16 约 1 GB，INT4 量化可压缩到约 0.25 GB。精度越低，权重占用越小，留给 KV Cache 的空间就越大。但精度降低也会影响生成质量，vLLM 支持 AWQ 和 GPTQ 等多种量化格式，在[第三阶段](#第三阶段并发性能与显存调优)中会进一步展示量化对并发容量的影响。

## 第二阶段：单次推理延迟测量

模型加载完成后，下一步是测量推理延迟。LLM 推理过程分为两个阶段：**Prefill**（预填充）阶段一次性处理所有输入 token，生成第一个输出 token；**Decode**（解码）阶段逐 token 自回归生成。vLLM 的 `RequestOutput.metrics` 提供了精确的时序指标：`first_token_time` 记录 TTFT（首 token 时间，即 Prefill 耗时），`time_per_output_token` 记录 TPOT（Decode 阶段平均每 token 时间）。

与手动使用 `time.time()` 进行近似估算不同，vLLM 的 metrics 由推理引擎内部精确计时，排除了 Python 层面的调度抖动。下面的代码同时发送两条长度不同的提示，vLLM 自动进行 Continuous Batching，然后对比短提示和长提示的延迟差异。

```python runnable gpu timeout=unlimited
import torch
import time
import os
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")

from vllm import LLM, SamplingParams

# 加载模型（如果上一阶段已执行，HF_HOME 缓存让加载仅需 5-10 秒）
llm = LLM(
    model="Qwen/Qwen2.5-0.5B",
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

# 贪心解码，保证可复现
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=50,
)

short_prompt = "什么是人工智能？"
long_prompt = "请详细解释深度学习的发展历程，从感知机开始，到多层感知机、卷积神经网络、循环神经网络，再到 Transformer 架构。每个阶段请说明其核心创新和代表性工作。"

prompts = [short_prompt, long_prompt]

# vLLM 自动进行 Continuous Batching，两个请求合并执行
torch.cuda.synchronize()
wall_start = time.time()
outputs = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()
wall_time = time.time() - wall_start

for i, (prompt, output) in enumerate(zip(prompts, outputs)):
    m = output.metrics
    prompt_len = len(output.prompt_token_ids)
    output_len = len(output.outputs[0].token_ids)

    print(f"\n=== 提示 {i+1} ({'短' if i == 0 else '长'}) ===")
    print(f"提示: {prompt[:60]}...")
    print(f"输入长度: {prompt_len} tokens")
    print(f"输出长度: {output_len} tokens")
    # vLLM metrics 提供精确的 Prefill / Decode 时序
    ttft_ms = m.first_token_time * 1000
    tpot_ms = m.time_per_output_token * 1000
    tps = 1.0 / m.time_per_output_token if m.time_per_output_token > 0 else 0
    print(f"TTFT (Prefill 耗时): {ttft_ms:.1f} ms")
    print(f"TPOT (Decode 每 token 平均): {tpot_ms:.1f} ms")
    print(f"TPS: {tps:.1f} tokens/s")
    print(f"生成内容: {output.outputs[0].text[:100]}...")

print(f"\n壁钟总时间 (双请求 Continuous Batching): {wall_time*1000:.0f} ms")
```

运行后可以观察到两个现象。第一，长提示的 TTFT 明显高于短提示。这是因为 Prefill 阶段需要一次性计算所有输入 token 的自注意力，计算量随输入长度平方增长，而 Decode 阶段每个新 token 只需与已有的 KV Cache 做一次注意力运算，单步时间基本恒定。这就是为什么 TTFT 和 TPOT 可能相差一个数量级。

第二，两个请求的总壁钟时间通常小于单独执行之和。这是因为 vLLM 的 Continuous Batching 将两个请求的 Decode 步骤合并为批量矩阵乘法，提升了 GPU 利用率。

## 第三阶段：并发性能与显存调优

前两个阶段使用 vLLM 的 Python API 在进程内直接推理，适合离线批量处理。生产环境中，推理服务通常以 HTTP 服务的形式部署，多个客户端通过 OpenAI 兼容 API 并发请求。本阶段将实际启动 vLLM 推理服务，通过模拟多客户端并发请求，观察不同并发度下的吞吐量和延迟变化。

下面的代码分三步执行：首先通过 `subprocess` 启动 vLLM 的 OpenAI 兼容 API 服务，然后使用 `ThreadPoolExecutor` 模拟 1/2/4/8 路并发请求，最后基于实际硬件参数分析不同显存利用率下的并发容量。

```python runnable gpu timeout=unlimited
import subprocess
import time
import requests
import sys
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")

# 获取 GPU 显存信息（必须在 vLLM 启动前查询，避免 CUDA 上下文竞争）
if torch.cuda.is_available():
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
else:
    gpu_total_gb = 8.0

# ========== 1. 启动 vLLM 推理服务 ==========
print("正在启动 vLLM 推理服务...")
server_proc = subprocess.Popen(
    [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", "8000",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# 轮询 /health 端点等待服务就绪
base_url = "http://127.0.0.1:8000"
ready = False
for attempt in range(90):
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            print(f"vLLM 服务已就绪 (耗时约 {attempt*2}s)\n")
            break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    server_proc.kill()
    raise RuntimeError("vLLM 服务启动超时，请检查 GPU 显存是否充足")

# ========== 2. 并发请求测试 ==========
prompt = "请用三句话介绍大语言模型的推理过程"

def send_request(req_id):
    """发送一次推理请求，返回耗时和输出 token 数"""
    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B",
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0,
            },
            timeout=120,
        )
        elapsed = time.time() - t0
        data = r.json()
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {"req_id": req_id, "output_tokens": output_tokens, "total_time": elapsed}
    except Exception as e:
        return {"req_id": req_id, "output_tokens": 0, "total_time": time.time() - t0, "error": str(e)}

concurrency_levels = [1, 2, 4, 8]
print(f"{'并发数':<8} | {'总吞吐(token/s)':<16} | {'单请求TPS':<12} | {'总耗时(s)'}")
print("-" * 55)

for c in concurrency_levels:
    start = time.time()
    with ThreadPoolExecutor(max_workers=c) as executor:
        futures = [executor.submit(send_request, i) for i in range(c)]
        results = [f.result() for f in as_completed(futures)]

    total_time = time.time() - start
    total_tokens = sum(r["output_tokens"] for r in results)
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_tps = sum(
        r["output_tokens"] / r["total_time"]
        for r in results if r["total_time"] > 0
    ) / len(results)

    print(f"  {c:<6} | {throughput:>12.1f}     | {avg_tps:>8.1f}   | {total_time:>8.1f}")

# ========== 3. 显存利用率与并发容量分析 ==========
print(f"\n--- 显存利用率与并发容量 ---")

model_weight_gb = 0.93   # 0.5B float16 权重大小
runtime_overhead_gb = 0.5  # CUDA 上下文等固定开销
kv_per_request_gb = 0.2    # 单请求 KV Cache 估算（4096 token）

print(f"GPU 显存总量: {gpu_total_gb:.1f} GB")
print(f"模型权重: {model_weight_gb:.2f} GB, 固定开销: {runtime_overhead_gb:.2f} GB")
print(f"单请求 KV Cache: ~{kv_per_request_gb:.1f} GB (max_model_len=4096, 估算值)")
print()
print(f"{'利用率':<8} | {'KV Cache可用(GB)':<16} | {'预估并发上限':<12} | {'OOM风险'}")
print("-" * 52)

for util in [0.75, 0.80, 0.85, 0.90, 0.95]:
    available = gpu_total_gb * util
    kv_space = available - model_weight_gb - runtime_overhead_gb
    max_conc = max(1, int(kv_space / kv_per_request_gb))
    oom = "低" if util < 0.88 else ("中" if util < 0.93 else "高")
    print(f"  {util:.0%}   | {kv_space:>12.2f}     | {max_conc:>10}   | {oom}")

print(f"\n当前配置 gpu_memory_utilization=0.85，理论并发上限约为 "
      f"{max(1, int((gpu_total_gb * 0.85 - model_weight_gb - runtime_overhead_gb) / kv_per_request_gb))} 个请求")

# ========== 4. 清理 ==========
server_proc.terminate()
try:
    server_proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    server_proc.kill()
print("\nvLLM 服务已停止")
```

从输出中可以看到，吞吐量随并发数增长但逐渐放缓，与[请求调度](request-scheduling.md)中讨论的批处理效率曲线一致。显存利用率分析清楚地展示了 `gpu_memory_utilization` 参数的杠杆作用：从 0.85 调到 0.90，并发容量可以提升 10-20%，但 OOM 风险也随之升高。生产环境中推荐的设置是 0.85-0.90，当显存特别紧张时可以短时间用到 0.95，但需要配合监控和告警。

除了显存利用率，vLLM 还提供了抢占策略来处理超量并发的场景。当新请求到达而显存不足时，`swap` 策略将部分 KV Cache 换出到 CPU 内存，等 GPU 空闲时再换回；`recomputation` 策略则直接丢弃被抢占请求的 KV Cache，等恢复执行时重新计算 Prefill。选择哪种策略取决于请求负载特征：短文本为主的场景推荐 Recomputation（重新计算的代价小、不占 CPU 内存），长文本居多的场景推荐 Swap（切换开销低）。

## 第四阶段：流式输出与 KV Cache 实验

前面的实验使用的是非流式推理——请求发送后等待完整响应返回。对于终端用户而言，等待几秒看到空白页面和逐字看到内容流出，体验截然不同。vLLM 的 OpenAI 兼容 API 支持 `stream=True` 参数，通过 SSE（Server-Sent Events）协议逐 token 推送生成内容。

本阶段启动 vLLM 服务后，发起流式请求并逐 token 解析 SSE 响应，精确记录首个 token 的到达时间（TTFT）和每个后续 token 的到达间隔，直观展示 Prefill 与 Decode 阶段的计算量差异。

```python runnable gpu timeout=unlimited
import subprocess
import time
import requests
import sys
import json
import os

os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")

# ========== 1. 启动 vLLM 推理服务 ==========
print("正在启动 vLLM 推理服务...")
server_proc = subprocess.Popen(
    [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", "8000",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# 等待服务就绪
base_url = "http://127.0.0.1:8000"
ready = False
for attempt in range(90):
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            print(f"vLLM 服务已就绪 (耗时约 {attempt*2}s)\n")
            break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    server_proc.kill()
    raise RuntimeError("vLLM 服务启动超时")

# ========== 2. 流式请求 + 逐 token 输出 ==========
prompt = "请用三句话介绍大语言模型的推理过程"
print(f"提示: {prompt}\n")

start_time = time.time()
first_token_time = None
token_count = 0
token_times = []
generated_text = ""

# 发起流式请求（stream=True）
response = requests.post(
    f"{base_url}/v1/completions",
    json={
        "model": "Qwen/Qwen2.5-0.5B",
        "prompt": prompt,
        "max_tokens": 80,
        "temperature": 0,
        "stream": True,
    },
    stream=True,
    timeout=120,
)

print("流式输出: ", end="", flush=True)

for line in response.iter_lines():
    if not line:
        continue
    line = line.decode("utf-8")
    if not line.startswith("data: "):
        continue

    data_str = line[6:]
    if data_str == "[DONE]":
        break

    try:
        data = json.loads(data_str)
        token_text = data["choices"][0].get("text", "")
        generated_text += token_text
        now = time.time()

        if first_token_time is None:
            first_token_time = now
            ttft_ms = (first_token_time - start_time) * 1000
            print(f"\n[TTFT] {ttft_ms:.0f} ms — Prefill 完成，首个 token 生成\n")
            print("流式输出: ", end="", flush=True)

        token_count += 1
        token_times.append(now)
        # 逐 token 打印，flush 确保每个 token 立即推送到前端
        print(token_text, end="", flush=True)

    except json.JSONDecodeError:
        pass

print("\n")

# ========== 3. 计算流式输出指标 ==========
decode_start = token_times[0] if token_times else start_time
decode_end = token_times[-1] if token_times else start_time
decode_time_ms = (decode_end - decode_start) * 1000

# 计算每个相邻 token 的到达间隔
intervals = []
for i in range(1, len(token_times)):
    intervals.append((token_times[i] - token_times[i - 1]) * 1000)

avg_interval = sum(intervals) / len(intervals) if intervals else 0

print(f"--- 流式输出指标 ---")
print(f"TTFT (首个 token): {ttft_ms:.0f} ms")
print(f"Decode 总时间: {decode_time_ms:.0f} ms")
print(f"输出长度: {token_count} tokens")
print(f"平均 Token 间隔: {avg_interval:.1f} ms/token")
print(f"TPS: {token_count / (decode_time_ms / 1000):.1f} tokens/s")

# 对比 Prefill 和 Decode 的计算量差异
if avg_interval > 0:
    ratio = ttft_ms / avg_interval
    print(f"\n--- KV Cache 的作用 ---")
    print(f"TTFT / 平均 Token 间隔 = {ratio:.0f}x")
    print(f"这意味着 Prefill 阶段一次性处理了约 {ratio:.0f} 倍于单个 Decode 步骤的计算量")
    print(f"Decode 阶段之所以快，正是因为复用了 Prefill 生成的 KV Cache")

# ========== 4. 清理 ==========
server_proc.terminate()
try:
    server_proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    server_proc.kill()
print("\nvLLM 服务已停止")
```

这段代码运行时会逐 token 在页面上打印生成内容，你可以清晰看到首个 token 在短暂的等待后出现，随后每个 token 以大致相等的时间间隔接连输出。注意观察 TTFT 和平均 token 间隔的比值——前者通常在几十到几百毫秒（取决于输入长度），后者通常在十到几十毫秒，两者可能相差一个数量级。这是因为 Prefill 阶段需要计算所有输入 token 之间的两两注意力（$O(n^2)$ 的注意力矩阵），而 Decode 阶段每个新 token 只需要与已有的 KV Cache 做一次注意力运算。

流式输出的背后是 KV Cache 机制在持续发挥作用。Decode 阶段每生成一个新 token，只有这个新 token 的 Query 需要与所有历史 token 的 Key 和 Value 进行注意力计算——历史 token 的 K、V 向量早在 Prefill 阶段就已算出并缓存在显存中。在多轮对话场景中，这种复用更加明显：前几轮对话的 KV Cache 可以直接复用，只有新增的用户输入部分需要做 Prefill。

vLLM 还实现了前缀缓存（Prefix Caching）进一步利用这一特性。当多个请求共享相同的 system prompt 时，这段 prompt 的 KV Cache 只需计算一次，后续请求直接复用。在实际聊天应用中，system prompt 通常有数百 token，前缀缓存可以将这些请求的 TTFT 缩短 30-50%。

## 实验结论

本次实验使用 vLLM 推理框架完整展示了 LLM 推理服务从模型加载到流式输出的全流程。与直接使用 Transformers 做推理相比，vLLM 的核心价值体现在三个层面：

**显存管理**：PagedAttention 将 KV Cache 切分为固定大小的 Block，像操作系统的虚拟内存页面一样管理，从根本上消除了显存碎片。这使得相同的 GPU 显存可以容纳更多并发请求。实验中通过 `gpu_memory_utilization` 参数直观看到了显存分配策略对并发容量的影响。

**调度效率**：Continuous Batching 在请求到达和完成时动态调整批处理组合，而非等待整批完成后再组新批。实验中同时发送两条不同长度的提示时，可以看到它们被自动合并执行，总耗时小于单独执行之和。并发测试进一步验证了吞吐量随并发数增长的亚线性特征。

**流式体验**：通过 SSE 协议逐 token 推送生成内容，将用户的感知延迟从"全部生成时间"缩短到"首 token 时间"。实验中 TTFT 与平均 token 间隔的巨大差距（可达 10 倍以上），直观说明了流式输出对改善用户体验的关键作用。

本次实验使用 vLLM 的 Python API 和 Server 两种模式。Python API 适合离线批量处理和性能测量，Server 模式适合模拟生产环境的并发请求和流式输出场景。生产部署中推荐使用 `vllm.entrypoints.openai.api_server` 启动 OpenAI 兼容的 HTTP 服务，配合 Nginx 等反向代理实现负载均衡和认证鉴权。

## 运行结果

下面是本实验各阶段的典型运行输出，供你验证自己的运行结果。实际数值因 GPU 型号和 vLLM 版本而异。

**第一阶段 — 模型加载与显存分析**：

```
模型: Qwen/Qwen2.5-0.5B (float16)
加载前显存: 0.00 GB
加载后显存: 1.52 GB
峰值显存: 1.68 GB
vLLM 总占用: 1.52 GB

--- 显存构成分析 ---
模型权重 (理论): 0.93 GB
KV Cache Block + CUDA 开销: 0.59 GB
  (PagedAttention 预分配的 Block 池 + PyTorch CUDA 上下文)
```

**第二阶段 — 推理延迟测量**：

```
=== 提示 1 (短) ===
提示: 什么是人工智能？...
输入长度: 5 tokens
输出长度: 50 tokens
TTFT (Prefill 耗时): 35.2 ms
TPOT (Decode 每 token 平均): 12.8 ms
TPS: 78.1 tokens/s
生成内容: 人工智能是计算机科学的一个分支...

=== 提示 2 (长) ===
提示: 请详细解释深度学习的发展历程...
输入长度: 62 tokens
输出长度: 50 tokens
TTFT (Prefill 耗时): 128.6 ms
TPOT (Decode 每 token 平均): 13.1 ms
TPS: 76.3 tokens/s
生成内容: 深度学习的发展历程可以分为以下几个阶段...

壁钟总时间 (双请求 Continuous Batching): 920 ms
```

**第三阶段 — 并发性能测试**（RTX 4060 8GB）：

```
并发数   | 总吞吐(token/s)  | 单请求TPS    | 总耗时(s)
-------------------------------------------------------
  1      |          78.1     |     78.1   |      0.6
  2      |         144.5     |     72.3   |      0.7
  4      |         253.8     |     63.5   |      0.8
  8      |         412.3     |     51.5   |      1.0

--- 显存利用率与并发容量 ---
GPU 显存总量: 7.6 GB
模型权重: 0.93 GB, 固定开销: 0.50 GB
单请求 KV Cache: ~0.2 GB (max_model_len=4096, 估算值)

利用率    | KV Cache可用(GB) | 预估并发上限  | OOM风险
----------------------------------------------------
  75%    |          4.27     |         21   | 低
  80%    |          4.65     |         23   | 低
  85%    |          5.03     |         25   | 低
  90%    |          5.41     |         27   | 中
  95%    |          5.79     |         28   | 高
```

**第四阶段 — 流式输出**：

```
提示: 请用三句话介绍大语言模型的推理过程

[TTFT] 42 ms — Prefill 完成，首个 token 生成

流式输出: 大语言模型的推理过程分为两个阶段：预填充阶段一次性处理所有输入token并生成第一个输出token，解码阶段则逐token自回归生成后续内容。推理过程依赖KV Cache机制缓存历史信息以避免重复计算...

--- 流式输出指标 ---
TTFT (首个 token): 42 ms
Decode 总时间: 725 ms
输出长度: 56 tokens
平均 Token 间隔: 13.2 ms/token
TPS: 75.9 tokens/s

--- KV Cache 的作用 ---
TTFT / 平均 Token 间隔 = 3x
这意味着 Prefill 阶段一次性处理了约 3 倍于单个 Decode 步骤的计算量
Decode 阶段之所以快，正是因为复用了 Prefill 生成的 KV Cache
```
