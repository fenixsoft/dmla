# 工程实训：部署 LLM 推理服务

通过前三章的学习，我们了解了推理服务的[架构原理](inference-service-architecture.md)、[请求调度与批处理机制](request-scheduling.md)、[GPU 资源管理策略](gpu-resource-management.md)。这些知识构成了理解 LLM 推理服务化的理论基础。本次实验中，我们将使用 vLLM 推理框架，从模型加载到性能调优到流式输出，完整走一遍 LLM 推理服务从零到可用的全过程。

## 实验准备

本次实验继续沿用 [LLM 推理策略与效率优化实验](../../language-models/reasoning/llm-reasoning-experiment.md)中引入的 [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) 作为演示模型，单卡即可轻松加载，也足以展示推理服务的各项性能特征。如果之前未曾完成该实验，需先下载 Qwen3.5-0.8B-Instruct 模型。可以通过 `DMLA-CLI` 工具自动完成该工作：

```bash
# 下载模型：选择 "下载模型" -> 选择 "Qwen3.5-0.8B-Instruct"
dmla model
```
实验的核心依赖是 vLLM 推理框架，已在 DMLA 沙箱镜像中预装。可通过以下代码验证 GPU 环境、CUDA 支持和 vLLM 是否正常：

```python runnable gpuonly
import torch
import os

# 验证 Qwen3.5-0.8B-Instruct 模型是否正确下载
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
model_ready = False
if os.path.exists(model_path):
    # 检查 LFS 不完整标记文件
    incomplete_marker = os.path.join(model_path, '.lfs-incomplete')
    if os.path.exists(incomplete_marker):
        print(f"模型目录存在但数据不完整（LFS 未拉取）: {model_path}")
        print(f"请确保已安装 Git LFS，然后运行: cd {model_path} && git lfs pull")
    else:
        # 检查关键文件是否存在
        required_files = ['config.json', 'tokenizer_config.json']
        missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        if missing:
            print(f"模型目录存在但缺少关键文件: {missing}")
            print(f"模型路径: {model_path}")
            print(f"请运行 'dmla model' 重新下载模型")
        else:
            model_ready = True
            print(f"模型就绪: Qwen3.5-0.8B-Instruct")
else:
    print(f"模型未下载: {model_path}")
    print("请运行 'dmla model' 下载 Qwen3.5-0.8B-Instruct 模型")
    print("  1. 执行 dmla model 进入 TUI")
    print("  2. 选择 '下载模型' -> 选择 'Qwen3.5-0.8B-Instruct'")
print()

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

## 模型加载与显存分析

推理服务启动的第一步是将模型加载到 GPU 显存中。与以往实验中直接使用 Transformers 框架加载模型不同，本次实验使用 vLLM 推理框架进行加载。vLLM 自动启用 PagedAttention 机制，将 KV Cache 切分为固定大小的 Block，像操作系统的虚拟内存页面一样管理，让有限的显存能容纳更多并发请求。

vLLM 通过 `gpu_memory_utilization` 参数控制显存使用上限（默认 0.90）。这个值设得越高，预分配的 KV Cache Block 就越多，能同时处理的请求也越多，但留给 CUDA 运行时和临时缓冲区的余量就越少。下面的代码以 0.85 的利用率加载模型，观察加载前后的显存变化，并分析各部分占比。

```python runnable gpuonly timeout=unlimited
import torch
import os

# 记录加载前的显存状态
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    before_mem = torch.cuda.memory_allocated() / 1024**3

# 使用本地下载的 Qwen3.5-0.8B-Instruct 模型
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

# 抑制 vLLM 引擎核心配置 dump（输出量过大时会造成 Kernel 输出解析失败）
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 使用 vLLM 加载模型，自动启用 PagedAttention
# gpu_memory_utilization=0.85：最多使用 85% 显存，剩余留给 CUDA 上下文
# max_model_len=4096：限制最大序列长度，直接影响 KV Cache Block 预分配量
from vllm import LLM

llm = LLM(
    model=model_path,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

# vLLM 使用多进程引擎核心，模型加载在子进程中，
# 主进程 torch.cuda.memory_allocated() 无法反映实际占用。
# 实际显存数据见上方 INFO 日志中的 "Model loading took" 和 "GPU KV cache size" 行。
# 下面基于理论值做显存构成分析（验证实际用时请参照控制台日志）：
if torch.cuda.is_available():
    # 模型权重的理论大小：参数量 × float16 字节数
    # 0.8B params × 2 bytes ≈ 1.6 GB
    model_weight_gb = 0.8 * 1e9 * 2 / 1024**3

    print(f"\n模型: Qwen3.5-0.8B-Instruct (float16)")
    print(f"\n--- 显存构成分析 ---")
    print(f"模型权重 (理论): {model_weight_gb:.2f} GB")
    print(f"KV Cache Block + CUDA 开销: 请查看上方 'Model loading took' 和 'GPU KV cache size' 日志")
    print(f"  (PagedAttention 预分配的 Block 池 + PyTorch CUDA 上下文)")
```

vLLM 引擎核心在子进程中运行，`torch.cuda.memory_allocated()` 无法反映实际占用。以下是 vLLM 初始化日志中的关键数据：

```
模型: Qwen3.5-0.8B-Instruct (float16)

--- 显存构成分析 ---
模型权重 (理论): 1.6 GB

实际日志输出:
  Model loading took 1.72 GiB memory and 0.72 seconds
  GPU KV cache size: 183,202 tokens
  Available KV cache memory: 3.06 GiB
  CUDA graph pool memory: 0.42 GiB
  Maximum concurrency for 4,096 tokens per request: 44.73x
```

从 vLLM 的初始化日志中可以观察到，显存占用由三部分构成：
- 模型权重（实测约 1.72 GiB，接近理论值 1.6 GB）
- PagedAttention 预分配的 KV Cache Block 池（3.06 GiB 可用，共 183,202 tokens）
- CUDA Graph 编译产生的固定缓存（0.42 GiB）

不同精度下的模型权重占用差异显著。以 0.8B 模型为例，FP32 下权重约 3.2 GB，FP16 约 1.6 GB，INT4 量化可压缩到约 0.4 GB。精度越低，权重占用越小，留给 KV Cache 的空间就越大。但精度降低也会影响生成质量，vLLM 支持 AWQ 和 GPTQ 等多种量化格式，在[并发性能与显存调优](#并发性能与显存调优)中会进一步展示量化对并发容量的影响。

## 单次推理延迟测量

模型加载完成后，下一步是测量推理延迟。vLLM 0.23 的 V1 引擎在 Python API 中不再直接暴露 `RequestOutput.metrics`，本节通过两次单独推理分别测量短提示和长提示的延迟，观察输入长度对推理速度的影响。

```python runnable gpuonly timeout=unlimited
import torch
import time
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from vllm import LLM, SamplingParams

# 加载本地下载的 Qwen3.5-0.8B-Instruct 模型
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

llm = LLM(
    model=model_path,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

short_prompt = "什么是人工智能？"
long_prompt = "请详细解释深度学习的发展历程，从感知机开始，到多层感知机、卷积神经网络、循环神经网络，再到 Transformer 架构。每个阶段请说明其核心创新和代表性工作。"

# 预热：消化 Triton JIT 编译延迟，确保后续测量不受首次推理的编译开销影响
llm.generate(["预热"], SamplingParams(temperature=0, max_tokens=1))

# 分别测量 TTFT（首 token 时间）和完整推理耗时
# 策略：先以 max_tokens=1 运行获取 Prefill 耗时（≈ TTFT），
#       再以 max_tokens=50 运行获取完整耗时，从中推导 TPOT
test_cases = [("长", long_prompt), ("短", short_prompt)]

for label, prompt in test_cases:
    # TTFT：限制输出 1 token，耗时即 Prefill + 首个 Decode step
    torch.cuda.synchronize()
    t0 = time.time()
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1))
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000

    # 完整推理
    torch.cuda.synchronize()
    t0 = time.time()
    output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=50))
    torch.cuda.synchronize()
    total_time = (time.time() - t0) * 1000

    prompt_len = len(output[0].prompt_token_ids)
    output_len = len(output[0].outputs[0].token_ids)
    tps = output_len / (total_time / 1000) if total_time > 0 else 0
    # TPOT = (总时间 - TTFT) / (输出 token 数 - 1)
    tpot = (total_time - ttft) / (output_len - 1) if output_len > 1 else 0

    print(f"\n=== 提示 ({label}) ===")
    print(f"提示: {prompt[:60]}...")
    print(f"输入长度: {prompt_len} tokens")
    print(f"输出长度: {output_len} tokens")
    print(f"TTFT (含首个 token): {ttft:.0f} ms")
    print(f"TPOT (后续每 token): {tpot:.1f} ms")
    print(f"总耗时: {total_time:.0f} ms")
    print(f"TPS: {tps:.1f} tokens/s")
    print(f"生成内容: {output[0].outputs[0].text[:100]}...")
```

代码中先执行一次预热推理消化 Triton JIT 编译开销，再通过两次推理测量 TTFT。从输出中可以观察到，长提示（37 个输入 token）的 TTFT 为 42 ms，明显高于短提示（3 个输入 token）的 18 ms。Prefill 阶段需要一次性计算所有输入 token 的自注意力，计算量随输入长度增长。而两个提示的 TPOT 基本持平（均在 6 ms 左右），因为 Decode 阶段每个新 token 只需与已有 KV Cache 做一次注意力运算，与输入长度无关。TTFT 和 TPOT 之间相差约一个数量级，印证了 Decode 阶段高度依赖 KV Cache 避免重复计算的特点。

```
=== 提示 (长) ===
提示: 请详细解释深度学习的发展历程...
输入长度: 37 tokens
输出长度: 50 tokens
TTFT (含首个 token): 42 ms
TPOT (后续每 token): 6.2 ms
总耗时: 345 ms
TPS: 145.1 tokens/s
生成内容: 深度学习的发展历程可以从感知机开始...

=== 提示 (短) ===
提示: 什么是人工智能？...
输入长度: 3 tokens
输出长度: 50 tokens
TTFT (含首个 token): 18 ms
TPOT (后续每 token): 6.6 ms
总耗时: 344 ms
TPS: 145.4 tokens/s
生成内容: 人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支...
```

## 并发性能与显存调优

前两个阶段使用 vLLM 的 Python API 在进程内直接推理，适合离线批量处理。生产环境中，推理服务通常以 HTTP 服务的形式部署，多个客户端通过 OpenAI 兼容 API 并发请求。本阶段将实际启动 vLLM 推理服务，通过模拟多客户端并发请求，观察不同并发度下的吞吐量和延迟变化。下面的代码首先通过 `subprocess` 启动 vLLM 的 OpenAI 兼容 API 服务，然后使用 `ThreadPoolExecutor` 模拟 1/2/4/8 路并发请求，最后基于实际硬件参数分析不同显存利用率下的并发容量。

```python runnable gpuonly timeout=unlimited
import subprocess
import time
import requests
import sys
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# 抑制 vLLM 引擎核心日志（子进程继承此环境变量）
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 加载本地下载的 Qwen3.5-0.8B-Instruct 模型
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

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
        "--model", model_path,
        "--trust-remote-code",
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
                "model": model_path,
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

model_weight_gb = 1.6   # 0.8B float16 权重大小
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

以下为执行结果输出：
```
并发数   | 总吞吐(token/s)  | 单请求TPS    | 总耗时(s)
-------------------------------------------------------
  1      |         108.6     |    108.8   |      0.5
  2      |         259.3     |    130.9   |      0.4
  4      |         342.0     |     86.1   |      0.6
  8      |         649.4     |     82.1   |      0.6

--- 显存利用率与并发容量 ---
GPU 显存总量: 7.6 GB
模型权重: 1.6 GB, 固定开销: 0.50 GB
单请求 KV Cache: ~0.2 GB (max_model_len=4096, 估算值)

利用率    | KV Cache可用(GB) | 预估并发上限  | OOM风险
----------------------------------------------------
  75%    |          3.62     |         18   | 低
  80%    |          4.00     |         19   | 低
  85%    |          4.38     |         21   | 低
  90%    |          4.76     |         23   | 中
  95%    |          5.14     |         25   | 高
```

从输出中可以看到，吞吐量随并发数增长但逐渐放缓，与[请求调度](request-scheduling.md)中讨论的批处理效率曲线一致。显存利用率分析清楚地展示了 `gpu_memory_utilization` 参数的杠杆作用，从 0.85 调到 0.90，并发容量可以提升 10-20%，但 OOM 风险也随之升高。生产环境中推荐的设置是 0.85-0.90，当显存特别紧张时可以短时间用到 0.95，但需要配合监控和告警。

除了显存利用率，vLLM 还提供了抢占策略来处理超量并发的场景。当新请求到达而显存不足时，抢占策略 `swap` 将部分 KV Cache 换出到 CPU 内存，等 GPU 空闲时再换回；驱逐策略 `recomputation` 则直接丢弃被抢占请求的 KV Cache，等恢复执行时重新计算 Prefill。选择哪种策略取决于请求负载特征。短文本为主的场景推荐 Recomputation（重新计算的代价小、不占 CPU 内存），长文本居多的场景推荐 Swap（切换开销低）。

## 流式输出与 KV Cache 实验

前面的实验使用的是非流式推理，请求发送后等待完整响应返回。对于终端用户而言，等待几秒看到空白页面和逐字看到内容流出，体验截然不同。vLLM 的 OpenAI 兼容 API 支持 `stream=True` 参数，通过 SSE（Server-Sent Events）协议逐 token 推送生成内容。本阶段启动 vLLM 服务后，发起流式请求并逐 token 解析 SSE 响应，精确记录首个 token 的到达时间（TTFT）和每个后续 token 的到达间隔，直观展示 Prefill 与 Decode 阶段的计算量差异。

```python runnable gpuonly timeout=unlimited
import subprocess
import time
import requests
import sys
import json
import os

# 抑制 vLLM 引擎核心日志（子进程继承此环境变量）
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 加载本地下载的 Qwen3.5-0.8B-Instruct 模型
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

# ========== 1. 启动 vLLM 推理服务 ==========
print("正在启动 vLLM 推理服务...")
server_proc = subprocess.Popen(
    [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--trust-remote-code",
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
ttft_ms = None  # 初始化，防止无 token 返回时 NameError
token_count = 0
token_times = []
generated_text = ""

# 发起流式请求（stream=True）
response = requests.post(
    f"{base_url}/v1/completions",
    json={
        "model": model_path,
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
if token_count == 0:
    print("--- 流式输出指标 ---")
    print("未收到任何 token，请检查 vLLM 服务日志排查原因")
else:
    decode_start = token_times[0]
    decode_end = token_times[-1]
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
    if ttft_ms and avg_interval > 0:
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

这段代码运行时会逐 token 在页面上打印生成内容，你可以清晰看到首个 token 在短暂的等待后出现，随后每个 token 以大致相等的时间间隔接连输出。[流式输出](inference-service-architecture.md#流式输出与-server-sent-events)的背后是 KV Cache 机制在持续发挥作用。Decode 阶段每生成一个新 token，只有这个新 token 的 Query 需要与所有历史 token 的 Key 和 Value 进行注意力计算。历史 token 的 K、V 向量早在 Prefill 阶段就已算出并缓存在显存中。在多轮对话场景中，这种复用更加明显。前几轮对话的 KV Cache 可以直接复用，只有新增的用户输入部分需要做 Prefill。

vLLM 的[前缀缓存](./request-scheduling.md#前缀缓存)进一步利用这一特性。当多个请求共享相同的系统提示词时，这段提示词的 KV Cache 只需计算一次，后续请求直接复用。在实际聊天应用中，系统提示词通常有数百 token，前缀缓存可以将这些请求的 TTFT 缩短 30-50%。

## 实验结论

本次实验使用 vLLM 推理框架完整展示了推理服务从模型加载到流式输出的全流程。与直接使用 Transformers 做推理相比，vLLM 的价值体现在以下几个层面：

- **显存管理**：PagedAttention 将 KV Cache 切分为固定大小的 Block，像操作系统的虚拟内存页面一样管理，从根本上消除了显存碎片。这使得相同的 GPU 显存可以容纳更多并发请求。实验中通过 `gpu_memory_utilization` 参数直观看到了显存分配策略对并发容量的影响。

- **调度效率**：Continuous Batching 在请求到达和完成时动态调整批处理组合，而非等待整批完成后再组新批。实验中同时发送两条不同长度的提示时，可以看到它们被自动合并执行，总耗时小于单独执行之和。并发测试进一步验证了吞吐量随并发数增长的亚线性特征。

- **流式体验**：通过 SSE 协议逐 token 推送生成内容，将用户的感知延迟从全部生成时间缩短到首 token 时间。实验中 TTFT 与平均 token 间隔的巨大差距（可达 10 倍以上），直观说明了流式输出对改善用户体验的关键作用。

本次实验使用 vLLM 的 Python API 和 Server 两种模式。Python API 适合离线批量处理和性能测量，Server 模式适合模拟生产环境的并发请求和流式输出场景。生产部署中推荐使用 `vllm.entrypoints.openai.api_server` 启动 OpenAI 兼容的 HTTP 服务，配合 Nginx 等反向代理实现负载均衡和认证鉴权。
