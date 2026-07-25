# Hands-on Lab: Deploying an LLM Inference Service

Through the first three chapters, we have learned about the [architecture principles](inference-service-architecture.md) of inference services, [request scheduling and batching mechanisms](request-scheduling.md), and [GPU resource management strategies](gpu-resource-management.md). This knowledge forms the theoretical foundation for understanding LLM inference serving. In this lab, we will use the vLLM inference framework to walk through the entire process of getting an LLM inference service from zero to production-ready, covering model loading, performance tuning, and streaming output.

## Experiment Preparation

This lab continues using the [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) model introduced in the [LLM Reasoning Strategy and Efficiency Optimization experiment](../../language-models/reasoning/llm-reasoning-experiment.md) as the demonstration model. It can be easily loaded on a single GPU and is sufficient to demonstrate the various performance characteristics of inference services. If you have not yet completed that experiment, you need to download the Qwen3.5-0.8B-Instruct model first. This can be done automatically using the `DMLA-CLI` tool:

```bash
# Download model: select "Download Model" -> select "Qwen3.5-0.8B-Instruct"
dmla model
```

The core dependency of this experiment is the vLLM inference framework, which comes pre-installed in the DMLA sandbox image. You can verify the GPU environment, CUDA support, and vLLM availability with the following code:

```python runnable gpuonly
import torch
import os

# Verify that the Qwen3.5-0.8B-Instruct model is correctly downloaded
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
model_ready = False
if os.path.exists(model_path):
    # Check for LFS incomplete marker file
    incomplete_marker = os.path.join(model_path, '.lfs-incomplete')
    if os.path.exists(incomplete_marker):
        print(f"Model directory exists but data is incomplete (LFS not pulled): {model_path}")
        print(f"Please ensure Git LFS is installed, then run: cd {model_path} && git lfs pull")
    else:
        # Check for required files
        required_files = ['config.json', 'tokenizer_config.json']
        missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        if missing:
            print(f"Model directory exists but missing required files: {missing}")
            print(f"Model path: {model_path}")
            print(f"Please run 'dmla model' to re-download the model")
        else:
            model_ready = True
            print(f"Model ready: Qwen3.5-0.8B-Instruct")
else:
    print(f"Model not downloaded: {model_path}")
    print("Please run 'dmla model' to download the Qwen3.5-0.8B-Instruct model")
    print("  1. Run dmla model to enter TUI")
    print("  2. Select 'Download Model' -> select 'Qwen3.5-0.8B-Instruct'")
print()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Current memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
else:
    print("CUDA is not available. Please check GPU driver and CUDA installation.")

# Verify vLLM availability
try:
    import vllm
    print(f"\nvLLM version: {vllm.__version__}")
    print("vLLM available ✓")
except ImportError:
    print("\nvLLM is not installed. If you are using an older Docker image, please re-pull or rebuild the image with vLLM included.")
    print("Manual installation: pip install vllm")
```

## Model Loading and Memory Analysis

The first step in starting an inference service is loading the model into GPU memory. Unlike previous experiments that used the Transformers framework directly for model loading, this experiment uses the vLLM inference framework. vLLM automatically enables the PagedAttention mechanism, which splits the KV Cache into fixed-size blocks and manages them like virtual memory pages in an operating system, allowing limited GPU memory to accommodate more concurrent requests.

vLLM controls the upper bound of memory usage through the `gpu_memory_utilization` parameter (default 0.90). The higher this value, the more KV Cache blocks are pre-allocated and the more requests can be processed simultaneously, but less headroom is left for the CUDA runtime and temporary buffers. The following code loads the model with a utilization of 0.85, observes the memory changes before and after loading, and analyzes the contribution of each component.

```python runnable gpuonly timeout=unlimited
import torch
import os

# Record memory state before loading
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    before_mem = torch.cuda.memory_allocated() / 1024**3

# Use locally downloaded Qwen3.5-0.8B-Instruct model
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

# Suppress vLLM engine core config dump (excessive output can cause Kernel output parsing failure)
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# Load model with vLLM, automatically enabling PagedAttention
# gpu_memory_utilization=0.85: use at most 85% of GPU memory, leaving the rest for CUDA context
# max_model_len=4096: limits the maximum sequence length, directly affecting KV Cache block pre-allocation
from vllm import LLM

llm = LLM(
    model=model_path,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

# vLLM uses a multi-process engine core where model loading happens in child processes;
# the main process's torch.cuda.memory_allocated() does not reflect actual usage.
# Actual memory data can be found in the INFO log lines "Model loading took" and "GPU KV cache size" above.
# The following analysis is based on theoretical values (refer to console logs for actual measurements):
if torch.cuda.is_available():
    # Theoretical model weight size: number of parameters x float16 bytes
    # 0.8B params x 2 bytes = 1.6 GB
    model_weight_gb = 0.8 * 2

    print(f"\nModel: Qwen3.5-0.8B-Instruct (float16)")
    print(f"\n--- Memory Composition Analysis ---")
    print(f"Model weights (theoretical): {model_weight_gb:.2f} GB")
    print(f"KV Cache Blocks + CUDA overhead: See 'Model loading took' and 'GPU KV cache size' logs above")
    print(f"  (PagedAttention pre-allocated block pool + PyTorch CUDA context)")
```

The vLLM engine core runs in child processes, so `torch.cuda.memory_allocated()` does not reflect actual GPU memory usage. Below are key data from the vLLM initialization logs:

```
Model: Qwen3.5-0.8B-Instruct (float16)

--- Memory Composition Analysis ---
Model weights (theoretical): 1.6 GB

Actual log output:
  Model loading took 1.72 GiB memory and 0.72 seconds
  GPU KV cache size: 183,202 tokens
  Available KV cache memory: 3.06 GiB
  CUDA graph pool memory: 0.42 GiB
  Maximum concurrency for 4,096 tokens per request: 44.73x
```

From vLLM's initialization logs, we can observe that GPU memory usage consists of three parts:
- Model weights (actual measurement ~1.72 GiB, close to the theoretical value of 1.6 GB)
- Pre-allocated KV Cache block pool for PagedAttention (3.06 GiB available, 183,202 tokens total)
- Fixed cache generated by CUDA Graph compilation (0.42 GiB)

The memory footprint of model weights varies significantly across different precisions. Taking a 0.8B model as an example, FP32 weights occupy approximately 3.2 GB, FP16 about 1.6 GB, and INT4 quantization can compress them to roughly 0.4 GB. Lower precision means smaller weight footprint and more space for the KV Cache. However, lower precision also affects generation quality. vLLM supports various quantization formats such as AWQ and GPTQ, and the impact of quantization on concurrency capacity will be further demonstrated in [Concurrent Performance and Memory Tuning](#concurrent-performance-and-memory-tuning).

## Single Inference Latency Measurement

After model loading, the next step is to measure inference latency. vLLM 0.23's V1 engine no longer directly exposes `RequestOutput.metrics` in the Python API. This section measures the latency of short and long prompts through two separate inference runs to observe the effect of input length on inference speed.

```python runnable gpuonly timeout=unlimited
import torch
import time
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from vllm import LLM, SamplingParams

# Load locally downloaded Qwen3.5-0.8B-Instruct model
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

llm = LLM(
    model=model_path,
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

short_prompt = "What is artificial intelligence?"
long_prompt = "Please explain in detail the development history of deep learning, starting from the perceptron, through multi-layer perceptrons, convolutional neural networks, recurrent neural networks, and finally the Transformer architecture. For each stage, please describe its core innovations and representative work."

# Warmup: absorb Triton JIT compilation latency to ensure subsequent measurements are not affected by first-inference compilation overhead
llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=1))

# Measure TTFT (Time to First Token) and total inference latency separately
# Strategy: first run with max_tokens=1 to get Prefill time (approximate TTFT),
#           then run with max_tokens=50 to get total time, deriving TPOT from both
test_cases = [("Long", long_prompt), ("Short", short_prompt)]

for label, prompt in test_cases:
    # TTFT: limit output to 1 token, time equals Prefill + first Decode step
    torch.cuda.synchronize()
    t0 = time.time()
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1))
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000

    # Full inference
    torch.cuda.synchronize()
    t0 = time.time()
    output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=50))
    torch.cuda.synchronize()
    total_time = (time.time() - t0) * 1000

    prompt_len = len(output[0].prompt_token_ids)
    output_len = len(output[0].outputs[0].token_ids)
    tps = output_len / (total_time / 1000) if total_time > 0 else 0
    # TPOT = (total time - TTFT) / (output tokens - 1)
    tpot = (total_time - ttft) / (output_len - 1) if output_len > 1 else 0

    print(f"\n=== Prompt ({label}) ===")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Input length: {prompt_len} tokens")
    print(f"Output length: {output_len} tokens")
    print(f"TTFT (including first token): {ttft:.0f} ms")
    print(f"TPOT (subsequent per-token): {tpot:.1f} ms")
    print(f"Total time: {total_time:.0f} ms")
    print(f"TPS: {tps:.1f} tokens/s")
    print(f"Generated content: {output[0].outputs[0].text[:100]}...")
```

The code first executes a warmup inference to absorb Triton JIT compilation overhead, then measures TTFT through two inference runs. From the output, we can observe that the long prompt (37 input tokens) has a TTFT of 42 ms, significantly higher than the short prompt (3 input tokens) at 18 ms. The Prefill phase needs to compute self-attention over all input tokens at once, with the computation scaling linearly with input length. Meanwhile, the TPOT for both prompts is roughly the same (around 6 ms), because during the Decode phase, each new token only needs to perform attention against the existing KV Cache, regardless of input length. The roughly order-of-magnitude gap between TTFT and TPOT confirms that the Decode phase relies heavily on the KV Cache to avoid redundant computation.

```
=== Prompt (Long) ===
Prompt: Please explain in detail the development history of deep learning...
Input length: 37 tokens
Output length: 50 tokens
TTFT (including first token): 42 ms
TPOT (subsequent per-token): 6.2 ms
Total time: 345 ms
TPS: 145.1 tokens/s
Generated content: The development history of deep learning starts from the perceptron...

=== Prompt (Short) ===
Prompt: What is artificial intelligence?...
Input length: 3 tokens
Output length: 50 tokens
TTFT (including first token): 18 ms
TPOT (subsequent per-token): 6.6 ms
Total time: 344 ms
TPS: 145.4 tokens/s
Generated content: Artificial Intelligence (AI) is a branch of computer science...
```

## Concurrent Performance and Memory Tuning

The first two phases used vLLM's Python API for in-process inference, which is suitable for offline batch processing. In production environments, inference services are typically deployed as HTTP services, with multiple clients sending concurrent requests through an OpenAI-compatible API. In this phase, we will actually start a vLLM inference service, simulate multi-client concurrent requests, and observe throughput and latency changes at different concurrency levels. The code below first starts vLLM's OpenAI-compatible API service via `subprocess`, then uses `ThreadPoolExecutor` to simulate 1/2/4/8 concurrent requests, and finally analyzes the concurrency capacity under different memory utilization levels based on actual hardware parameters.

```python runnable gpuonly timeout=unlimited
import subprocess
import time
import requests
import sys
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress vLLM engine core logs (child process inherits this environment variable)
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# Load locally downloaded Qwen3.5-0.8B-Instruct model
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

# Get GPU memory info (must query before vLLM starts to avoid CUDA context conflicts)
if torch.cuda.is_available():
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
else:
    gpu_total_gb = 8.0

# ========== 1. Start vLLM Inference Service ==========
print("Starting vLLM inference service...")
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

# Poll the /health endpoint until the service is ready
base_url = "http://127.0.0.1:8000"
ready = False
for attempt in range(90):
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            print(f"vLLM service ready (took approx {attempt*2}s)\n")
            break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    server_proc.kill()
    raise RuntimeError("vLLM service startup timed out. Please check if GPU memory is sufficient.")

# ========== 2. Concurrent Request Test ==========
prompt = "Please describe the inference process of large language models in three sentences"

def send_request(req_id):
    """Send one inference request, return elapsed time and output token count"""
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
print(f"{'Concurrency':<10} | {'Throughput(token/s)':<20} | {'Avg TPS/req':<14} | {'Total Time(s)'}")
print("-" * 65)

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

    print(f"  {c:<8} | {throughput:>14.1f}     | {avg_tps:>10.1f}   | {total_time:>10.1f}")

# ========== 3. Memory Utilization and Concurrency Capacity Analysis ==========
print(f"\n--- Memory Utilization and Concurrency Capacity ---")

model_weight_gb = 1.6   # 0.8B float16 weight size
runtime_overhead_gb = 0.5  # CUDA context and other fixed overhead
kv_per_request_gb = 0.2    # Single request KV Cache estimate (4096 tokens)

print(f"GPU total memory: {gpu_total_gb:.1f} GB")
print(f"Model weights: {model_weight_gb:.2f} GB, Fixed overhead: {runtime_overhead_gb:.2f} GB")
print(f"KV Cache per request: ~{kv_per_request_gb:.1f} GB (max_model_len=4096, estimate)")
print()
print(f"{'Utilization':<12} | {'KV Cache Avail(GB)':<18} | {'Est. Max Concurrency':<20} | {'OOM Risk'}")
print("-" * 62)

for util in [0.75, 0.80, 0.85, 0.90, 0.95]:
    available = gpu_total_gb * util
    kv_space = available - model_weight_gb - runtime_overhead_gb
    max_conc = max(1, int(kv_space / kv_per_request_gb))
    oom = "Low" if util < 0.88 else ("Medium" if util < 0.93 else "High")
    print(f"  {util:.0%}      | {kv_space:>14.2f}     | {max_conc:>16}   | {oom}")

print(f"\nWith current config gpu_memory_utilization=0.85, the theoretical concurrency upper bound is approximately "
      f"{max(1, int((gpu_total_gb * 0.85 - model_weight_gb - runtime_overhead_gb) / kv_per_request_gb))} requests")

# ========== 4. Cleanup ==========
server_proc.terminate()
try:
    server_proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    server_proc.kill()
print("\nvLLM service stopped")
```

Below is the sample execution output:
```
Concurrency | Throughput(token/s) | Avg TPS/req  | Total Time(s)
-----------------------------------------------------------------
  1         |             108.6   |      108.8   |        0.5
  2         |             259.3   |      130.9   |        0.4
  4         |             342.0   |       86.1   |        0.6
  8         |             649.4   |       82.1   |        0.6

--- Memory Utilization and Concurrency Capacity ---
GPU total memory: 7.6 GB
Model weights: 1.6 GB, Fixed overhead: 0.50 GB
KV Cache per request: ~0.2 GB (max_model_len=4096, estimate)

Utilization  | KV Cache Avail(GB) | Est. Max Concurrency | OOM Risk
--------------------------------------------------------------
  75%        |           3.62     |                 18   | Low
  80%        |           4.00     |                 19   | Low
  85%        |           4.38     |                 21   | Low
  90%        |           4.76     |                 23   | Medium
  95%        |           5.14     |                 25   | High
```

From the output, we can see that throughput increases with concurrency but gradually slows down, consistent with the batching efficiency curve discussed in [Request Scheduling](request-scheduling.md). The memory utilization analysis clearly demonstrates the leverage of the `gpu_memory_utilization` parameter. Increasing it from 0.85 to 0.90 can boost concurrency capacity by 10-20%, but also raises the OOM risk accordingly. In production environments, a setting of 0.85-0.90 is recommended. When memory is particularly tight, 0.95 can be used for short periods, but monitoring and alerting should be in place.

Beyond memory utilization, vLLM also provides preemption strategies for handling situations with excessive concurrency. When a new request arrives and memory is insufficient, the `swap` preemption strategy moves some KV Cache blocks to CPU memory and swaps them back when the GPU is free; the `recomputation` eviction strategy directly discards the preempted request's KV Cache and recomputes the Prefill when execution resumes. Which strategy to choose depends on the request load characteristics. For scenarios dominated by short texts, Recomputation is recommended (low recomputation cost, no CPU memory consumption), while for scenarios with many long texts, Swap is recommended (low switching overhead).

## Streaming Output and KV Cache Experiment

The previous experiments used non-streaming inference, where the response is returned only after the entire output is generated. For end users, the experience of staring at a blank page for several seconds is vastly different from watching content appear word by word. vLLM's OpenAI-compatible API supports the `stream=True` parameter, pushing generated content token by token through the SSE (Server-Sent Events) protocol. In this phase, after starting the vLLM service, we send a streaming request and parse the SSE response token by token, precisely recording the arrival time of the first token (TTFT) and the interval between each subsequent token, providing an intuitive demonstration of the computational difference between the Prefill and Decode phases.

```python runnable gpuonly timeout=unlimited
import subprocess
import time
import requests
import sys
import json
import os

# Suppress vLLM engine core logs (child process inherits this environment variable)
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# Load locally downloaded Qwen3.5-0.8B-Instruct model
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')

# ========== 1. Start vLLM Inference Service ==========
print("Starting vLLM inference service...")
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

# Wait for service to be ready
base_url = "http://127.0.0.1:8000"
ready = False
for attempt in range(90):
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            print(f"vLLM service ready (took approx {attempt*2}s)\n")
            break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    server_proc.kill()
    raise RuntimeError("vLLM service startup timed out")

# ========== 2. Streaming Request + Token-by-Token Output ==========
prompt = "Please describe the inference process of large language models in three sentences"
print(f"Prompt: {prompt}\n")

start_time = time.time()
first_token_time = None
ttft_ms = None  # Initialize to prevent NameError if no token is returned
token_count = 0
token_times = []
generated_text = ""

# Send streaming request (stream=True)
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

print("Streaming output: ", end="", flush=True)

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
            print(f"\n[TTFT] {ttft_ms:.0f} ms — Prefill complete, first token generated\n")
            print("Streaming output: ", end="", flush=True)

        token_count += 1
        token_times.append(now)
        # Print token by token, flush ensures each token is immediately pushed to the front end
        print(token_text, end="", flush=True)

    except json.JSONDecodeError:
        pass

print("\n")

# ========== 3. Calculate Streaming Output Metrics ==========
if token_count == 0:
    print("--- Streaming Output Metrics ---")
    print("No tokens received. Please check vLLM service logs for troubleshooting.")
else:
    decode_start = token_times[0]
    decode_end = token_times[-1]
    decode_time_ms = (decode_end - decode_start) * 1000

    # Calculate intervals between consecutive tokens
    intervals = []
    for i in range(1, len(token_times)):
        intervals.append((token_times[i] - token_times[i - 1]) * 1000)

    avg_interval = sum(intervals) / len(intervals) if intervals else 0

    print(f"--- Streaming Output Metrics ---")
    print(f"TTFT (first token): {ttft_ms:.0f} ms")
    print(f"Decode total time: {decode_time_ms:.0f} ms")
    print(f"Output length: {token_count} tokens")
    print(f"Average Token Interval: {avg_interval:.1f} ms/token")
    print(f"TPS: {token_count / (decode_time_ms / 1000):.1f} tokens/s")

    # Compare Prefill and Decode computation differences
    if ttft_ms and avg_interval > 0:
        ratio = ttft_ms / avg_interval
        print(f"\n--- The Role of KV Cache ---")
        print(f"TTFT / Average Token Interval = {ratio:.0f}x")
        print(f"This means the Prefill phase processes approximately {ratio:.0f} times the computation of a single Decode step in one shot")
        print(f"The Decode phase is fast precisely because it reuses the KV Cache generated during Prefill")

# ========== 4. Cleanup ==========
server_proc.terminate()
try:
    server_proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    server_proc.kill()
print("\nvLLM service stopped")
```

When this code runs, generated content is printed token by token on the page. You can clearly see the first token appear after a brief wait, followed by subsequent tokens output at roughly equal intervals. Behind [Streaming Output](inference-service-architecture.md#streaming-output-and-server-sent-events), the KV Cache mechanism is continuously at work. During the Decode phase, each time a new token is generated, only this new token's Query needs to perform attention against the Keys and Values of all historical tokens. The K and V vectors of historical tokens were already computed during the Prefill phase and cached in GPU memory. In multi-turn conversation scenarios, this reuse is even more pronounced. The KV Cache from previous conversation turns can be directly reused, with only the newly added user input requiring Prefill computation.

vLLM's [Prefix Caching](./request-scheduling.md#prefix-caching) further leverages this property. When multiple requests share the same system prompt, the KV Cache for this prompt only needs to be computed once, and subsequent requests reuse it directly. In real-world chat applications, system prompts are typically hundreds of tokens long, and prefix caching can reduce TTFT for these requests by 30-50%.

## Conclusion

This experiment uses the vLLM inference framework to demonstrate the complete pipeline of an inference service, from model loading to streaming output. Compared with directly using Transformers for inference, vLLM's value is reflected in the following aspects:

- **Memory Management**: PagedAttention splits the KV Cache into fixed-size blocks, managed like virtual memory pages in an operating system, fundamentally eliminating memory fragmentation. This allows the same GPU memory to accommodate more concurrent requests. The experiment demonstrated the impact of memory allocation strategies on concurrency capacity through the `gpu_memory_utilization` parameter.

- **Scheduling Efficiency**: Continuous Batching dynamically adjusts the batching combination as requests arrive and complete, rather than waiting for an entire batch to finish before forming a new one. In the experiment, when two prompts of different lengths were sent simultaneously, they were automatically merged for execution, with the total time being less than the sum of individual executions. The concurrency test further verified the sub-linear characteristic of throughput growth with concurrency.

- **Streaming Experience**: By pushing generated content token by token through the SSE protocol, the user's perceived latency is reduced from total generation time to time-to-first-token. The large gap between TTFT and average token interval (up to 10x or more) visually demonstrates the critical role of streaming output in improving user experience.

This experiment used both vLLM's Python API and Server modes. The Python API is suitable for offline batch processing and performance measurement, while the Server mode is suitable for simulating production environment concurrent requests and streaming output scenarios. For production deployments, it is recommended to use `vllm.entrypoints.openai.api_server` to start an OpenAI-compatible HTTP service, combined with a reverse proxy such as Nginx for load balancing and authentication.
