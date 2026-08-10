# LLM Reasoning Strategies and Efficiency Optimization Experiment

In previous chapters, we explored chain-of-thought, test-time scaling, and inference efficiency optimization from a theoretical perspective. In this section, we shift to practice and experience the engineering trade-offs in reasoning strategies. The 64M parameter model trained in [the earlier experiment](../architecture-basics/llm-pretrain-experiment.md) is too small for chain-of-thought and reasoning scaling. This experiment uses the open-source [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) model to implement a full-chain practice ranging from chain-of-thought prompting to reasoning scaling and efficiency optimization.

## Experiment Setup

Before starting the experiment, please ensure you have [mounted the data directory](../../appendixes/sandbox.md#data-management) and downloaded the GSM8K evaluation subset and the Qwen3.5-0.8B-Instruct model. You can accomplish both tasks using the `DMLA-CLI` tool:

```bash
# Download dataset: select "Download dataset" -> select "GSM8K 200 (Math Reasoning Evaluation Set)"
dmla data

# Download model: select "Download model" -> select "Qwen3.5-0.8B-Instruct"
dmla model
```

GSM8K (Grade School Math 8K) is a benchmark of elementary math word problems containing 7,473 training questions and 1,319 test questions, where models need multi-step calculations to arrive at the correct answer. From this dataset, we randomly sampled 200 problems as an evaluation subset to demonstrate the performance differences across reasoning strategies under limited computational resources, while keeping evaluation time within a reasonable range. After downloading the dataset, run the following code to verify that data and model loading are working correctly, and pre-quantize and save INT8 and INT4 models for [Phase 3 (Inference Efficiency Optimization)](#phase-3-inference-efficiency-optimization):

```python runnable gpuonly timeout=unlimited
import os
import json
import logging
import warnings
import torch
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# Suppress bitsandbytes quantization precision conversion warnings and PyTorch internal API deprecation warnings
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dmla_progress import ProgressReporter

# ========== Verify Dataset ==========
gsm8k_dir = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
if os.path.exists(gsm8k_dir):
    with open(os.path.join(gsm8k_dir, 'gsm8k_200.jsonl'), 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    print(f"GSM8K evaluation subset: {len(questions)} questions")
    print(f"Example question: {questions[0]['question'][:80]}...")
else:
    print("GSM8K evaluation subset: not downloaded. Please run 'dmla data' to download the dataset")

# ========== Load Model ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
print(f"\nLoading model {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params / 1e9:.2f}B ({total_params:,})")
print(f"Model dtype: {model.dtype}")
print(f"Device: {model.device}")

# ========== Quantized Model Saving ==========
progress = ProgressReporter(total_steps=2, description="Quantized model saving")

# INT8 quantization and save
int8_save_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int8')
if os.path.exists(int8_save_path):
    print(f"\nINT8 quantized model already exists, skipping: {int8_save_path}")
else:
    progress.update(1, message="Quantizing INT8 model...")
    int8_config = BitsAndBytesConfig(load_in_8bit=True)
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=int8_config, device_map="auto"
    )
    progress.update(1, message="Saving INT8 model to disk...")
    model_int8.save_pretrained(int8_save_path, safe_serialization=True)
    tokenizer.save_pretrained(int8_save_path)
    del model_int8
    print(f"\nINT8 quantized model saved: {int8_save_path}")

torch.cuda.empty_cache()

# INT4 quantization and save
int4_save_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int4')
if os.path.exists(int4_save_path):
    print(f"INT4 quantized model already exists, skipping: {int4_save_path}")
else:
    progress.update(2, message="Quantizing INT4 model...")
    int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model_int4 = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=int4_config, device_map="auto"
    )
    progress.update(2, message="Saving INT4 model to disk...")
    model_int4.save_pretrained(int4_save_path, safe_serialization=True)
    tokenizer.save_pretrained(int4_save_path)
    del model_int4
    print(f"INT4 quantized model saved: {int4_save_path}")

torch.cuda.empty_cache()

# Calculate disk usage of model weight files (only model.safetensors, excluding .git, tokenizer, etc.)
def get_model_file_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.startswith('model') and f.endswith('.safetensors'):
                total += os.path.getsize(os.path.join(dirpath, f))
    return total / 1024**3

fp16_size = get_model_file_size(model_path)
int8_size = get_model_file_size(int8_save_path)
int4_size = get_model_file_size(int4_save_path)

print(f"\n{'Model':<12} {'Weight Size':>12}")
print("-" * 27)
print(f"{'FP16 (Original)':<12} {fp16_size:>11.2f}GB")
print(f"{'INT8':<12} {int8_size:>11.2f}GB")
print(f"{'INT4':<12} {int4_size:>11.2f}GB")
print(f"\nINT8 vs FP16: {int8_size/fp16_size*100:.1f}%")
print(f"INT4 vs FP16: {int4_size/fp16_size*100:.1f}%")

progress.complete(message="Quantized model saving complete")
```

## Phase 1: Chain of Thought and Prompt Engineering

In the [Chain of Thought and Reasoning Models](chain-of-thought.md#chain-of-thought) chapter, we saw that chain-of-thought prompting can shift a model from "answering by intuition" to "reasoning step by step." In this phase, we compare three prompting strategies on the GSM8K evaluation set: zero-shot direct answering, zero-shot CoT, and few-shot CoT. The three strategies differ in their prompt design:

- **Zero-Shot Direct**: The question is posed directly without any reasoning guidance; the model produces the answer immediately.
- **Zero-Shot CoT**: A guiding phrase "Please think step by step" is appended after the question, prompting the model to display its reasoning process. This is the zero-shot chain-of-thought method introduced in the [Chain of Thought](chain-of-thought.md#chain-of-thought) chapter.
- **Few-Shot CoT**: 2-3 examples with complete reasoning processes are provided before the question, allowing the model to learn how to organize reasoning steps.

The Qwen3.5 series natively supports Thinking/Non-Thinking mode switching. However, in practice, when the lightweight 0.8B model enables Thinking mode, it tends to generate meta-descriptions about "how to think" (such as "analyze request" or "decompose problem") rather than directly performing mathematical reasoning, which actually degrades problem-solving quality. Therefore, this experiment uniformly uses Non-Thinking mode and achieves the comparison of three strategies through differences in prompting.

::: info Inference Time

Although the Qwen3.5-0.8B-Instruct model is already very lightweight, running 200 inferences for each of the three prompts (600 total) still takes considerable time. On an RTX 5080, this takes approximately 90 minutes.

:::

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import torch
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== Configuration ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_samples = 200  # Number of evaluation questions

# ========== Load Data ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_samples]
print(f"Evaluation questions: {len(questions)}")

# ========== Load Model ==========
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

# ========== Answer Extraction ==========
def extract_answer(text):
    """Extract the final numerical answer from model output and GSM8K references"""
    # 1. GSM8K reference answer format: "#### 123"
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # 2. Thinking mode: extract from content after </think> tag
    think_end_pos = text.rfind('</think>')
    if think_end_pos != -1:
        after_think = text[think_end_pos + len('</think>'):].strip()
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', after_think)
        if match:
            return match.group(1).replace(',', '')
        match = re.search(r'(?:答案是|answer\s+is|final\s+answer)\s*[=:：]?\s*(-?[\d,]+\.?\d*)', after_think, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')
        nums = re.findall(r'-?\d+\.?\d*', after_think.replace(',', ''))
        if nums:
            return nums[-1]
    # 3. Extract from "答案是/answer is" patterns
    match = re.search(r'(?:答案是|answer\s+is|final\s+answer)\s*[=:：]?\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    # 4. Extract from non-calculation lines at the end (avoid extracting intermediate step numbers)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if '=' in line and not re.match(r'^-?[\d,]+\.?\d*$', line):
            continue
        nums = re.findall(r'-?\d+\.?\d*', line.replace(',', ''))
        if nums:
            return nums[-1]
    # 5. Fallback: take the last number
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== Inference Function ==========
def generate_response(messages, max_new_tokens=1024):
    """Generate a response from the model"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

# ========== Few-Shot Examples ==========
few_shot_examples = [
    {
        "role": "user",
        "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    },
    {
        "role": "assistant",
        "content": "Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast, so she has 16 - 3 = 13 eggs left.\nShe uses 4 for muffins, so she has 13 - 4 = 9 eggs left to sell.\nShe sells each egg for $2, so she makes 9 * 2 = $18 per day.\n#### 18"
    },
    {
        "role": "user",
        "content": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
    },
    {
        "role": "assistant",
        "content": "The robe takes 2 bolts of blue fiber.\nWhite fiber is half of blue: 2 / 2 = 1 bolt.\nTotal bolts = 2 + 1 = 3.\n#### 3"
    }
]

# ========== Evaluate Three Strategies ==========
strategies = {
    "Zero-Shot Direct": lambda q: [{"role": "user", "content": q}],
    "Zero-Shot CoT": lambda q: [{"role": "user", "content": q + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}],
    "Few-Shot CoT": lambda q: few_shot_examples + [{"role": "user", "content": q + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}],
}

results = {}
progress = ProgressReporter(total_steps=len(strategies) * num_samples, description="CoT Prompt Strategy Evaluation")

for strategy_name, make_messages in strategies.items():
    correct = 0
    total = 0
    total_time = 0
    total_tokens = 0

    for i, item in enumerate(questions):
        messages = make_messages(item['question'])
        ref_answer = extract_answer(item['answer'])

        start_time = time.time()
        try:
            response = generate_response(messages)
        except Exception as e:
            response = ""
            print(f"  Question {i+1} generation failed: {e}")
        elapsed = time.time() - start_time

        pred_answer = extract_answer(response)
        is_correct = pred_answer == ref_answer if pred_answer and ref_answer else False

        correct += int(is_correct)
        total += 1
        total_time += elapsed
        total_tokens += len(tokenizer.encode(response))

        progress.update(
            len(results) * num_samples + i + 1,
            message=f"{strategy_name}: {i+1}/{num_samples}, current accuracy {correct/total*100:.1f}%"
        )

    accuracy = correct / total * 100
    avg_time = total_time / total
    avg_tokens = total_tokens / total
    results[strategy_name] = {
        "accuracy": accuracy,
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "correct": correct,
        "total": total
    }
    print(f"\n{strategy_name}:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Avg time: {avg_time:.2f}s/question")
    print(f"  Avg output tokens: {avg_tokens:.0f}")

# ========== Summary Results ==========
print("\n" + "="*60)
print("Chain of Thought Prompt Strategy Comparison")
print("="*60)
print(f"{'Strategy':<25} {'Accuracy':>8} {'Avg Time':>10} {'Avg Tokens':>12}")
print("-"*60)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['avg_tokens']:>11.0f}")

progress.complete(message="CoT prompt strategy evaluation complete")
```

After running the code above, you should observe the following patterns:

- Zero-shot direct answering has the lowest accuracy because the model does not display its reasoning process, making it prone to errors in multi-step calculations or missing key information.
- Zero-shot CoT shows a clear improvement in accuracy; the "please think step by step" prompt makes the model show its reasoning process, reducing calculation omissions.
- Few-shot CoT achieves the highest accuracy, as the examples provide the model with a reasoning format template, showing it how to organize reasoning steps and annotate the final answer.

Below are the actual results of Qwen3.5-0.8B-Instruct on the GSM8K evaluation subset:

| Strategy | Accuracy | Avg Time |
|----------|----------|----------|
| Zero-Shot Direct | 40.0% | 9.76s |
| Zero-Shot CoT | 49.0% | 10.48s |
| Few-Shot CoT | 54.0% | 8.83s |

These results corroborate the analysis in the [Chain of Thought](chain-of-thought.md#chain-of-thought) chapter: chain-of-thought improves reasoning by decomposing complex problems, activating relevant knowledge, and providing opportunities for error correction. At the same time, we can observe that the CoT improvement for the 0.8B model is limited (+9% and +14%), which is consistent with the research finding that "larger models benefit more from CoT."

## Phase 2: Test-Time Scaling Strategies

In the test-time scaling laws chapter, we saw that investing more computation at inference time can systematically improve model performance. This phase implements two scaling strategies: Best-of-N sampling and self-consistency voting, and validates the reasoning decay model. [Best-of-N sampling](test-time-compute.md#best-of-n-sampling) is the simplest test-time scaling strategy: it generates N candidate answers for the same question and selects the best one. When the scoring function is "majority voting," Best-of-N reduces to [self-consistency](test-time-compute.md#verification-and-self-correction). This experiment compares accuracy changes across N=1, 2, 4, 8 sampling counts and observes the trade-off between computation and accuracy.

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import torch
from collections import Counter
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== Configuration ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_samples = 200
n_values = [1, 2, 4, 8]  # Sampling counts
max_new_tokens = 1024

# ========== Load ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_samples]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Extract from non-calculation lines at the end (avoid extracting intermediate step numbers)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if '=' in line and not re.match(r'^-?[\d,]+\.?\d*$', line):
            continue
        nums = re.findall(r'-?\d+\.?\d*', line.replace(',', ''))
        if nums:
            return nums[-1]
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

def generate_response(question):
    messages = [{"role": "user", "content": question + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# ========== Best-of-N Evaluation ==========
# First generate max(N) candidate answers for each question, then evaluate for different N values
max_n = max(n_values)

progress = ProgressReporter(total_steps=num_samples, description="Generating candidate answers")
all_candidates = []  # all_candidates[i] = [answer1, answer2, ...]

for i, item in enumerate(questions):
    ref_answer = extract_answer(item['answer'])
    candidates = []
    for _ in range(max_n):
        try:
            response = generate_response(item['question'])
            pred = extract_answer(response)
            candidates.append(pred)
        except:
            candidates.append(None)
    all_candidates.append({"candidates": candidates, "ref": ref_answer})
    progress.update(i + 1, message=f"Question {i+1}/{num_samples}")

# ========== Calculate Accuracy for Different N Values ==========
print("\n" + "="*60)
print("Best-of-N Sampling vs Self-Consistency Voting")
print("="*60)

scaling_results = {}

for n in n_values:
    # Self-consistency voting: take the most frequent answer among N candidates
    sc_correct = 0
    # Best-of-N (random selection): first correct answer among N candidates
    bon_correct = 0
    total_compute = 0  # Total generations

    for item_data in all_candidates:
        candidates_n = item_data["candidates"][:n]
        ref = item_data["ref"]

        # Self-consistency voting: majority voting
        valid_answers = [c for c in candidates_n if c is not None]
        if valid_answers:
            counter = Counter(valid_answers)
            majority_answer = counter.most_common(1)[0][0]
            if majority_answer == ref:
                sc_correct += 1

        # Best-of-N (random): probability of at least one correct answer in N samples
        if ref in [c for c in candidates_n if c is not None]:
            bon_correct += 1

        total_compute += n

    sc_accuracy = sc_correct / num_samples * 100
    bon_accuracy = bon_correct / num_samples * 100

    scaling_results[n] = {
        "sc_accuracy": sc_accuracy,
        "bon_accuracy": bon_accuracy,
        "total_compute": total_compute
    }
    print(f"\nN = {n}:")
    print(f"  Self-consistency voting accuracy: {sc_accuracy:.1f}%")
    print(f"  At least one correct probability: {bon_accuracy:.1f}%")
    print(f"  Total generations: {total_compute}")

# ========== Validate Reasoning Decay Model ==========
# a(n) = a0 + (amax - a0) * (1 - exp(-k*n))
# Fit this model to the self-consistency voting accuracy
import math

a0 = scaling_results[1]["sc_accuracy"]
amax = max(r["sc_accuracy"] for r in scaling_results.values())

# Grid search for optimal k
best_k = 0.1
best_error = float('inf')
for k in [i * 0.01 for i in range(1, 200)]:
    error = 0
    for n in n_values:
        predicted = a0 + (amax - a0) * (1 - math.exp(-k * n))
        error += (predicted - scaling_results[n]["sc_accuracy"]) ** 2
    if error < best_error:
        best_error = error
        best_k = k

print(f"\nReasoning decay model fitting:")
print(f"  a0 = {a0:.1f}% (accuracy at N=1)")
print(f"  amax = {amax:.1f}% (highest observed accuracy)")
print(f"  k = {best_k:.2f} (inference efficiency coefficient)")
print(f"  Fitted formula: a(n) = {a0:.1f} + ({amax:.1f} - {a0:.1f}) x (1 - e^(-{best_k:.2f}xn))")

# Print comparison of fitted vs actual values
print(f"\n{'N':>4} {'Actual':>8} {'Fitted':>8} {'Error':>8}")
print("-" * 32)
for n in n_values:
    actual = scaling_results[n]["sc_accuracy"]
    predicted = a0 + (amax - a0) * (1 - math.exp(-best_k * n))
    print(f"{n:>4} {actual:>7.1f}% {predicted:>7.1f}% {abs(actual - predicted):>7.1f}%")

progress.complete(message="Test-time scaling strategy evaluation complete")
```

After running the code above, you can observe the following patterns:

- Self-consistency voting accuracy improves as N increases, but the growth rate gradually slows, which is exactly the diminishing marginal returns phenomenon described by the [reasoning decay model](test-time-compute.md#reasoning-decay-model).
- The "at least one correct" probability grows faster, because it only takes one correct answer out of N samples. However, this does not mean we can directly use that correct sample; in practice, we don't know which answer is correct and still need a scoring function (such as majority voting) to select.
- The fitted reasoning decay curve $a(n) = a_0 + (a_{\max} - a_0)(1 - e^{-kn})$ matches the actual data reasonably well, providing preliminary experimental support for the quantitative laws of test-time scaling.

## Phase 3: Inference Efficiency Optimization

In the [Inference Efficiency Optimization](inference-efficiency.md) chapter, we saw that the essence of inference efficiency lies in finding an engineering-feasible balance between "answering well" and "answering quickly." This phase explores specific techniques from three directions: quantization, KV Cache measurement, and speculative decoding.

### Quantization Comparison

The [Model Lightweighting](inference-efficiency.md#model-lightweighting) section mentions that quantization reduces model size and accelerates inference by lowering numerical precision. However, the actual speedup depends on whether the underlying operators support low-precision matrix multiplication. The bitsandbytes quantization used in this experiment employs online dequantization: weights are stored in INT8/INT4 format and dynamically dequantized to FP16 at inference time before computation. The main benefit of this approach is reduced GPU memory usage (enabling larger models to run on limited memory), rather than inference speedup. During the experiment setup phase, we already quantized Qwen3.5-0.8B to INT8 and INT4 and saved them to disk. In this section, we directly load models at three precisions and compare disk size, GPU memory usage, inference speed, and quality.

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import warnings
import torch
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# Suppress bitsandbytes quantization precision conversion warnings (bfloat16 inputs are cast to float16 during INT8 inference)
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
# Suppress PyTorch internal API deprecation warnings (torch._check_is_size used by bitsandbytes will be removed)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dmla_progress import ProgressReporter

# ========== Configuration ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
int8_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int8')
int4_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int4')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_eval = 50  # Use fewer questions for quantization evaluation to save time

# ========== Load Data ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_eval]

tokenizer = AutoTokenizer.from_pretrained(model_path)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Extract from non-calculation lines at the end (avoid extracting intermediate step numbers)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if '=' in line and not re.match(r'^-?[\d,]+\.?\d*$', line):
            continue
        nums = re.findall(r'-?\d+\.?\d*', line.replace(',', ''))
        if nums:
            return nums[-1]
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== Weight File Size Calculation ==========
def get_model_file_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.startswith('model') and f.endswith('.safetensors'):
                total += os.path.getsize(os.path.join(dirpath, f))
    return total / 1024**3

# ========== Evaluation Function ==========
def evaluate_model(model, questions, label, progress, progress_offset):
    model.eval()
    correct = 0
    total_time = 0
    total_tokens = 0

    for i, item in enumerate(questions):
        messages = [{"role": "user", "content": item['question'] + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        ref = extract_answer(item['answer'])
        if pred == ref:
            correct += 1
        total_time += elapsed
        total_tokens += outputs.shape[-1] - inputs["input_ids"].shape[-1]

        progress.update(
            progress_offset + i + 1,
            message=f"{label}: {i+1}/{len(questions)}, current accuracy {correct/(i+1)*100:.1f}%"
        )

    accuracy = correct / len(questions) * 100
    avg_time = total_time / len(questions)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    return {"accuracy": accuracy, "avg_time": avg_time, "tokens_per_sec": tokens_per_sec}

# ========== Three Precision Comparison ==========
quant_results = {}
total_eval_steps = 3 * num_eval
progress = ProgressReporter(total_steps=total_eval_steps, description="Quantization precision comparison")

# FP16 (baseline)
progress.update(1, message="Loading FP16 model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
fp16_vram = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1024**3
fp16_weight = get_model_file_size(model_path)
print(f"\nFP16 GPU memory: {fp16_vram:.2f} GB, Weight file: {fp16_weight:.2f} GB")
quant_results["FP16"] = evaluate_model(model_fp16, questions, "FP16", progress, 0)
quant_results["FP16"]["vram_gb"] = fp16_vram
quant_results["FP16"]["weight_gb"] = fp16_weight
del model_fp16
torch.cuda.empty_cache()

# INT8 quantization (load saved quantized model)
progress.update(num_eval + 1, message="Loading INT8 model...")
model_int8 = AutoModelForCausalLM.from_pretrained(int8_path, device_map="auto")
int8_vram = sum(p.numel() * p.element_size() for p in model_int8.parameters()) / 1024**3
int8_weight = get_model_file_size(int8_path)
print(f"INT8 GPU memory: {int8_vram:.2f} GB, Weight file: {int8_weight:.2f} GB")
quant_results["INT8"] = evaluate_model(model_int8, questions, "INT8", progress, num_eval)
quant_results["INT8"]["vram_gb"] = int8_vram
quant_results["INT8"]["weight_gb"] = int8_weight
del model_int8
torch.cuda.empty_cache()

# INT4 quantization (load saved quantized model)
progress.update(2 * num_eval + 1, message="Loading INT4 model...")
model_int4 = AutoModelForCausalLM.from_pretrained(int4_path, device_map="auto")
int4_vram = sum(p.numel() * p.element_size() for p in model_int4.parameters()) / 1024**3
int4_weight = get_model_file_size(int4_path)
print(f"INT4 GPU memory: {int4_vram:.2f} GB, Weight file: {int4_weight:.2f} GB")
quant_results["INT4"] = evaluate_model(model_int4, questions, "INT4", progress, 2 * num_eval)
quant_results["INT4"]["vram_gb"] = int4_vram
quant_results["INT4"]["weight_gb"] = int4_weight
del model_int4
torch.cuda.empty_cache()

# ========== Summary Results ==========
print("\n" + "="*80)
print("Quantization Precision Comparison")
print("="*80)
print(f"{'Precision':<8} {'Weight':>10} {'GPU Memory':>10} {'Accuracy':>8} {'Avg Time':>10} {'Speed':>14}")
print("-"*80)
for name, r in quant_results.items():
    print(f"{name:<8} {r['weight_gb']:>9.2f}GB {r['vram_gb']:>9.2f}GB {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['tokens_per_sec']:>10.1f} tok/s")

# Compression ratio and quality loss
print(f"\nCompression ratio (relative to FP16):")
print(f"  INT8 weight: {int8_weight/fp16_weight*100:.1f}%, GPU memory: {int8_vram/fp16_vram*100:.1f}%")
print(f"  INT4 weight: {int4_weight/fp16_weight*100:.1f}%, GPU memory: {int4_vram/fp16_vram*100:.1f}%")
print(f"\nQuality loss (relative to FP16):")
fp16_acc = quant_results["FP16"]["accuracy"]
print(f"  INT8: {quant_results['INT8']['accuracy'] - fp16_acc:+.1f}%")
print(f"  INT4: {quant_results['INT4']['accuracy'] - fp16_acc:+.1f}%")

progress.complete(message="Quantization precision comparison complete")
```

Below are the actual quantization comparison results for Qwen3.5-0.8B-Instruct on the GSM8K evaluation subset (50 questions):

| Precision | Weight File | GPU Memory | Accuracy | Avg Time | Speed |
|-----------|------------|------------|----------|----------|-------|
| FP16 | 1.63 GB | 1.40 GB | 52.0% | 8.41s | 41.3 tok/s |
| INT8 | 0.94 GB | 0.94 GB | 48.0% | 31.05s | 11.5 tok/s |
| INT4 | 0.74 GB | 0.71 GB | 32.0% | 8.12s | 31.2 tok/s |

The compression ratios (relative to FP16) are INT8 weight 57.7%, GPU memory 66.9%; INT4 weight 45.2%, GPU memory 50.4%. Quality losses (relative to FP16) are INT8 -4.0%, INT4 -20.0%. The results reveal the following:

- **Inference actually becomes slower after quantization**: This seems counterintuitive but is fully consistent with bitsandbytes' implementation. bitsandbytes' LLM.int8() and Q4_K quantization are both online dequantization schemes. Weights are stored at low precision, and at each forward pass during inference, the quantized weights are dynamically dequantized to FP16 before performing matrix multiplication. Dequantization itself is extra computation, and for a 0.8B model where FP16 weights already fit entirely in GPU memory, the memory bandwidth savings from quantization are negligible, while the dequantization overhead actually slows down inference. INT8 is slower than INT4 (11.5 vs 31.2 tok/s) because INT8's dequantization path involves more complex block-wise matrix multiplication and outlier decomposition, whereas INT4's double quantization produces a more compact structure with lower dequantization overhead.
- **The true value of quantization lies in reducing GPU memory**: INT8 reduces memory to 67% of FP16, INT4 to 50%. For a small model like 0.8B, memory savings are not significant, but for 7B or 70B scale models, INT4 quantization can reduce memory requirements from 140GB to 35GB, making single-GPU inference feasible. This is the practical engineering application of quantization.
- **Accuracy loss increases with quantization granularity**: INT8 loses only 4%, while INT4 loses 20%. A 0.8B model has limited parameters to begin with, and INT4's aggressive compression directly damages the model's representational capacity. For larger models (7B+), INT4 accuracy loss is typically within 1-3% because larger models have more redundant parameters to absorb quantization errors.

::: info True Quantization Speedup

To achieve inference acceleration after quantization, you need an inference engine that supports low-precision matrix multiplication, such as llama.cpp (GGUF format, CPU/GPU hybrid inference), vLLM (AWQ/GPTQ format, GPU inference), or TensorRT-LLM (INT8/INT8 Tensor Core operators). These engines directly perform INT8 or INT4 matrix multiplication at the kernel level without requiring dequantization steps, thus truly converting the memory savings from quantization into speed improvements.

:::

### KV Cache Memory Measurement

The [Inference Bottleneck Analysis](inference-efficiency.md#inference-bottleneck-analysis) section provides an estimation formula for KV Cache memory usage. This section verifies the accuracy of this formula through measurement and observes the impact of sequence length on memory usage.

$$M_{\text{KV}} = 2 \times n_{\text{layer}} \times d_{\text{head}} \times n_{\text{head}} \times n_{\text{max}} \times b \times sizeof(\text{dtype})$$

```python runnable gpuonly
import os
import logging
import torch
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

# Extract model architecture parameters
config = model.config
n_layer = config.num_hidden_layers
n_head = config.num_attention_heads
n_kv_head = getattr(config, 'num_key_value_heads', n_head)  # Under GQA, KV heads may be fewer than Q heads
d_head = config.hidden_size // n_head  # Dimension per head
hidden_size = config.hidden_size

print(f"Model architecture parameters:")
print(f"  Layers (n_layer): {n_layer}")
print(f"  Attention heads (n_head): {n_head}")
print(f"  KV heads (n_kv_head): {n_kv_head}")
print(f"  Dimension per head (d_head): {d_head}")
print(f"  Hidden dimension (hidden_size): {hidden_size}")

# Measure KV Cache size at different sequence lengths
test_lengths = [128, 256, 512, 1024, 2048]
print(f"\n{'Seq Length':>10} {'Formula Est.':>12} {'KV Cache Meas.':>14} {'Error':>8}")
print("-"*50)

# Count standard Attention layers (FLA layers do not produce KV Cache)
n_attn_layers = sum(
    1 for layer in model.model.layers
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'k_proj')
)
print(f"Standard Attention layers: {n_attn_layers}/{n_layer} (remaining are FLA layers, no KV Cache)\n")

for seq_len in test_lengths:
    # Construct input
    input_ids = torch.randint(0, len(tokenizer), (1, seq_len), device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)

    # Measure KV Cache tensor size per layer, skip FLA layers (no keys/values attributes)
    kv_bytes = 0
    for layer in outputs.past_key_values.layers:
        if hasattr(layer, 'keys') and layer.keys is not None:
            kv_bytes += layer.keys.nelement() * layer.keys.element_size()
        if hasattr(layer, 'values') and layer.values is not None:
            kv_bytes += layer.values.nelement() * layer.values.element_size()
    measured_mb = kv_bytes / 1024**2

    # Formula estimate: only standard Attention layers produce KV Cache
    dtype_size = 2  # bfloat16 = 2 bytes
    estimated_bytes = 2 * n_attn_layers * d_head * n_kv_head * seq_len * 1 * dtype_size
    estimated_mb = estimated_bytes / 1024**2

    error_pct = abs(measured_mb - estimated_mb) / max(estimated_mb, 0.01) * 100

    print(f"{seq_len:>10} {estimated_mb:>10.1f}MB {measured_mb:>12.1f}MB {error_pct:>6.1f}%")

    # Cleanup
    del outputs, input_ids
    torch.cuda.empty_cache()
```

After running the code above, the formula estimates should closely match the measured values (error within 1%), verifying the accuracy of the KV Cache memory estimation formula. Qwen3.5-0.8B has two notable structural features:

- **[GQA](../architecture-basics/architecture-evolution.md#gqa-grouped-query-attention)** (Grouped-Query Attention): `n_kv_head = 2`, much smaller than `n_head = 8`. This means every 4 Query heads share one set of KV heads, and the KV Cache memory usage is only 1/4 (2/8) of standard MHA. This is a key advantage of GQA, significantly reducing KV Cache memory requirements without notably affecting model quality.
- **[FLA](../architecture-basics/architecture-evolution.md#linear-attention) Hybrid Architecture** (Flash Linear Attention): Some layers of Qwen3.5 use linear attention and do not produce KV Cache. Only standard Attention layers have KV Cache, so the layer count in the formula should be `n_attn_layers` rather than `n_layer`.

From the measurements, we can also observe that KV Cache size scales linearly with sequence length: doubling the sequence length doubles the corresponding KV Cache. This means that in long-text inference scenarios, KV Cache memory growth can easily become a bottleneck. GQA and FLA are designed precisely to alleviate this bottleneck: GQA reduces KV Cache to `n_kv_head / n_head` of MHA, while FLA layers eliminate KV Cache entirely. These practices align with the theoretical descriptions of attention mechanism improvements in [Transformer Evolution and Variants](../architecture-basics/architecture-evolution.md).

### Speculative Decoding

The design philosophy of [speculative decoding](inference-efficiency.md#speculative-decoding) is to use a small model (Draft Model) to quickly generate candidate tokens, and then use the large model (Target Model) to verify them in a single forward pass. The [Inference Efficiency Optimization](inference-efficiency.md#speculative-decoding) chapter introduced a framework called Medusa that does not require a separate Draft Model, but instead adds multiple prediction heads on top of the model's last hidden layer. This section trains Medusa heads and measures the actual speedup from speculative decoding.

Note that the experiment's purpose is educational; in production, there is no reason to apply speculative decoding to a model as small as Qwen3.5-0.8B — it would be pointless or even counterproductive. The principle of speculative decoding speedup is that the decode phase is memory bandwidth-bound: each forward pass requires moving all model parameters from GPU memory to the compute units but only generates a single token, resulting in low compute utilization. In this scenario, speculative decoding uses 2 forward passes (Draft + Verify) to generate multiple tokens; as long as the speculation acceptance rate is high enough, the time of 2 forward passes is less than the time to generate the same number of tokens one by one. However, for a 0.8B model with only 1.6GB of parameters, the memory bandwidth bottleneck is not severe. In this case, speculative decoding's 2 forward passes are pure overhead, and each verification step requires a full forward computation over increasingly long sequences, making it actually slower than autoregressive decoding. The speedup from speculative decoding becomes significant as model size increases, typically requiring at least a 7B/13B model before the savings in decoding steps outweigh the additional verification overhead.

In the original Medusa paper, each head predicts the next token (position t+1) but provides diverse candidates through different probability distributions, using a tree attention mechanism to verify multiple candidate paths simultaneously. To reduce implementation complexity, this experiment adopts a simplified approach where each head predicts a future token at a different position (Head_k predicts t+k+1), and each candidate is verified sequentially against the target model.

The Medusa head structure consists of a residual block (ResBlock) followed by an output layer. The residual block uses a bottleneck structure (Hidden Size → Bottleneck → Hidden Size) to reduce parameter count, and the output layer uses low-rank decomposition (Hidden Size → Rank → Vocab Size) to avoid parameter explosion from a large vocabulary. The down-projection weights between layers use [He initialization](../../deep-learning/neural-network-stability/weight-initialization.md#he-initialization) to ensure gradient flow, while the up-projection weights are zero-initialized so they do not affect the backbone model during early training.

```python runnable gpuonly timeout=unlimited
import os
import json
import logging
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Suppress Qwen3.5 model's FLA acceleration library missing warnings (does not affect functionality, only falls back to pure PyTorch implementation)
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# Suppress bitsandbytes quantization precision conversion warnings and PyTorch internal API deprecation warnings
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== Medusa Head Definition ==========
class ResBlock(nn.Module):
    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return x + self.up(nn.functional.silu(self.down(x)))

class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bottleneck=256, rank=128):
        super().__init__()
        self.res_block = ResBlock(hidden_size, bottleneck)
        self.out_down = nn.Linear(hidden_size, rank, bias=False)
        self.out_up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.kaiming_uniform_(self.out_down.weight, a=5**0.5)
        nn.init.zeros_(self.out_up.weight)
    def forward(self, x):
        return self.out_up(nn.functional.silu(self.out_down(self.res_block(x))))

# ========== Load Model ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
base_model.eval()

# Create 4 Medusa heads
# Head_k at position t predicts the token at position t+k+1
# Head_0 predicts t+1 (same position as the backbone lm_head), Head_1 predicts t+2, and so on
hidden_size = base_model.config.hidden_size
vocab_size = base_model.config.vocab_size
medusa_heads = nn.ModuleList([
    MedusaHead(hidden_size, vocab_size).to(dtype=torch.bfloat16, device="cuda")
    for _ in range(4)
])

# Freeze backbone, only train Medusa heads
for param in base_model.parameters():
    param.requires_grad = False
trainable_params = sum(p.numel() for p in medusa_heads.parameters())
print(f"Backbone model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
print(f"Medusa head parameters: {trainable_params:,}")

# ========== Training Data ==========
class GSM8KTrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.encoded = []
        for item in self.data:
            text = f"User: {item['question']}\nAssistant: {item['answer']}"
            enc = tokenizer(text, truncation=True, max_length=max_length,
                          padding='max_length', return_tensors='pt')
            self.encoded.append(enc['input_ids'].squeeze(0))
    def __len__(self):
        return len(self.encoded)
    def __getitem__(self, idx):
        ids = self.encoded[idx]
        return ids, ids.clone()

train_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-train.jsonl')
dataset = GSM8KTrainDataset(train_path, tokenizer, max_length=128)
loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

# ========== Training (Medusa-1: freeze backbone, train heads only) ==========
optimizer = torch.optim.AdamW(
    [p for p in medusa_heads.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=0.0
)

num_steps = 1000
progress = ProgressReporter(total_steps=num_steps, description="Training Medusa heads")
step = 0
total_loss = 0.0

for epoch in range(100):
    for input_ids, _ in loader:
        if step >= num_steps:
            break
        input_ids = input_ids.to("cuda")
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        loss = 0.0
        for k, head in enumerate(medusa_heads):
            head_logits = head(hidden_states)
            target = input_ids[:, k + 1:]
            pred = head_logits[:, :target.shape[1]]
            if target.shape[1] > 0:
                loss += nn.functional.cross_entropy(
                    pred.reshape(-1, pred.shape[-1]),
                    target.reshape(-1)
                )
        loss = loss / len(medusa_heads)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1
        progress.update(step, message=f"Step {step}/{num_steps}, Loss={loss.item():.4f}")
        if step >= num_steps:
            break
    if step >= num_steps:
        break

progress.complete(message=f"Training complete, {num_steps} steps, Avg Loss={total_loss/num_steps:.4f}")

# ========== Evaluate Each Head's Accuracy ==========
eval_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
with open(eval_path, 'r', encoding='utf-8') as f:
    eval_questions = [json.loads(line) for line in f][:100]

head_correct = [0] * 4
head_total = [0] * 4
base_model.eval()

for item in eval_questions:
    text = f"User: {item['question']}\nAssistant: {item['answer']}"
    enc = tokenizer(text, truncation=True, max_length=256, return_tensors='pt')
    input_ids = enc['input_ids'].to("cuda")
    with torch.no_grad():
        outputs = base_model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    for k, head in enumerate(medusa_heads):
        head_logits = head(hidden_states)
        target = input_ids[:, k + 1:]
        pred = head_logits[:, :target.shape[1]].argmax(dim=-1)
        if target.shape[1] > 0:
            head_correct[k] += (pred == target).sum().item()
            head_total[k] += target.numel()

print(f"\nMedusa head accuracy (Top-1, 100-question evaluation set):")
print(f"{'Head':>6} {'Position':>10} {'Accuracy':>10}")
print("-" * 30)
for k in range(4):
    acc = head_correct[k] / max(head_total[k], 1) * 100
    print(f"Head_{k:>2} {'t+' + str(k+1):>10} {acc:>9.1f}%")

# ========== Speculative Decoding Evaluation ==========
def autoregressive_generate(model, tokenizer, prompt, max_new_tokens=200):
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    generated = []
    past_kv = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_tok)
            input_ids = torch.tensor([[next_tok]], device=model.device)
            if next_tok == tokenizer.eos_token_id:
                break
    return generated

def speculative_generate(model, heads, tokenizer, prompt, max_new_tokens=200, topk=5):
    """Simplified Medusa speculative decoding (top-k candidates)

    Each head takes top-k candidates instead of just top-1 to improve matching probability.
    Verification checks if the target model's argmax falls within the candidate's top-k list.
    """
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    current_ids = prompt_ids.clone()
    generated = []
    accepted_total = 0
    speculated_total = 0
    step_count = 0
    with torch.no_grad():
        while len(generated) < max_new_tokens:
            step_count += 1
            outputs = model(input_ids=current_ids, use_cache=False, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            base_next = outputs.logits[:, -1, :].argmax(dim=-1).item()
            # Each head takes top-k candidates, use top-1 to construct the verification sequence
            head_topk = [heads[k](hidden[:, -1:, :])[:, 0, :].topk(topk, dim=-1).indices[0].tolist()
                         for k in range(1, len(heads))]
            spec_tokens = [base_next] + [hk[0] for hk in head_topk]
            speculated_total += len(head_topk)
            verify_ids = torch.cat([current_ids, torch.tensor([spec_tokens], device=model.device)], dim=1)
            vout = model(input_ids=verify_ids, use_cache=False)
            orig_len = current_ids.shape[1]
            accepted = 0
            generated.append(base_next)
            for k, topk_list in enumerate(head_topk):
                verified = vout.logits[0, orig_len + k, :].argmax().item()
                if verified in topk_list:
                    generated.append(verified)
                    accepted += 1
                else:
                    generated.append(verified)
                    break
            else:
                bonus = vout.logits[0, orig_len + len(head_topk), :].argmax().item()
                generated.append(bonus)
            accepted_total += accepted
            new_ids = torch.tensor([generated], device=model.device)
            current_ids = torch.cat([prompt_ids, new_ids], dim=1)
            if any(t == tokenizer.eos_token_id for t in generated[-(1+accepted):]):
                break
    hit_rate = accepted_total / max(speculated_total, 1) * 100
    return generated, step_count, hit_rate

num_bench = 20
# Autoregressive decoding
ar_total_time = 0
ar_total_tokens = 0
for item in eval_questions[:num_bench]:
    prompt = f"User: {item['question']}\nPlease think step by step.\nAssistant:"
    start = time.time()
    toks = autoregressive_generate(base_model, tokenizer, prompt, max_new_tokens=200)
    ar_total_time += time.time() - start
    ar_total_tokens += len(toks)

# Medusa speculative decoding (top-5 candidates)
spec_total_time = 0
spec_total_tokens = 0
spec_hit_rates = []
spec_steps_list = []
for item in eval_questions[:num_bench]:
    prompt = f"User: {item['question']}\nPlease think step by step.\nAssistant:"
    start = time.time()
    toks, steps, hit_rate = speculative_generate(
        base_model, medusa_heads, tokenizer, prompt, max_new_tokens=200, topk=5
    )
    spec_total_time += time.time() - start
    spec_total_tokens += len(toks)
    spec_hit_rates.append(hit_rate)
    spec_steps_list.append(steps)

# Calculate top-1 vs top-5 head accuracy comparison
topk_acc = {}
for tk in [1, 5]:
    tk_correct = [0] * 4; tk_total = [0] * 4
    for item in eval_questions[:50]:
        text = f"User: {item['question']}\nAssistant: {item['answer']}"
        enc = tokenizer(text, truncation=True, max_length=256, return_tensors='pt')
        input_ids = enc['input_ids'].to("cuda")
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        for k, head in enumerate(medusa_heads):
            head_logits = head(hidden_states)
            target = input_ids[:, k + 1:]
            pred_logits = head_logits[:, :target.shape[1]]
            if target.shape[1] > 0:
                topk_preds = pred_logits.topk(tk, dim=-1).indices
                target_exp = target.unsqueeze(-1).expand_as(topk_preds)
                tk_correct[k] += (topk_preds == target_exp).any(dim=-1).sum().item()
                tk_total[k] += target.numel()
    topk_acc[tk] = [tk_correct[k] / max(tk_total[k], 1) * 100 for k in range(4)]

ar_tps = ar_total_tokens / ar_total_time
spec_tps = spec_total_tokens / spec_total_time
speedup = ar_total_time / spec_total_time
avg_hit = sum(spec_hit_rates) / len(spec_hit_rates)

print(f"\nMedusa head accuracy comparison (Top-1 vs Top-5):")
print(f"{'Head':>6} {'Position':>10} {'Top-1':>8} {'Top-5':>8}")
print("-" * 36)
for k in range(4):
    print(f"Head_{k:>2} {'t+' + str(k+1):>10} {topk_acc[1][k]:>7.1f}% {topk_acc[5][k]:>7.1f}%")

print(f"\nDecoding speed comparison ({num_bench} questions, max 200 tokens, greedy decoding, top-5 candidates):")
print(f"{'Method':<20} {'Speed':>12} {'Avg Time':>10} {'Speedup':>8} {'Hit Rate':>8}")
print("-" * 62)
print(f"{'Autoregressive':<20} {ar_tps:>10.1f} tok/s {ar_total_time/num_bench:>8.2f}s {'1.00x':>8} {'-':>8}")
print(f"{'Medusa Speculative':<20} {spec_tps:>10.1f} tok/s {spec_total_time/num_bench:>8.2f}s {speedup:>6.2f}x {avg_hit:>6.1f}%")
```

After running the code above, the Medusa head accuracy decreases as the prediction position moves further away, which is expected: tokens further into the future are harder to predict. Using top-5 candidates (each head takes the 5 most probable tokens) shows significant improvement over top-1. Below are the experimental results:

| Head | Position | Top-1 Accuracy | Top-5 Accuracy |
|------|----------|----------------|----------------|
| Head_0 | t+1 | 60.8% | 77.6% |
| Head_1 | t+2 | 44.8% | 64.2% |
| Head_2 | t+3 | 27.9% | 53.1% |
| Head_3 | t+4 | 21.1% | 46.1% |

| Decoding Method | Speed | Speedup | Speculation Hit Rate |
|-----------------|-------|---------|---------------------|
| Autoregressive | 40.3 tok/s | 1.00x | - |
| Medusa Speculative (top-5) | 10.5 tok/s | 0.26x | 14.4% |

Note that the "speculation hit rate" in the table refers to the number of speculated tokens verified as correct by the target model divided by the total number of speculated tokens. This value decreases as the speculation length $\gamma$ increases, measuring around 14% in practice. This metric is not the same as the 50-85% speculation acceptance rate commonly cited in the industry. The per-token acceptance rate $\alpha$ refers to the probability that each token generated by the Draft Model is independently accepted. In traditional speculative decoding, the Draft Model generates candidates token by token, and each token is accepted with probability $\alpha = \min\!\left(1,\;\frac{p(x)}{q(x)}\right)$ (where $p$ and $q$ are the probability distributions of the target and draft models, respectively). $\alpha$ depends on how well the draft model's distribution aligns with the target model's, and is independent of the speculation length $\gamma$.

## Experiment Conclusions

This experiment used the Qwen3.5-0.8B-Instruct model on the GSM8K evaluation set to complete a full-chain practice from chain-of-thought prompting to reasoning scaling and efficiency optimization. The experiment validated the following conclusions:

- **Chain-of-thought prompting** is effective but limited by model scale. For the 0.8B model, CoT does show measurable improvement, but the effect is relatively modest, corroborating the research finding that "larger models benefit more from CoT."
- **Test-time scaling** (Best-of-N, self-consistency voting) can systematically improve accuracy, with growth following the diminishing marginal returns pattern of the reasoning decay model. Dynamic inference depth can adaptively allocate computational resources based on problem difficulty, saving computation while maintaining accuracy.
- **Inference efficiency optimization** (quantization, KV Cache management, speculative decoding) involves trade-offs between "answering well" and "answering quickly." In this experiment, each optimization measure came with different engineering decisions; without understanding the underlying principles, one cannot correctly choose the right application scenario and may even produce counterproductive results.

These three techniques are not isolated; they are often used in combination. A quantized small model, combined with dynamic inference depth and self-consistency voting, can achieve better reasoning results under limited computational resources. This also echoes the unified logic of the [three scaling laws](test-time-compute.md#a-unified-view-of-three-scaling-laws): pretraining determines the upper bound, post-training makes capabilities usable, test-time scaling realizes the potential, and inference efficiency optimization determines whether the entire system can be deployed in engineering practice.
