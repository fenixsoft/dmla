# LLM 推理策略与效率优化实验

在前几章中，我们从理论层面了解了思维链、推理时缩放和推理效率优化。本节我们会转入实践，体会推理策略的工程抉择。[此前实验](../architecture-basics/llm-pretrain-experiment.md)中训练的 64M 参数模型对于思维链和推理缩放来说规模过小，本次实验将使用开源的 [Qwen3.5-0.8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) 模型，实现从思维链提示到推理缩放再到效率优化的全链路实践。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../appendixes/sandbox.md#数据管理)并下载好 GSM8K 评测子集和 Qwen3.5-0.8B-Instruct 模型。你可以通过 `DMLA-CLI` 工具分别完成这两项工作：

```bash
# 下载数据集：选择 "下载数据集" -> 选择 "GSM8K 200 (数学推理评测集)"
dmla data

# 下载模型：选择 "下载模型" -> 选择 "Qwen3.5-0.8B-Instruct"
dmla model
```

GSM8K（Grade School Math 8K）是一个包含 7473 道训练题和 1319 道测试题的小学数学应用题推理基准，模型需要通过多步计算才能得出正确答案。本实验从中随机抽取了 200 道题作为评测子集，在有限算力下尽可能体现不同推理策略的性能差异趋势，同时将评测时间控制在合理范围内。数据集下载完成后，运行以下代码验证数据和模型加载是否正常，并预先量化保存 INT8 和 INT4 模型供[第三阶段（推理效率优化）](#第三阶段：推理效率优化)使用：

```python runnable gpuonly timeout=unlimited
import os
import json
import logging
import warnings
import torch
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# 抑制 bitsandbytes 量化精度转换提示和 PyTorch 内部 API 废弃提示
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dmla_progress import ProgressReporter

# ========== 验证数据集 ==========
gsm8k_dir = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
if os.path.exists(gsm8k_dir):
    with open(os.path.join(gsm8k_dir, 'gsm8k_200.jsonl'), 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    print(f"GSM8K 评测子集: {len(questions)} 题")
    print(f"示例问题: {questions[0]['question'][:80]}...")
else:
    print("GSM8K 评测子集: 未下载，请运行 'dmla data' 下载数据集")

# ========== 加载模型 ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
print(f"\n正在加载模型 {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e9:.2f}B ({total_params:,})")
print(f"模型 dtype: {model.dtype}")
print(f"设备: {model.device}")

# ========== 量化模型保存 ==========
progress = ProgressReporter(total_steps=2, description="量化模型保存")

# INT8 量化并保存
int8_save_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int8')
if os.path.exists(int8_save_path):
    print(f"\nINT8 量化模型已存在，跳过: {int8_save_path}")
else:
    progress.update(1, message="量化 INT8 模型...")
    int8_config = BitsAndBytesConfig(load_in_8bit=True)
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=int8_config, device_map="auto"
    )
    progress.update(1, message="保存 INT8 模型到磁盘...")
    model_int8.save_pretrained(int8_save_path, safe_serialization=True)
    tokenizer.save_pretrained(int8_save_path)
    del model_int8
    print(f"\nINT8 量化模型已保存: {int8_save_path}")

torch.cuda.empty_cache()

# INT4 量化并保存
int4_save_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int4')
if os.path.exists(int4_save_path):
    print(f"INT4 量化模型已存在，跳过: {int4_save_path}")
else:
    progress.update(2, message="量化 INT4 模型...")
    int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model_int4 = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=int4_config, device_map="auto"
    )
    progress.update(2, message="保存 INT4 模型到磁盘...")
    model_int4.save_pretrained(int4_save_path, safe_serialization=True)
    tokenizer.save_pretrained(int4_save_path)
    del model_int4
    print(f"INT4 量化模型已保存: {int4_save_path}")

torch.cuda.empty_cache()

# 统计模型权重文件磁盘占用（仅计算 model.safetensors，排除 .git、tokenizer 等）
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

print(f"\n{'模型':<12} {'权重文件大小':>12}")
print("-" * 27)
print(f"{'FP16 (原始)':<12} {fp16_size:>11.2f}GB")
print(f"{'INT8':<12} {int8_size:>11.2f}GB")
print(f"{'INT4':<12} {int4_size:>11.2f}GB")
print(f"\nINT8 相比 FP16: {int8_size/fp16_size*100:.1f}%")
print(f"INT4 相比 FP16: {int4_size/fp16_size*100:.1f}%")

progress.complete(message="量化模型保存完成")
```

## 第一阶段：思维链与提示工程

[思维链与推理模型](chain-of-thought.md)一章中，我们看到思维链提示能让模型从"凭直觉给答案"转变为"按步骤做推导"。本阶段将在 GSM8K 评测集上，对比零样本直接回答、零样本 CoT、少样本 CoT 三种提示策略的效果差异，三种策略的区别在于提示词的设计：

- **零样本直接回答**（Zero-Shot Direct）：直接提出问题，模型给出答案，不附加任何推理引导。
- **零样本 CoT**（Zero-Shot CoT）：在问题后追加"请一步一步思考"（Let's think step by step）的引导语，让模型自主展示推理过程。这就是[思维链](chain-of-thought.md#思维链)一章中介绍的零样本思维链方法。
- **少样本 CoT**（Few-Shot CoT）：在问题前提供 2-3 个带有完整推理过程的示例，让模型学习如何组织推理步骤。

Qwen3.5 系列模型原生支持 Thinking/Non-Thinking 模式切换。然而在实践中，0.8B 参数量的轻量级模型开启 Thinking 模式后，模型倾向于生成关于"如何思考"的元描述（如"分析请求"、"分解问题"），而非直接执行数学推理，反而降低了解题质量。因此本实验统一使用 Non-Thinking 模式，通过提示词的差异来实现三种策略的对比。

::: info 推理耗时

尽管 Qwen3.5-0.8B-Instruct 模型已经十分轻量，但对三种提示词各完成 200 次推理（共 600 次）仍需耗费不少时间。在 RTX 5080 上运行约需要 90 分钟。

:::

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import torch
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_samples = 200  # 评测题数

# ========== 加载数据 ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_samples]
print(f"评测题目数: {len(questions)}")

# ========== 加载模型 ==========
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

# ========== 答案提取 ==========
def extract_answer(text):
    """从模型输出和GSM8K参考答案中提取最终数值"""
    # 1. GSM8K参考答案格式: "#### 123"
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # 2. Thinking模式：从</think>标签后的内容提取
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
    # 3. 从"答案是/answer is"等模式提取
    match = re.search(r'(?:答案是|answer\s+is|final\s+answer)\s*[=:：]?\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    # 4. 从末尾非计算行提取（避免提取中间步骤的数字）
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
    # 5. 兜底：取最后一个数字
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== 推理函数 ==========
def generate_response(messages, max_new_tokens=1024):
    """调用模型生成回答"""
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

# ========== Few-Shot 示例 ==========
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

# ========== 评测三种策略 ==========
strategies = {
    "零样本直接回答": lambda q: [{"role": "user", "content": q}],
    "零样本CoT": lambda q: [{"role": "user", "content": q + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}],
    "少样本CoT": lambda q: few_shot_examples + [{"role": "user", "content": q + "\nPlease think step by step, and give the final numerical answer in the format #### answer at the last line."}],
}

results = {}
progress = ProgressReporter(total_steps=len(strategies) * num_samples, description="思维链提示策略评测")

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
            print(f"  题目 {i+1} 生成失败: {e}")
        elapsed = time.time() - start_time

        pred_answer = extract_answer(response)
        is_correct = pred_answer == ref_answer if pred_answer and ref_answer else False

        correct += int(is_correct)
        total += 1
        total_time += elapsed
        total_tokens += len(tokenizer.encode(response))

        progress.update(
            len(results) * num_samples + i + 1,
            message=f"{strategy_name}: {i+1}/{num_samples}, 当前准确率 {correct/total*100:.1f}%"
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
    print(f"  准确率: {accuracy:.1f}% ({correct}/{total})")
    print(f"  平均耗时: {avg_time:.2f}s/题")
    print(f"  平均输出 token 数: {avg_tokens:.0f}")

# ========== 汇总结果 ==========
print("\n" + "="*60)
print("思维链提示策略对比")
print("="*60)
print(f"{'策略':<25} {'准确率':>8} {'平均耗时':>10} {'平均token数':>12}")
print("-"*60)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['avg_tokens']:>11.0f}")

progress.complete(message="思维链提示策略评测完成")
```

运行上方代码后，理论上应观察到以下规律：

- 零样本直接回答的准确率最低，因为模型没有展示推理过程，容易在多步计算中出错或遗漏关键信息。
- 零样本 CoT 准确率有明显提升，"请一步一步思考"的提示让模型展示了推理过程，减少了计算遗漏。
- 少样本 CoT 准确率最高，示例为模型提供了推理格式的模板，让模型知道应该如何组织推理步骤和标注最终答案。

以下是 Qwen3.5-0.8B-Instruct 在 GSM8K 评测子集上的实测结果：

| 策略 | 准确率 | 平均耗时 |
|------|--------|---------|
| 零样本直接回答 | 40.0% | 9.76s |
| 零样本 CoT | 49.0% | 10.48s |
| 少样本 CoT | 54.0% | 8.83s |

这些结果印证了[思维链](chain-of-thought.md#思维链)一章的分析：思维链通过分解复杂问题、激活相关知识和提供纠错机会来提升推理能力。同时也能观察到，0.8B 模型的 CoT 效果提升幅度有限（+9% 和 +14%），这与"模型规模越大 CoT 效果越明显"的研究结论一致。

## 第二阶段：推理时缩放策略

推理时缩放定律一章中，我们看到在推理阶段投入更多计算可以系统地提升模型性能。本阶段将实现 Best-of-N 采样和自一致性投票两种缩放策略，并验证推理衰减模型。[Best-of-N 采样](test-time-compute.md#best-of-n-采样)是最简单的推理缩放策略，对同一个问题生成 N 个候选答案，选择最好的一个。当评分函数是"多数投票"时，Best-of-N 退化为[自一致性](test-time-compute.md#验证与自我纠错)（Self-Consistency）策略。本实验对比 N=1, 2, 4, 8 四种采样数下的准确率变化，并观察计算量与准确率之间的权衡关系。

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import torch
from collections import Counter
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_samples = 200
n_values = [1, 2, 4, 8]  # 采样数
max_new_tokens = 1024

# ========== 加载 ==========
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
    # 从末尾非计算行提取（避免提取中间步骤的数字）
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

# ========== Best-of-N 评测 ==========
# 先对每道题生成 max(N) 个候选答案，然后对不同 N 值分别计算
max_n = max(n_values)

progress = ProgressReporter(total_steps=num_samples, description="生成候选答案")
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
    progress.update(i + 1, message=f"题目 {i+1}/{num_samples}")

# ========== 计算不同 N 值下的准确率 ==========
print("\n" + "="*60)
print("Best-of-N 采样与自一致性投票对比")
print("="*60)

scaling_results = {}

for n in n_values:
    # 自一致性投票：取 N 个候选中出现次数最多的答案
    sc_correct = 0
    # Best-of-N（随机选择）：取 N 个候选中第一个正确答案
    bon_correct = 0
    total_compute = 0  # 总生成次数

    for item_data in all_candidates:
        candidates_n = item_data["candidates"][:n]
        ref = item_data["ref"]

        # 自一致性投票：多数投票
        valid_answers = [c for c in candidates_n if c is not None]
        if valid_answers:
            counter = Counter(valid_answers)
            majority_answer = counter.most_common(1)[0][0]
            if majority_answer == ref:
                sc_correct += 1

        # Best-of-N（随机）：计算 N 次采样中至少有一个正确答案的概率
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
    print(f"  自一致性投票准确率: {sc_accuracy:.1f}%")
    print(f"  至少一个正确概率:   {bon_accuracy:.1f}%")
    print(f"  总生成次数: {total_compute}")

# ========== 验证推理衰减模型 ==========
# a(n) = a0 + (amax - a0) * (1 - exp(-k*n))
# 对自一致性投票的准确率拟合这个模型
import math

a0 = scaling_results[1]["sc_accuracy"]
amax = max(r["sc_accuracy"] for r in scaling_results.values())

# 网格搜索最优 k
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

print(f"\n推理衰减模型拟合:")
print(f"  a0 = {a0:.1f}% (N=1 时的准确率)")
print(f"  amax = {amax:.1f}% (观测到的最高准确率)")
print(f"  k = {best_k:.2f} (推理效率系数)")
print(f"  拟合公式: a(n) = {a0:.1f} + ({amax:.1f} - {a0:.1f}) × (1 - e^(-{best_k:.2f}×n))")

# 打印拟合值与实际值对比
print(f"\n{'N':>4} {'实际值':>8} {'拟合值':>8} {'误差':>8}")
print("-" * 32)
for n in n_values:
    actual = scaling_results[n]["sc_accuracy"]
    predicted = a0 + (amax - a0) * (1 - math.exp(-best_k * n))
    print(f"{n:>4} {actual:>7.1f}% {predicted:>7.1f}% {abs(actual - predicted):>7.1f}%")

progress.complete(message="推理缩放策略评测完成")
```

运行上方代码后，可以观察到以下规律：

- 自一致性投票的准确率随 N 增大而提升，但增长速度逐渐放缓，这正是[推理衰减模型](test-time-compute.md#推理衰减模型)所描述的边际收益递减现象。
- "至少一个正确"的概率增长更快，因为只要 N 次采样中有一次碰巧答对就能被捕获。但这并不代表我们能直接使用正确的那次采样，实际使用时我们并不知道哪个答案是正确的，仍需要评分函数（如多数投票）来选择。
- 拟合的推理衰减曲线 $a(n) = a_0 + (a_{\max} - a_0)(1 - e^{-kn})$ 能较好地匹配实际数据，为推理缩放的定量规律提供了初步的实验支持。

## 第三阶段：推理效率优化

[推理效率优化](inference-efficiency.md)一章中，我们看到推理效率的本质是在"答得好"与"答得快"之间找到工程上可落地的平衡。本阶段将从量化、KV Cache 实测和投机解码三个方向，实践其具体技术。

### 量化对比

[模型轻量化](inference-efficiency.md#模型轻量化)一节提到，量化通过降低数值精度来减少模型体积、加速推理。但"加速"的实现依赖于底层算子是否支持低精度矩阵乘法。本实验使用的 bitsandbytes 量化采用在线反量化方案，权重以 INT8/INT4 存储，推理时动态反量化为 FP16 再计算。这种方式的主要收益是降低显存占用（使大模型能在有限显存上运行），而非加速推理。实验准备阶段已将 Qwen3.5-0.8B 量化为 INT8 和 INT4 并保存到磁盘，本节直接加载三种精度的模型，对比磁盘大小、显存占用、推理速度和质量差异。

```python runnable gpuonly timeout=unlimited
import os
import json
import re
import time
import logging
import warnings
import torch
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# 抑制 bitsandbytes 量化精度转换提示（INT8 推理时 bfloat16 输入会被转为 float16 计算）
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
# 抑制 PyTorch 内部 API 废弃提示（bitsandbytes 使用的 torch._check_is_size 将被移除）
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
int8_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int8')
int4_path = os.path.join(DATA_DIR, 'models', 'qwen3.5-0.8b-int4')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_eval = 50  # 量化评测用较少题数以节省时间

# ========== 加载数据 ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_eval]

tokenizer = AutoTokenizer.from_pretrained(model_path)

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # 从末尾非计算行提取（避免提取中间步骤的数字）
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

# ========== 权重文件大小统计 ==========
def get_model_file_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.startswith('model') and f.endswith('.safetensors'):
                total += os.path.getsize(os.path.join(dirpath, f))
    return total / 1024**3

# ========== 评测函数 ==========
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
            message=f"{label}: {i+1}/{len(questions)}, 当前准确率 {correct/(i+1)*100:.1f}%"
        )

    accuracy = correct / len(questions) * 100
    avg_time = total_time / len(questions)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    return {"accuracy": accuracy, "avg_time": avg_time, "tokens_per_sec": tokens_per_sec}

# ========== 三种精度对比 ==========
quant_results = {}
total_eval_steps = 3 * num_eval
progress = ProgressReporter(total_steps=total_eval_steps, description="量化精度对比评测")

# FP16（基线）
progress.update(1, message="加载 FP16 模型...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
fp16_vram = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1024**3
fp16_weight = get_model_file_size(model_path)
print(f"\nFP16 显存占用: {fp16_vram:.2f} GB, 权重文件: {fp16_weight:.2f} GB")
quant_results["FP16"] = evaluate_model(model_fp16, questions, "FP16", progress, 0)
quant_results["FP16"]["vram_gb"] = fp16_vram
quant_results["FP16"]["weight_gb"] = fp16_weight
del model_fp16
torch.cuda.empty_cache()

# INT8 量化（加载已保存的量化模型）
progress.update(num_eval + 1, message="加载 INT8 模型...")
model_int8 = AutoModelForCausalLM.from_pretrained(int8_path, device_map="auto")
int8_vram = sum(p.numel() * p.element_size() for p in model_int8.parameters()) / 1024**3
int8_weight = get_model_file_size(int8_path)
print(f"INT8 显存占用: {int8_vram:.2f} GB, 权重文件: {int8_weight:.2f} GB")
quant_results["INT8"] = evaluate_model(model_int8, questions, "INT8", progress, num_eval)
quant_results["INT8"]["vram_gb"] = int8_vram
quant_results["INT8"]["weight_gb"] = int8_weight
del model_int8
torch.cuda.empty_cache()

# INT4 量化（加载已保存的量化模型）
progress.update(2 * num_eval + 1, message="加载 INT4 模型...")
model_int4 = AutoModelForCausalLM.from_pretrained(int4_path, device_map="auto")
int4_vram = sum(p.numel() * p.element_size() for p in model_int4.parameters()) / 1024**3
int4_weight = get_model_file_size(int4_path)
print(f"INT4 显存占用: {int4_vram:.2f} GB, 权重文件: {int4_weight:.2f} GB")
quant_results["INT4"] = evaluate_model(model_int4, questions, "INT4", progress, 2 * num_eval)
quant_results["INT4"]["vram_gb"] = int4_vram
quant_results["INT4"]["weight_gb"] = int4_weight
del model_int4
torch.cuda.empty_cache()

# ========== 汇总结果 ==========
print("\n" + "="*80)
print("量化精度对比")
print("="*80)
print(f"{'精度':<8} {'权重文件':>10} {'显存占用':>10} {'准确率':>8} {'平均耗时':>10} {'生成速度':>14}")
print("-"*80)
for name, r in quant_results.items():
    print(f"{name:<8} {r['weight_gb']:>9.2f}GB {r['vram_gb']:>9.2f}GB {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['tokens_per_sec']:>10.1f} tok/s")

# 压缩比与质量损失
print(f"\n压缩比（相对 FP16）:")
print(f"  INT8 权重: {int8_weight/fp16_weight*100:.1f}%, 显存: {int8_vram/fp16_vram*100:.1f}%")
print(f"  INT4 权重: {int4_weight/fp16_weight*100:.1f}%, 显存: {int4_vram/fp16_vram*100:.1f}%")
print(f"\n质量损失（相对 FP16）:")
fp16_acc = quant_results["FP16"]["accuracy"]
print(f"  INT8: {quant_results['INT8']['accuracy'] - fp16_acc:+.1f}%")
print(f"  INT4: {quant_results['INT4']['accuracy'] - fp16_acc:+.1f}%")

progress.complete(message="量化精度对比评测完成")
```

以下是 Qwen3.5-0.8B-Instruct 在 GSM8K 评测子集（50 题）上的量化对比实测结果：

| 精度 | 权重文件 | 显存占用 | 准确率 | 平均耗时 | 生成速度 |
|------|---------|---------|--------|---------|---------|
| FP16 | 1.63 GB | 1.40 GB | 52.0% | 8.41s | 41.3 tok/s |
| INT8 | 0.94 GB | 0.94 GB | 48.0% | 31.05s | 11.5 tok/s |
| INT4 | 0.74 GB | 0.71 GB | 32.0% | 8.12s | 31.2 tok/s |

压缩比（相对 FP16）为 INT8 权重 57.7%、显存 66.9%，INT4 权重 45.2%、显存 50.4%。质量损失（相对 FP16）为 INT8 -4.0%，INT4 -20.0%。从结果中可以观察到：

- **量化后推理速度反而变慢**：这与直觉相悖，但完全符合 bitsandbytes 的实现原理。bitsandbytes 的 LLM.int8() 和 Q4_K 量化都是在线反量化方案，权重以低精度存储，推理时每步前向传播都需将量化权重动态反量化为 FP16 再做矩阵乘法。反量化本身是额外计算，且 0.8B 模型的 FP16 权重本就完全能放入 GPU 的显存中，量化节省的显存带宽收益微乎其微，反量化的开销反而拖慢了推理。INT8 比 INT4 更慢（11.5 vs 31.2 tok/s），是因为 INT8 的反量化路径涉及更复杂的分块矩阵乘法和异常值分离（Outlier Decomposition），而 INT4 的双量化（Double Quantization）结构更紧凑，反量化开销相对更小。
- **量化的真正价值在于降低显存占用**：INT8 显存降至 FP16 的 67%，INT4 降至 50%。对于 0.8B 这样的小模型，显存节省意义不大；但对于 7B、70B 级别的模型，INT4 量化可以将显存需求从 140GB 降至 35GB，使单卡推理成为可能。这正是量化在工程上的应用场景。
- **准确率损失随量化粒度加剧**：INT8 仅损失 4%，INT4 损失 20%。0.8B 模型参数本就有限，INT4 的激进压缩直接损害了模型的表达能力。对于更大的模型（7B+），INT4 的准确率损失通常在 1%-3% 以内，因为大模型有更多的冗余参数来吸收量化误差。

::: info 真正的量化加速

要实现量化后的推理加速，需要使用支持低精度矩阵乘法的推理引擎，如 llama.cpp（GGUF 格式，CPU/GPU 混合推理）、vLLM（AWQ/GPTQ 格式，GPU 推理）或 TensorRT-LLM（INT8/INT8 Tensor Core 算子）。这些引擎在 kernel 层面直接执行 INT8 或 INT4 的矩阵乘法，无需反量化步骤，才能真正将量化的显存节省转化为速度提升。

:::

### KV Cache 显存实测

[推理瓶颈分析](inference-efficiency.md#推理瓶颈分析)一节给出了 KV Cache 显存占用的估算公式。本节将通过实测验证这个公式的准确性，并观察序列长度对显存占用的影响。

$$M_{\text{KV}} = 2 \times n_{\text{layer}} \times d_{\text{head}} \times n_{\text{head}} \times n_{\text{max}} \times b \times sizeof(\text{dtype})$$

```python runnable gpuonly
import os
import logging
import torch
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

# 提取模型结构参数
config = model.config
n_layer = config.num_hidden_layers
n_head = config.num_attention_heads
n_kv_head = getattr(config, 'num_key_value_heads', n_head)  # GQA 下 KV 头数可能少于 Q 头数
d_head = config.hidden_size // n_head  # 每头维度
hidden_size = config.hidden_size

print(f"模型结构参数:")
print(f"  层数 (n_layer): {n_layer}")
print(f"  注意力头数 (n_head): {n_head}")
print(f"  KV 头数 (n_kv_head): {n_kv_head}")
print(f"  每头维度 (d_head): {d_head}")
print(f"  隐藏维度 (hidden_size): {hidden_size}")

# 实测不同序列长度下的 KV Cache 大小
test_lengths = [128, 256, 512, 1024, 2048]
print(f"\n{'序列长度':>10} {'公式估算':>12} {'KV Cache实测':>14} {'误差':>8}")
print("-"*50)

# 统计标准 Attention 层数（FLA 层不产生 KV Cache）
n_attn_layers = sum(
    1 for layer in model.model.layers
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'k_proj')
)
print(f"标准 Attention 层数: {n_attn_layers}/{n_layer}（其余为 FLA 层，无 KV Cache）\n")

for seq_len in test_lengths:
    # 构造输入
    input_ids = torch.randint(0, len(tokenizer), (1, seq_len), device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)

    # 逐层测量 KV Cache 张量大小，跳过 FLA 层（无 keys/values 属性）
    kv_bytes = 0
    for layer in outputs.past_key_values.layers:
        if hasattr(layer, 'keys') and layer.keys is not None:
            kv_bytes += layer.keys.nelement() * layer.keys.element_size()
        if hasattr(layer, 'values') and layer.values is not None:
            kv_bytes += layer.values.nelement() * layer.values.element_size()
    measured_mb = kv_bytes / 1024**2

    # 公式估算：仅标准 Attention 层产生 KV Cache
    dtype_size = 2  # bfloat16 = 2 bytes
    estimated_bytes = 2 * n_attn_layers * d_head * n_kv_head * seq_len * 1 * dtype_size
    estimated_mb = estimated_bytes / 1024**2

    error_pct = abs(measured_mb - estimated_mb) / max(estimated_mb, 0.01) * 100

    print(f"{seq_len:>10} {estimated_mb:>10.1f}MB {measured_mb:>12.1f}MB {error_pct:>6.1f}%")

    # 清理
    del outputs, input_ids
    torch.cuda.empty_cache()
```

运行上方代码后，公式估算与实测值应该非常接近（误差在 1% 以内），验证了 KV Cache 显存估算公式的准确性。Qwen3.5-0.8B 有两个值得注意的结构特点：

- **[GQA](../architecture-basics/architecture-evolution.md#gqa-分组查询注意力)**（Grouped-Query Attention）：`n_kv_head = 2`，远小于 `n_head = 8`。这意味着每 4 个 Query 头共享一组 KV 头，KV Cache 的显存占用仅为标准 MHA 的 1/4（2/8），这是 GQA 的重要优势，在不显著影响模型质量的前提下大幅降低 KV Cache 的显存需求。
- **[FLA](../architecture-basics/architecture-evolution.md#线性注意力) 混合架构**（Flash Linear Attention）：Qwen3.5 的部分层使用线性注意力，不产生 KV Cache。只有标准 Attention 层才有 KV Cache，因此公式中的层数应取 `n_attn_layers` 而非 `n_layer`。

实测中还可以观察到，KV Cache 的大小与序列长度严格成线性关系，序列长度翻倍，对应 KV Cache 也翻倍。这意味着在长文本推理场景中，KV Cache 的显存增长很容易会成为瓶颈。GQA 和 FLA 正是为了缓解这一瓶颈而设计的，GQA 将 KV Cache 缩减为 MHA 的 `n_kv_head / n_head` 倍，FLA 层则完全消除了 KV Cache，这些实践与 [Transformer 演进与变体](../architecture-basics/architecture-evolution.md)中对注意力机制改进的理论描述互相印证。

### 投机解码

[投机解码](inference-efficiency.md#投机解码)的设计思想是用一个小模型（Draft Model）快速生成候选 token，再用大模型（Target Model）一次前向传播批量验证。[推理效率优化](inference-efficiency.md#投机解码)一章中曾介绍过一种名为 Medusa 的附加框架，它不需要独立的 Draft Model，而是在模型最后一个隐藏层上添加多个预测头。本节将训练 Medusa 头并实测投机解码的加速效果。

注意，实验的目的是进行教学，生产实践中没有任何理由对 Qwen3.5-0.8B 这种 0.8B 参数规模的模型做投机解码，这是没有意义甚至有反效果的。投机解码加速的原理是 Decode 阶段受内存带宽限制，每次前向传播需要把全部模型参数从显存搬到计算单元，却只生成一个 token，算力利用率低下。这种情况下，投机解码用 2 次前向（Draft + Verify）生成多个 token，只要投机命中率足够高，2 次前向的耗时就低于逐 token 生成同样多 token 的耗时。但 0.8B 模型的参数只有 1.6GB，内存带宽瓶颈并不严重。此时投机解码的 2 次前向传播就是纯粹的额外开销，每次验证还需要对越来越长的序列做完整前向计算，反而比自回归更慢。投机解码的加速效果随模型规模增大而显著，通常至少在 7B/13B 的模型才能观察到投机节省的步数能覆盖额外的验证开销。

Medusa 的原始论文中，每个头都预测下一个 token（t+1 位置），但通过不同的概率分布提供多样化的候选，再利用树形注意力机制同时验证多条候选路径。为了降低实现复杂度，本实验采用简化方案，让每个头预测不同位置的未来 token（Head_k 预测 t+k+1），顺序验证每个候选是否与目标模型一致。

Medusa 头的结构是一个残差块（ResBlock）加一个输出层。残差块使用瓶颈结构（Hidden Size → Bottleneck → Hidden Size）减少参数量，输出层使用低秩分解（Hidden Size → Rank → Vocab Size）避免词汇表过大导致的参数爆炸。每层之间的下投影权重用 [He 初始化](../../deep-learning/neural-network-stability/weight-initialization.md#he-初始化)保证梯度流动，上投影权重零初始化使训练初期不影响主干模型。

```python runnable gpuonly timeout=unlimited
import os
import json
import logging
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# 抑制 Qwen3.5 模型的 FLA 加速库缺失提示（不影响功能，仅回退到纯 PyTorch 实现）
logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
# 抑制 bitsandbytes 量化精度转换提示和 PyTorch 内部 API 废弃提示
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*to float16.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== Medusa 头定义 ==========
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

# ========== 加载模型 ==========
model_path = os.path.join(DATA_DIR, 'models', 'llm', 'qwen3.5-0.8b-instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
base_model.eval()

# 创建 4 个 Medusa 头
# Head_k 在位置 t 的输出预测位置 t+k+1 的 token
# Head_0 预测 t+1（与主干 lm_head 相同位置），Head_1 预测 t+2，依此类推
hidden_size = base_model.config.hidden_size
vocab_size = base_model.config.vocab_size
medusa_heads = nn.ModuleList([
    MedusaHead(hidden_size, vocab_size).to(dtype=torch.bfloat16, device="cuda")
    for _ in range(4)
])

# 冻结主干，只训练 Medusa 头
for param in base_model.parameters():
    param.requires_grad = False
trainable_params = sum(p.numel() for p in medusa_heads.parameters())
print(f"主干模型参数: {sum(p.numel() for p in base_model.parameters()):,}")
print(f"Medusa 头参数: {trainable_params:,}")

# ========== 训练数据 ==========
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

# ========== 训练（Medusa-1：冻结主干，只训练头）==========
optimizer = torch.optim.AdamW(
    [p for p in medusa_heads.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=0.0
)

num_steps = 1000
progress = ProgressReporter(total_steps=num_steps, description="训练 Medusa 头")
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

progress.complete(message=f"训练完成，{num_steps} 步，平均 Loss={total_loss/num_steps:.4f}")

# ========== 评测各头准确率 ==========
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

print(f"\nMedusa 头准确率（Top-1，100 题评测集）:")
print(f"{'头':>6} {'预测位置':>10} {'准确率':>10}")
print("-" * 30)
for k in range(4):
    acc = head_correct[k] / max(head_total[k], 1) * 100
    print(f"Head_{k:>2} {'t+' + str(k+1):>10} {acc:>9.1f}%")

# ========== 投机解码评测 ==========
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
    """简化版 Medusa 投机解码（top-k 候选）

    每个头取 top-k 候选而非仅 top-1，提升匹配概率。
    验证时检查目标模型的 argmax 是否落在候选的 top-k 列表中。
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
            # 每个头取 top-k 候选，用 top-1 拼接验证序列
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
# 自回归解码
ar_total_time = 0
ar_total_tokens = 0
for item in eval_questions[:num_bench]:
    prompt = f"User: {item['question']}\nPlease think step by step.\nAssistant:"
    start = time.time()
    toks = autoregressive_generate(base_model, tokenizer, prompt, max_new_tokens=200)
    ar_total_time += time.time() - start
    ar_total_tokens += len(toks)

# Medusa 投机解码（top-5 候选）
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

# 计算 top-1 vs top-5 头准确率对比
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

print(f"\nMedusa 头准确率对比（Top-1 vs Top-5）:")
print(f"{'头':>6} {'预测位置':>10} {'Top-1':>8} {'Top-5':>8}")
print("-" * 36)
for k in range(4):
    print(f"Head_{k:>2} {'t+' + str(k+1):>10} {topk_acc[1][k]:>7.1f}% {topk_acc[5][k]:>7.1f}%")

print(f"\n解码速度对比（{num_bench} 题，最多 200 tokens，greedy decoding，top-5 候选）:")
print(f"{'方式':<20} {'生成速度':>12} {'平均耗时':>10} {'加速比':>8} {'命中率':>8}")
print("-" * 62)
print(f"{'自回归解码':<20} {ar_tps:>10.1f} tok/s {ar_total_time/num_bench:>8.2f}s {'1.00x':>8} {'-':>8}")
print(f"{'Medusa 投机解码':<20} {spec_tps:>10.1f} tok/s {spec_total_time/num_bench:>8.2f}s {speedup:>6.2f}x {avg_hit:>6.1f}%")
```

运行上方代码后，Medusa 头的准确率随预测位置增大而递减，这是符合预期的，越远的 token 越难预测。使用 top-5 候选（每个头取概率最高的 5 个 token）相比 top-1 有显著提升。以下是实验的测试结果：

| 头 | 预测位置 | Top-1 准确率 | Top-5 准确率 |
|------|---------|------------|------------|
| Head_0 | t+1 | 60.8% | 77.6% |
| Head_1 | t+2 | 44.8% | 64.2% |
| Head_2 | t+3 | 27.9% | 53.1% |
| Head_3 | t+4 | 21.1% | 46.1% |

| 解码方式 | 生成速度 | 加速比 | 投机命中率 |
|---------|---------|--------|--------|
| 自回归解码 | 40.3 tok/s | 1.00x | - |
| Medusa 投机解码 (top-5) | 10.5 tok/s | 0.26x | 14.4% |


注意，表格中的"投机命中率"是指被目标模型验证通过的投机 token 数 / 总投机 token 数，投机长度 $\gamma$ 越大值越低，实际测得约为 14%。这个指标与业界通常说的 50-85% 投机接受率并不是一个概念。投机接受率（Per-token Acceptance Rate，$\alpha$）指的是每个 Draft Model 生成的 token 独立被接受的概率。在传统投机解码中，Draft Model 逐 token 生成候选，每个 token 以概率 $\alpha = \min\!\left(1,\;\frac{p(x)}{q(x)}\right)$ 被接受（$p$、$q$ 分别是目标模型和草稿模型的概率分布）。$\alpha$ 取决于草稿模型与目标模型分布的对齐程度，与投机长度 $\gamma$ 无关。

## 实验结论

本次实验使用 Qwen3.5-0.8B-Instruct 模型，在 GSM8K 评测集上完成了从思维链提示到推理缩放再到效率优化的全链路实践。实验验证了以下结论：

- **思维链提示**有效但受限于模型规模。0.8B 模型的 CoT 能够观察到确有提升，但效果相对有限，这印证了"模型规模越大 CoT 效果越明显"的研究发现。
- **推理时缩放**（Best-of-N、自一致性投票）能系统性地提升准确率，且增长符合推理衰减模型的边际收益递减规律。动态推理深度可以根据问题难度自适应分配计算资源，在保持准确率的前提下节省计算量。
- **推理效率优化**（量化、KV Cache 管理、投机解码）在"答得好"与"答得快"之间做出权衡。本次实验中，每一种优化措施都伴随有不同的工程决策，如果不理解原理，无法正确选择应用场景，甚至可能会带来反效果。

三种技术并非孤立，实践中往往组合使用。一个经过量化的小模型，配合动态推理深度和自一致性投票，可以在有限的计算资源下获得更好的推理效果。这也呼应了[三大缩放定律](test-time-compute.md#三大缩放定律的统一视角)的统一逻辑：预训练决定上限，后训练让能力可用，推理缩放让潜力兑现，而推理效率优化决定了这套体系能否在工程上落地。
