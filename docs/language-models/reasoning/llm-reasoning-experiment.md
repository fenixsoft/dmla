# LLM 推理策略与效率优化实验

在前几章中，我们从理论层面了解了思维链、推理时缩放和推理效率优化。这些知识构成了理解推理模型的原理基础，但仅凭理论难以真正体会推理策略的工程抉择。本次实验中，我们将使用 Qwen3.5-0.8B-Instruct 模型，亲手实现从思维链提示到推理缩放再到效率优化的完整流程，在实践中理解"答得好"与"答得快"之间的权衡。

## 实验准备

在开始实验之前，请确保已[挂载数据目录](../../sandbox.md#数据管理)并下载好 GSM8K 评测子集，你可以通过 `DMLA-CLI` 工具自动完成该工作：

```bash
# 选择 "下载数据集" -> 选择 "GSM8K 200 (数学推理评测集)"
dmla data
```

GSM8K（Grade School Math 8K）是一个包含 1319 道小学数学应用题的推理基准，模型需要通过多步计算才能得出正确答案。本实验从中随机抽取了 200 道题作为评测子集，足以反映模型在不同推理策略下的性能差异，同时将评测时间控制在合理范围内。数据集下载完成后，以下代码可验证数据和模型加载是否正常：

```python runnable gpuonly
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 检查 GSM8K 评测数据
gsm8k_dir = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
if os.path.exists(gsm8k_dir):
    import json
    with open(os.path.join(gsm8k_dir, 'gsm8k_200.jsonl'), 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    print(f"GSM8K 评测子集: {len(questions)} 题")
    print(f"示例问题: {questions[0]['question'][:80]}...")
else:
    print("GSM8K 评测子集: 未下载，请运行 'dmla data' 下载数据集")

# 加载 Qwen3.5-0.8B-Instruct 模型
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
print(f"\n正在加载模型 {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e9:.2f}B ({total_params:,})")
print(f"模型 dtype: {model.dtype}")
print(f"设备: {model.device}")
```

## 第一阶段：思维链与提示工程

[思维链与推理模型](chain-of-thought.md)一章中，我们看到思维链提示能让模型从"凭直觉给答案"转变为"按步骤做推导"。本阶段将在 GSM8K 评测集上，对比三种提示策略的效果差异：零样本直接回答、零样本 CoT、少样本 CoT。

三种策略的区别在于提示词的设计：

- **零样本直接回答**（Zero-Shot Direct）：直接提出问题，模型给出答案，不附加任何推理引导。
- **零样本 CoT**（Zero-Shot CoT）：在问题后追加"请一步一步思考"的引导语，让模型自主生成推理步骤。这就是[思维链](chain-of-thought.md#思维链)一章中介绍的零样本思维链方法。
- **少样本 CoT**（Few-Shot CoT）：在问题前提供 2-3 个带有完整推理过程的示例，让模型学习如何组织推理步骤。

Qwen3.5 系列模型原生支持 thinking/non-thinking 模式切换。thinking 模式下，模型会在 `<think>...</think>` 标签内生成推理过程，然后给出最终答案，这与零样本 CoT 的效果类似但更结构化。non-thinking 模式下，模型直接输出答案。下面我们对比这几种策略的效果：

```python runnable gpu timeout=unlimited
import os
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
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
    """从模型输出或 GSM8K 参考答案中提取最终数值"""
    # GSM8K 参考答案格式: "#### 123"
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # 模型输出中的最终答案
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== 推理函数 ==========
def generate_response(messages, max_new_tokens=1024, enable_thinking=True):
    """调用模型生成回答"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6 if enable_thinking else 0.7,
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
        "content": "Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast, so she has 16 - 3 = 13 eggs left.\nShe uses 4 for muffins, so she has 13 - 4 = 9 eggs left to sell.\nShe sells each egg for $2, so she makes 9 × $2 = $18 per day.\n#### 18"
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
    "零样本直接回答": lambda q: ([{"role": "user", "content": q}], False),
    "零样本CoT（thinking模式）": lambda q: ([{"role": "user", "content": q + "\n请一步一步思考并给出最终答案。"}], True),
    "少样本CoT": lambda q: (few_shot_examples + [{"role": "user", "content": q + "\n请一步一步思考并给出最终答案。"}], True),
}

results = {}
progress = ProgressReporter(total_steps=len(strategies) * num_samples, description="思维链提示策略评测")

for strategy_name, make_messages in strategies.items():
    correct = 0
    total = 0
    total_time = 0
    total_tokens = 0

    for i, item in enumerate(questions):
        messages, thinking = make_messages(item['question'])
        ref_answer = extract_answer(item['answer'])

        start_time = time.time()
        try:
            response = generate_response(messages, enable_thinking=thinking)
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

运行上方代码后，你可以观察到以下规律：

- 零样本直接回答的准确率最低，因为模型没有展示推理过程，容易在多步计算中出错。
- 零样本 CoT（thinking 模式）准确率有提升，模型在 `<think>` 标签内生成了推理步骤，但 0.8B 模型的推理链质量有限，可能出现在推理过程中犯错的情况。
- 少样本 CoT 准确率最高，示例为模型提供了推理格式的模板，让模型知道应该如何组织推理步骤。

这些结果印证了[思维链](chain-of-thought.md#思维链)一章的分析：思维链通过分解复杂问题、激活相关知识和提供纠错机会来提升推理能力。同时也能观察到，0.8B 模型的 CoT 效果提升幅度有限，这与"模型规模越大 CoT 效果越明显"的研究结论一致。

## 第二阶段：推理时缩放策略

[推理时缩放定律](test-time-compute.md)一章中，我们看到在推理阶段投入更多计算可以系统地提升模型性能。本阶段将实现 Best-of-N 采样和自一致性投票两种缩放策略，并验证推理衰减模型。

### Best-of-N 采样与自一致性投票

[Best-of-N 采样](test-time-compute.md#best-of-n-采样)是最简单的推理缩放策略：对同一个问题生成 N 个候选答案，选择最好的一个。当评分函数是"多数投票"时，Best-of-N 就变成了[自一致性](test-time-compute.md#验证与自我纠错)（Self-Consistency）策略。本实验对比 N=1, 2, 4, 8 四种采样数下的准确率变化。

```python runnable gpu timeout=unlimited
import os
import json
import re
import time
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
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
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

def generate_response(question, enable_thinking=True):
    messages = [{"role": "user", "content": question + "\n请一步一步思考并给出最终答案。"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
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

        # Best-of-N（随机）：等价于 N=1 的单次采样
        # 更公平的对比是看"至少一个正确"的概率
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

运行上方代码后，你可以观察到以下规律：

- 自一致性投票的准确率随 N 增大而提升，但增长速度逐渐放缓，符合[推理衰减模型](test-time-compute.md#推理衰减模型)的边际收益递减规律。
- "至少一个正确"的概率增长更快，因为只要 N 次采样中有一次碰巧答对就能被捕获。但这不意味着我们可以直接使用正确的那次采样，因为实际使用时我们并不知道哪个答案是正确的，仍需要评分函数（如多数投票）来选择。
- 拟合的推理衰减曲线 $a(n) = a_0 + (a_{\max} - a_0)(1 - e^{-kn})$ 能较好地匹配实际数据，验证了推理缩放的定量规律。

### 动态推理深度

[动态推理深度](test-time-compute.md#动态推理深度)的核心思想是根据问题难度分配不同的推理计算资源。简单问题用少量采样，困难问题用更多采样。本节将实现一个简单的问题难度评估器，并对比固定采样和动态采样的效率差异。

```python runnable gpu timeout=unlimited
import os
import json
import re
import math
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_samples = 100  # 动态深度评测用较少题数以节省时间

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
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

def generate_response(question):
    messages = [{"role": "user", "content": question + "\n请一步一步思考并给出最终答案。"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# ========== 阶段一：评估每道题的难度 ==========
# 用模型通过率作为难度指标：对每题生成 4 次，统计正确次数
progress = ProgressReporter(total_steps=num_samples, description="评估题目难度")
difficulty_map = {}  # question_index -> difficulty (0-4)

for i, item in enumerate(questions):
    ref = extract_answer(item['answer'])
    correct_count = 0
    for _ in range(4):
        try:
            response = generate_response(item['question'])
            pred = extract_answer(response)
            if pred == ref:
                correct_count += 1
        except:
            pass
    difficulty_map[i] = correct_count  # 0=最困难, 4=最简单
    progress.update(i + 1, message=f"题目 {i+1}/{num_samples}, 通过率={correct_count}/4")

# 按难度分组
easy = [i for i, d in difficulty_map.items() if d >= 3]     # 通过率 >= 75%
medium = [i for i, d in difficulty_map.items() if 1 <= d <= 2]  # 通过率 25%-50%
hard = [i for i, d in difficulty_map.items() if d == 0]      # 通过率 0%

print(f"\n题目难度分布:")
print(f"  简单 (≥3/4): {len(easy)} 题")
print(f"  中等 (1-2/4): {len(medium)} 题")
print(f"  困难 (0/4):  {len(hard)} 题")

# ========== 阶段二：对比固定采样 vs 动态采样 ==========
# 动态策略：简单题 N=1, 中等题 N=4, 困难题 N=8
# 固定策略：所有题 N=4
dynamic_n_map = {**{i: 1 for i in easy}, **{i: 4 for i in medium}, **{i: 8 for i in hard}}
fixed_n = 4

# 先生成所有需要的候选答案
progress.reset(total_steps=num_samples, description="生成候选答案（动态深度）")
all_dynamic_candidates = {}

for i, item in enumerate(questions):
    ref = extract_answer(item['answer'])
    n = dynamic_n_map[i]
    candidates = []
    for _ in range(n):
        try:
            response = generate_response(item['question'])
            pred = extract_answer(response)
            candidates.append(pred)
        except:
            candidates.append(None)
    all_dynamic_candidates[i] = {"candidates": candidates, "ref": ref}
    progress.update(i + 1, message=f"题目 {i+1}/{num_samples} (N={n})")

# 计算动态策略的准确率和总生成次数
dynamic_correct = 0
dynamic_total_compute = 0
for i in range(num_samples):
    candidates = all_dynamic_candidates[i]["candidates"]
    ref = all_dynamic_candidates[i]["ref"]
    valid = [c for c in candidates if c is not None]
    if valid:
        majority = Counter(valid).most_common(1)[0][0]
        if majority == ref:
            dynamic_correct += 1
    dynamic_total_compute += len(candidates)

dynamic_accuracy = dynamic_correct / num_samples * 100

# 固定策略使用已有数据（之前 Best-of-N 实验中 N=4 的结果）
# 这里重新从 all_dynamic_candidates 中截取前 4 个
fixed_correct = 0
fixed_total_compute = num_samples * fixed_n
for i in range(num_samples):
    candidates = all_dynamic_candidates[i]["candidates"][:fixed_n]
    ref = all_dynamic_candidates[i]["ref"]
    valid = [c for c in candidates if c is not None]
    if valid:
        majority = Counter(valid).most_common(1)[0][0]
        if majority == ref:
            fixed_correct += 1

fixed_accuracy = fixed_correct / num_samples * 100

print(f"\n" + "="*60)
print(f"固定采样 vs 动态采样对比")
print(f"="*60)
print(f"{'策略':<20} {'准确率':>8} {'总生成次数':>12} {'平均N':>8}")
print(f"-"*60)
print(f"{'固定 N=4':<20} {fixed_accuracy:>7.1f}% {fixed_total_compute:>12} {fixed_n:>8}")
avg_dynamic_n = dynamic_total_compute / num_samples
print(f"{'动态 (1/4/8)':<20} {dynamic_accuracy:>7.1f}% {dynamic_total_compute:>12} {avg_dynamic_n:>8.1f}")
print(f"\n动态策略节省计算量: {(1 - dynamic_total_compute/fixed_total_compute)*100:.1f}%")

progress.complete(message="动态推理深度评测完成")
```

动态推理深度的实验结果通常会显示：动态策略在准确率与固定策略接近的前提下，节省了可观的计算量。这是因为简单题目不需要多次采样就能答对，将计算资源集中到困难题目上更有效率。这正是[动态推理深度](test-time-compute.md#动态推理深度)一节所讨论的"量力而行"策略。

## 第三阶段：推理效率优化

[推理效率优化](inference-efficiency.md)一章中，我们看到推理效率的本质是在"答得好"与"答得快"之间找到工程上可落地的平衡点。本阶段将从量化、KV Cache 实测和投机解码三个方向，实践推理效率优化的具体技术。

### 量化对比

[模型轻量化](inference-efficiency.md#模型轻量化)一节提到，量化通过降低数值精度来减少模型体积和加速推理。本节对比 FP16、INT8 和 INT4 三种精度下模型的推理速度和质量差异。

```python runnable gpu timeout=unlimited
import os
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
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
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== 评测函数 ==========
def evaluate_model(model, questions, label=""):
    model.eval()
    correct = 0
    total_time = 0
    total_tokens = 0

    for item in questions:
        messages = [{"role": "user", "content": item['question'] + "\n请一步一步思考并给出最终答案。"}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
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

    accuracy = correct / len(questions) * 100
    avg_time = total_time / len(questions)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    return {"accuracy": accuracy, "avg_time": avg_time, "tokens_per_sec": tokens_per_sec}

# ========== 三种精度对比 ==========
quant_results = {}
progress = ProgressReporter(total_steps=3, description="量化精度对比评测")

# FP16（基线）
progress.update(1, message="加载 FP16 模型...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
fp16_size = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1024**3
print(f"\nFP16 模型显存占用: {fp16_size:.2f} GB")
progress.update(1, message="评测 FP16 模型...")
quant_results["FP16"] = evaluate_model(model_fp16, questions, "FP16")
quant_results["FP16"]["model_size_gb"] = fp16_size
del model_fp16
torch.cuda.empty_cache()

# INT8 量化
progress.update(2, message="加载 INT8 模型...")
int8_config = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_path, quantization_config=int8_config, device_map="auto"
)
int8_size = sum(p.numel() * p.element_size() for p in model_int8.parameters()) / 1024**3
print(f"INT8 模型显存占用: {int8_size:.2f} GB")
progress.update(2, message="评测 INT8 模型...")
quant_results["INT8"] = evaluate_model(model_int8, questions, "INT8")
quant_results["INT8"]["model_size_gb"] = int8_size
del model_int8
torch.cuda.empty_cache()

# INT4 量化
progress.update(3, message="加载 INT4 模型...")
int4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model_int4 = AutoModelForCausalLM.from_pretrained(
    model_path, quantization_config=int4_config, device_map="auto"
)
int4_size = sum(p.numel() * p.element_size() for p in model_int4.parameters()) / 1024**3
print(f"INT4 模型显存占用: {int4_size:.2f} GB")
progress.update(3, message="评测 INT4 模型...")
quant_results["INT4"] = evaluate_model(model_int4, questions, "INT4")
quant_results["INT4"]["model_size_gb"] = int4_size
del model_int4
torch.cuda.empty_cache()

# ========== 汇总结果 ==========
print("\n" + "="*70)
print("量化精度对比")
print("="*70)
print(f"{'精度':<8} {'模型大小':>10} {'准确率':>8} {'平均耗时':>10} {'生成速度':>14}")
print("-"*70)
for name, r in quant_results.items():
    print(f"{name:<8} {r['model_size_gb']:>9.2f}GB {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['tokens_per_sec']:>10.1f} tok/s")

progress.complete(message="量化精度对比评测完成")
```

### KV Cache 显存实测

[推理瓶颈分析](inference-efficiency.md#推理瓶颈分析)一节给出了 KV Cache 显存占用的估算公式。本节将通过实测验证这个公式的准确性，并观察序列长度对显存占用的影响。

$$M_{\text{KV}} = 2 \times n_{\text{layer}} \times d_{\text{head}} \times n_{\text{head}} \times n_{\text{max}} \times b \times sizeof(\text{dtype})$$

```python runnable gpu
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

# 提取模型结构参数
config = model.config
n_layer = config.num_hidden_layers
n_head = config.num_attention_heads
d_head = config.hidden_size // n_head  # 每头维度
hidden_size = config.hidden_size

print(f"模型结构参数:")
print(f"  层数 (n_layer): {n_layer}")
print(f"  注意力头数 (n_head): {n_head}")
print(f"  每头维度 (d_head): {d_head}")
print(f"  隐藏维度 (hidden_size): {hidden_size}")

# 实测不同序列长度下的 KV Cache 大小
test_lengths = [128, 256, 512, 1024, 2048]
print(f"\n{'序列长度':>10} {'公式估算':>12} {'实测占用':>12} {'误差':>8}")
print("-"*50)

for seq_len in test_lengths:
    # 构造输入
    input_ids = torch.randint(0, len(tokenizer), (1, seq_len), device=model.device)

    # 记录推理前显存
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        outputs = model(input_ids)

    # 记录推理后显存（包含 KV Cache）
    mem_after = torch.cuda.max_memory_allocated() / 1024**2
    measured_mb = mem_after - mem_before

    # 公式估算
    dtype_size = 2  # bfloat16 = 2 bytes
    estimated_bytes = 2 * n_layer * d_head * n_head * seq_len * 1 * dtype_size
    estimated_mb = estimated_bytes / 1024**2

    error_pct = abs(measured_mb - estimated_mb) / max(estimated_mb, 0.01) * 100

    print(f"{seq_len:>10} {estimated_mb:>10.1f}MB {measured_mb:>10.1f}MB {error_pct:>6.1f}%")

    # 清理
    del outputs, input_ids
    torch.cuda.empty_cache()

print(f"\n注意：实测值可能包含中间计算缓冲区，因此可能略高于公式估算值。")
```

### 投机解码

[投机解码](inference-efficiency.md#投机解码)的核心思想是用一个小模型（Draft Model）快速生成候选 token，再用大模型（Target Model）一次前向传播批量验证。Qwen3.5 模型在设计时内置了 MTP Heads（Multi-Token Prediction Heads），无需额外添加独立的小模型即可支持投机解码。

Qwen3.5 的 MTP 机制在每次前向传播时，除了预测下一个 token 外，还会同时预测未来第 2、3 个 token。这些额外预测头的计算开销极低（只有一个线性层），但在推理时可以提供"推测"的候选 token，配合 Target Model 的验证实现加速。

由于 0.8B 模型本身已经足够小，用更小的 Draft Model 意义不大。本节改为对比**自回归逐 token 生成**与**多次采样策略**的效率差异，这更能体现实际工程中的权衡——是用更长的时间做一次深度推理，还是用相同的时间做多次浅层推理然后投票。

```python runnable gpu timeout=unlimited
import os
import json
import re
import time
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from dmla_progress import ProgressReporter

# ========== 配置 ==========
model_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200')
gsm8k_path = os.path.join(DATA_DIR, 'datasets', 'gsm8k-200', 'gsm8k_200.jsonl')
num_eval = 50

# ========== 加载 ==========
with open(gsm8k_path, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = all_questions[:num_eval]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16
).to("cuda")
model.eval()

def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return matches[-1] if matches else None

# ========== 三种推理策略对比 ==========
# 策略 A: 单次深度推理（长 CoT，max_new_tokens=1024）
# 策略 B: 4 次浅层推理 + 多数投票（max_new_tokens=256）
# 策略 C: 8 次浅层推理 + 多数投票（max_new_tokens=128）
strategies = {
    "单次深度推理 (1024 tokens)": {"n": 1, "max_tokens": 1024, "thinking": True},
    "4次浅层推理+投票 (256×4)": {"n": 4, "max_tokens": 256, "thinking": True},
    "8次浅层推理+投票 (128×8)": {"n": 8, "max_tokens": 128, "thinking": True},
}

results = {}
progress = ProgressReporter(total_steps=len(strategies) * num_eval, description="推理效率策略对比")

for strat_name, strat_config in strategies.items():
    correct = 0
    total_time = 0
    total_tokens = 0
    n = strat_config["n"]
    max_tokens = strat_config["max_tokens"]

    for i, item in enumerate(questions):
        ref = extract_answer(item['answer'])
        candidates = []
        start = time.time()

        for _ in range(n):
            messages = [{"role": "user", "content": item['question'] + "\n请一步一步思考并给出最终答案。"}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=strat_config["thinking"]
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(response)
            candidates.append(pred)
            total_tokens += outputs.shape[-1] - inputs["input_ids"].shape[-1]

        elapsed = time.time() - start
        total_time += elapsed

        # 多数投票
        valid = [c for c in candidates if c is not None]
        if valid:
            majority = Counter(valid).most_common(1)[0][0]
            if majority == ref:
                correct += 1

        progress.update(
            len(results) * num_eval + i + 1,
            message=f"{strat_name}: {i+1}/{num_eval}"
        )

    accuracy = correct / num_eval * 100
    avg_time = total_time / num_eval
    tps = total_tokens / total_time if total_time > 0 else 0
    total_budget = n * max_tokens  # 每题总 token 预算

    results[strat_name] = {
        "accuracy": accuracy,
        "avg_time": avg_time,
        "tps": tps,
        "total_budget": total_budget
    }

# ========== 汇总 ==========
print("\n" + "="*80)
print("推理效率策略对比：相同 token 预算，不同分配方式")
print("="*80)
print(f"{'策略':<35} {'准确率':>8} {'平均耗时':>10} {'token预算':>10} {'生成速度':>14}")
print("-"*80)
for name, r in results.items():
    print(f"{name:<35} {r['accuracy']:>7.1f}% {r['avg_time']:>9.2f}s {r['total_budget']:>10} {r['tps']:>10.1f} tok/s")

progress.complete(message="推理效率策略对比完成")
```

运行上方代码后，你可以观察到以下规律：

- 三种策略的总 token 预算相近（1024 vs 256×4 vs 128×8），但准确率和速度的差异揭示了推理缩放与效率之间的深层权衡。
- 单次深度推理的优势在于推理链完整、逻辑连贯，但一旦出错就无法纠正。多次浅层推理的优势在于通过投票纠错，但每次推理的深度有限，可能无法完成复杂的多步推理。
- 这正是[推理效率优化](inference-efficiency.md)一章所讨论的核心矛盾：推理缩放靠"多采样"提升准确率，但多采样意味着更多计算。如何在准确率和效率之间找到平衡，取决于具体的应用场景。

## 实验结论

本次实验使用 Qwen3.5-0.8B-Instruct 模型，在 GSM8K 评测集上完成了从思维链提示到推理缩放再到效率优化的完整流程。实验验证了以下结论：

- **思维链提示**有效但受限于模型规模。0.8B 模型的 CoT 效果提升有限，印证了"模型规模越大 CoT 效果越明显"的研究结论。
- **推理时缩放**（Best-of-N、自一致性投票）能系统性地提升准确率，且增长符合推理衰减模型的边际收益递减规律。动态推理深度可以根据问题难度自适应分配计算资源，在保持准确率的前提下节省计算量。
- **推理效率优化**（量化、KV Cache 管理、计算预算分配）在"答得好"与"答得快"之间做出权衡。量化能以较小的质量损失换取显著的效率提升；相同 token 预算下，"单次深度推理"和"多次浅层推理+投票"各有适用场景。

三种技术并非孤立，实践中往往组合使用。一个经过量化的小模型，配合动态推理深度和自一致性投票，可以在有限的计算资源下获得最优的推理效果。这印证了[三大缩放定律](test-time-compute.md#三大缩放定律的统一视角)的核心思想：预训练决定上限，后训练让能力可用，推理缩放让潜力兑现——而推理效率优化，则决定了这套体系能否在工程上落地。
