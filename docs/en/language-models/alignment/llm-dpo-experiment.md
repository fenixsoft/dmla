# DPO Alignment Training Experiment

In the [SFT Model Chat Experiment](../pretraining/llm-sft-experiment.md), we taught the model to follow conversational formats through supervised fine-tuning, achieving the transition from text continuation to dialogue. SFT training data consists of instruction-response pairs following the "user asks → model answers" format, where the model learns to imitate the response patterns in the training data. However, the same question can have multiple valid answers, and SFT cannot tell the model which one better aligns with human preferences — whether the response is more accurate, the tone friendlier, or the refusal more polite. SFT fails to capture these nuanced differences.

This experiment uses Direct Preference Optimization to perform alignment training on top of the SFT model. DPO training data consists of two responses to the same question, with human annotations indicating which one is better. The model learns to distinguish between good and bad responses, and during generation, it tends to favor the preferred response style.

## Experiment Preparation

Before starting the experiment, please ensure the following preparations are complete:

1. The [SFT Model Chat Experiment](../pretraining/llm-sft-experiment.md) has been completed, and the model weight file `full_sft_768.pth` has been correctly generated in the data directory.
2. You have [mounted the data directory](../../appendixes/sandbox.md#data-management) and downloaded the DPO preference dataset.

```bash
# Select "Download dataset" -> Select "MiniMind Alignment"
dmla data
```

The MiniMind project's DPO preference dataset (`dpo.jsonl`) contains approximately 20K preference comparison pairs, sampled from [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k), with a size of about 53 MB. Once the dataset is downloaded, the following code verifies that both the SFT model and DPO data are complete:

```python runnable gpu
import os

# Check SFT model (generated in the previous chapter)
sft_dir = os.path.join(DATA_DIR, 'models', 'minimind', 'sft')
sft_path = os.path.join(sft_dir, 'full_sft_768.pth')
if os.path.exists(sft_path):
    size_mb = os.path.getsize(sft_path) / (1024 ** 2)
    print(f"SFT model: exists ({size_mb:.1f} MB)")
else:
    # Try epoch checkpoint
    for epoch in [2, 1]:
        ckp = os.path.join(sft_dir, f'sft_epoch{epoch}.pth')
        if os.path.exists(ckp):
            size_mb = os.path.getsize(ckp) / (1024 ** 2)
            print(f"SFT model: using epoch {epoch} checkpoint ({size_mb:.1f} MB)")
            break
    else:
        print("SFT model: not found! Please complete the SFT experiment first.")

# Check DPO data
dpo_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-alignment')
if os.path.exists(dpo_dir):
    print(f"DPO data directory: exists")
    for f in os.listdir(dpo_dir):
        fpath = os.path.join(dpo_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"  {f}: {size_mb:.1f} MB")
else:
    print("DPO data: not downloaded. Run 'dmla data' to download the MiniMind Alignment dataset.")

# Check tokenizer (reuse pre-trained)
tokenizer_dir = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer_json = os.path.join(tokenizer_dir, 'tokenizer.json')
print(f"Tokenizer: {'exists' if os.path.exists(tokenizer_json) else 'not found'}")
```

## Phase 1: Preference Comparison Dataset

DPO training data format differs from SFT. Each SFT sample is an instruction-response pair $(x, y)$, while each DPO sample is a preference comparison triple $(x, y_w, y_l)$, where $x$ is the user instruction, $y_w$ is the chosen (preferred) response, and $y_l$ is the rejected (dispreferred) response. The chosen and rejected responses correspond to the same user instruction, differing only in the assistant's reply.

The data is stored in JSONL format, with one preference pair per line:

```json
{
  "chosen": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data..."}
  ],
  "rejected": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is when computers learn things by themselves"}
  ]
}
```

The following code implements DPODataset, which converts preference comparison data into a trainable format. Each sample contains chosen and rejected conversations, which are tokenized separately to generate corresponding input sequences and masks. The mask ensures that log-probabilities are only computed on the assistant's response portion, while the user's question part is excluded from the DPO loss calculation. This code is called during the training phase and does not need to be run manually.

```python runnable gpuonly extract-class="DPODataset"
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value
from datasets import logging as datasets_logging

class DPODataset(Dataset):
    """
    DPO dataset: tokenizes preference comparison data into a trainable format

    Each sample format: {"chosen": [{role, content}, ...], "rejected": [{role, content}, ...]}
    Outputs input_ids, target ids, and loss_mask for both chosen and rejected
    loss_mask is 1 only on the assistant response part, 0 elsewhere
    """
    CHATML_TEMPLATE = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    def __init__(self, jsonl_path, tokenizer, max_length=768):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        if not tokenizer.chat_template:
            tokenizer.chat_template = self.CHATML_TEMPLATE
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # Locate start and end token IDs of the assistant response
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        features = Features({
            'chosen': [{'role': Value('string'), 'content': Value('string')}],
            'rejected': [{'role': Value('string'), 'content': Value('string')}]
        })
        datasets_logging.set_verbosity_error()
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        datasets_logging.set_verbosity_warning()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']

        # Convert conversation to ChatML format text
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # Tokenize and pad to fixed length
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # DPO uses next-token prediction input-target alignment
        # x is the input sequence (excluding the last token), y is the target sequence (excluding the first token)
        # mask aligns with y positions, used to compute DPO loss only on assistant response tokens
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """Generate loss mask: 1 only on assistant response tokens"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
```

## Phase 2: DPO Loss Function

The DPO loss function is the core of the entire training process and the fundamental difference between DPO and PPO. In [Alignment Paradigm Evolution](./alignment-new-paradigms.md), we derived the DPO loss function:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

where $\pi_\theta$ is the policy model (parameters updated during training), $\pi_{\text{ref}}$ is the reference model (frozen parameters), $y_w$ is the chosen response, $y_l$ is the rejected response, and $\beta$ controls how far the model can deviate from the reference model. In implementation, we first compute the log probability at each token position, then sum along the sequence to obtain the log probability of the entire response, and finally plug into the DPO loss formula. The key steps are:

1. **Compute log probabilities**: Apply Softmax to the model output logits, then extract the values at the target token positions to obtain the log probability $\log \pi(y_t | x, y_{<t})$ for each token.
2. **Masked summation**: Use the mask to exclude the question portion, summing only over the response part to obtain the full response log probability $\sum_{t \in \text{assistant}} \log \pi(y_t | x, y_{<t})$.
3. **Compute implicit reward**: $\beta(\log \pi_\theta - \log \pi_{\text{ref}})$, the log-probability ratio between the policy and reference models multiplied by $\beta$.
4. **Compute DPO loss**: $-\log \sigma(\text{chosen\_reward} - \text{rejected\_reward})$.

```python runnable gpuonly extract-class="logits_to_log_probs, dpo_loss"
import torch
import torch.nn.functional as F

def logits_to_log_probs(logits, labels):
    """
    Compute log probabilities at each token position from model output logits

    Args:
        logits: Model output, shape [batch, seq_len, vocab_size]
        labels: Target token ids, shape [batch, seq_len]

    Returns:
        Log probability at each position, shape [batch, seq_len]
    """
    # Compute log_softmax in float32 to avoid numerical overflow from bfloat16 precision
    log_probs = F.log_softmax(logits.float(), dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    Compute DPO loss

    Args:
        ref_log_probs: Reference model log probabilities, shape [batch, seq_len]
        policy_log_probs: Policy model log probabilities, shape [batch, seq_len]
        mask: Loss mask, shape [batch, seq_len]
        beta: DPO temperature parameter

    Returns:
        Scalar loss value
    """
    # Sum along sequence (only at positions where mask is 1)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # Split into chosen and rejected data
    # The first half of the batch is chosen, the second half is rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # Compute implicit reward difference
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios

    # DPO loss = -log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

## Phase 3: DPO Training

DPO training starts from the SFT model. Unlike SFT which requires only one model, DPO maintains two models simultaneously: the policy model $\pi_\theta$ (trainable) and the reference model $\pi_{\text{ref}}$ (frozen). The policy model is initialized from SFT weights and its parameters are updated during training. The reference model is also initialized from SFT weights but its parameters remain frozen, serving as a behavioral baseline. During training, the DPO loss drives the policy model to adjust its generation probabilities relative to the reference model, increasing the probability of chosen responses and decreasing the probability of rejected responses. The table below lists the key engineering decisions for this experiment and their rationale:

| Training Decision | MiniMind | This Experiment | Rationale |
|---------|----------|-------|---------|
| Learning rate | 4e-8 | 1e-5 | MiniMind uses an extremely small learning rate because its DPO training runs thousands of steps on the full 20K dataset, with cosine scheduling providing enough steps for gradual decay. This experiment has the same data volume per batch but uses gradient accumulation (4 steps, effective batch size of 16), resulting in fewer total steps per epoch where a 4e-8 learning rate would barely produce effective parameter updates. However, DPO is very sensitive to parameter updates (even small changes in the log-probability difference between policy and reference models can significantly affect gradients), and too large a learning rate would cause loss oscillation. 1e-5 strikes a balance between effective updates and training stability |
| $\beta$ | 0.15 | 0.1 | $\beta$ controls how far the model can deviate from the reference model. MiniMind uses a slightly conservative 0.15; this experiment lowers it to 0.1 to make the preference signal more pronounced, facilitating observation of training effects |
| Sequence length | 1024 | 768 | Consistent with the SFT experiment. DPO memory usage is about 2.5 times that of SFT (policy model + reference model + chosen + rejected), and longer sequences increase memory pressure |
| Batch size | 4 | 4 | Consistent. Each DPO batch contains both chosen and rejected sequences, making the effective forward propagation batch_size equivalent to 8, leading to higher memory usage |
| Gradient accumulation | 1 | 4 | This experiment uses batch_size = 4 with 4 gradient accumulation steps, resulting in an effective batch_size of 16, close to MiniMind's effective batch size |

::: info Estimated Training Time

`dpo.jsonl` contains approximately 20K preference comparison samples, with a total size of about 53 MB. With sequence length 768, batch size 4 (gradient accumulation x 4, effective batch size 16), and 1 epoch, approximately 16 GB of GPU memory is required (policy model + reference model loaded simultaneously). Training time on an RTX 5080 GPU is approximately 20 minutes.

DPO memory usage is significantly higher than SFT because the training process loads two complete model copies (policy model and reference model) simultaneously, and each batch requires forward propagation on both chosen and rejected sequences. If memory is insufficient, reduce `batch_size` and proportionally increase `accumulation_steps`.

:::

```python runnable gpuonly timeout=unlimited
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

# Import progress reporter
from dmla_progress import ProgressReporter

# Import shared modules
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig
from shared.llm.dpodataset import DPODataset
from shared.llm.logits_to_log_probs import logits_to_log_probs, dpo_loss

# ========== Path Configuration ==========
TOKENIZER_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
DPO_DATA_PATH = os.path.join(DATA_DIR, 'datasets', 'minimind-alignment', 'dpo.jsonl')
SFT_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')
SAVE_DIR = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo')

# ========== Training Hyperparameters ==========
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 768
batch_size = 4             # DPO memory usage is high (dual model + chosen/rejected), batch_size should not be too large
learning_rate = 1e-5       # DPO learning rate (DPO is sensitive to parameters, learning rate should not be too large)
beta = 0.1                 # DPO temperature parameter, controls deviation from the reference model
num_epochs = 1
accumulation_steps = 4     # Gradient accumulation (effective batch_size = 4 x 4 = 16)
grad_clip = 1.0
log_interval = 50
save_interval = 200

# ========== 1. Initialize Environment ==========
progress = ProgressReporter(total_steps=10, description="Preparing DPO training environment")
progress.update(0, message="Checking runtime environment...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Warning: No GPU detected, training will be very slow")

torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

# ========== 2. Load Tokenizer and Data ==========
progress.update(2, message="Loading tokenizer and DPO training data...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
train_ds = DPODataset(DPO_DATA_PATH, tokenizer, max_length=max_seq_len)
print(f"Training samples: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
total_steps_per_epoch = len(train_loader) // accumulation_steps
total_steps = num_epochs * total_steps_per_epoch
print(f"Optimization steps per epoch: {total_steps_per_epoch:,} (mini-steps: {len(train_loader):,} / accumulation: {accumulation_steps})")
print(f"Total optimization steps: {total_steps:,}")

# ========== 3. Create Policy Model and Reference Model ==========
progress.update(4, message="Creating policy model and reference model...")
lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)

# Policy model (trainable)
model = MiniMindForCausalLM(lm_config)

# Reference model (frozen)
ref_model = MiniMindForCausalLM(lm_config)

# Load SFT weights as initialization for both models
weight_path = None
if os.path.exists(SFT_MODEL_PATH):
    weight_path = SFT_MODEL_PATH
else:
    for epoch in [2, 1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', f'sft_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    ref_model.load_state_dict(weights, strict=False)
    print(f"Loaded SFT weights: {weight_path}")
else:
    print("SFT weights not found, using random initialization")

model = model.to(device)
ref_model = ref_model.to(device)
ref_model.eval()
ref_model.requires_grad_(False)

total_params = sum(p.numel() for p in model.parameters())
print(f"Policy model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Reference model parameters: {total_params:,} ({total_params/1e6:.2f}M, frozen)")

# ========== 4. Configure Training Components ==========
progress.update(6, message="Configuring optimizer and learning rate schedule...")

device_type = "cuda" if device.type == "cuda" else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

def get_lr(current_step, total_steps, lr):
    """Linear warmup (first 10%) + cosine decay"""
    warmup_steps = int(0.1 * total_steps)
    if current_step < warmup_steps:
        return lr * current_step / warmup_steps
    progress_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress_ratio)))

os.makedirs(SAVE_DIR, exist_ok=True)
progress.update(8, message="DPO training environment ready")

# ========== 5. Start Training ==========
progress.reset(total_steps=total_steps, description="DPO alignment training")

global_step = 0

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_dpo_loss = 0.0
    log_step_count = 0

    for step, batch in enumerate(train_loader):
        # Concatenate chosen and rejected into one batch, compute both in one forward pass
        x_chosen = batch['x_chosen'].to(device)
        x_rejected = batch['x_rejected'].to(device)
        y_chosen = batch['y_chosen'].to(device)
        y_rejected = batch['y_rejected'].to(device)
        mask_chosen = batch['mask_chosen'].to(device)
        mask_rejected = batch['mask_rejected'].to(device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # Forward pass (mixed precision)
        with autocast_ctx:
            # Reference model forward pass (no gradient computation)
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # Policy model forward pass
            outputs = model(x)
            policy_logits = outputs.logits
            policy_log_probs = logits_to_log_probs(policy_logits, y)

            # Compute DPO loss
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val / accumulation_steps

        # Backward pass
        loss.backward()

        # Record loss (at every mini-step, for logging average)
        current_dpo = dpo_loss_val.item()
        running_dpo_loss += current_dpo
        log_step_count += 1

        # Gradient accumulation + parameter update
        if (step + 1) % accumulation_steps == 0:
            # Learning rate schedule (based on actual optimization steps)
            lr = get_lr(global_step, total_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                avg_dpo = running_dpo_loss / log_step_count
                elapsed = time.time() - epoch_start
                eta_min = elapsed / max(global_step, 1) * (total_steps - global_step) / 60
                print(f"Epoch[{epoch+1}/{num_epochs}] Step[{global_step}/{total_steps}], "
                      f"dpo_loss: {avg_dpo:.4f}, lr: {lr:.8f}, eta: {eta_min:.1f}min")
                progress.update(
                    global_step,
                    message=f"Epoch {epoch+1}/{num_epochs}, Step {global_step}/{total_steps}, DPO Loss={avg_dpo:.4f}",
                    extra_data={"dpo_loss": avg_dpo, "lr": lr, "epoch": epoch + 1}
                )
                running_dpo_loss = 0.0
                log_step_count = 0

            # Periodic model saving
            if global_step % save_interval == 0:
                model.eval()
                save_path = os.path.join(SAVE_DIR, f'dpo_step{global_step}.pth')
                state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
                torch.save(state_dict, save_path)
                print(f"  -> Model saved: step={global_step}, dpo_loss={avg_dpo:.4f}")
                model.train()
                del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask, ref_outputs, ref_logits, ref_log_probs
        del outputs, policy_logits, policy_log_probs, dpo_loss_val

    # Save at end of each epoch
    epoch_time = time.time() - epoch_start
    model.eval()
    epoch_save_path = os.path.join(SAVE_DIR, f'dpo_epoch{epoch+1}.pth')
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, epoch_save_path)
    print(f"\nEpoch {epoch+1} completed, duration {epoch_time/60:.1f}min, model saved")
    model.train()
    del state_dict

# Save final model
final_path = os.path.join(SAVE_DIR, 'full_dpo_768.pth')
state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, final_path)
progress.complete(message=f"DPO training complete! Model saved to {final_path}")
print(f"\nFinal model saved: {final_path}")
```

## Phase 4: Chat Inference

After DPO training, the model has learned to distinguish between good and bad responses on top of the SFT foundation. Compared to the SFT model, the DPO model's response style better reflects the preference tendencies in the training data — responses are more organized, the tone is more appropriate, and refusals are more polite. However, a 64M-parameter model has limited capacity, so the improvement from DPO alignment is less pronounced than what would be seen in a 7B-parameter model, though the training pipeline and principles are the same.

After running the code block below, the model will be loaded into the sandbox. Once loaded, you can chat with the aligned model using the dialog below. When finished, click the Stop button to terminate the inference process.

```python runnable gpuonly mode=chat
import torch
import os
from transformers import AutoTokenizer
from shared.llm.mini_mind_config import MiniMindForCausalLM, MiniMindConfig

# Load tokenizer
tokenizer_path = os.path.join(DATA_DIR, 'datasets', 'minimind-pretrain')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if not tokenizer.chat_template:
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

# Load DPO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# Find available DPO weights
dpo_model_path = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo', 'full_dpo_768.pth')
weight_path = None
if os.path.exists(dpo_model_path):
    weight_path = dpo_model_path
else:
    for epoch in [1]:
        ckp = os.path.join(DATA_DIR, 'models', 'minimind', 'dpo', f'dpo_epoch{epoch}.pth')
        if os.path.exists(ckp):
            weight_path = ckp
            break

if not weight_path:
    # Fall back to SFT model
    sft_path = os.path.join(DATA_DIR, 'models', 'minimind', 'sft', 'full_sft_768.pth')
    if os.path.exists(sft_path):
        weight_path = sft_path
        print("DPO model not found, falling back to SFT model")

if weight_path:
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
    print(f"Loaded weights: {weight_path}")
else:
    print("No model weights found, using random initialization")

model = model.half().to(device).eval()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print("Chat service ready")

# Define chat function
def chat(user_message, history=None):
    if history is None:
        history = []
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return response.strip()
```

::: details After running the code above, click here to chat
<ChatDemo />
:::

## Conclusion

This experiment performed DPO alignment training on top of the SFT model. After training, the following files are saved to the data directory:

- **Model files**:
    - `<DATA_DIR>/models/minimind/dpo/full_dpo_768.pth` - Final DPO weights (FP16 precision)
    - `<DATA_DIR>/models/minimind/dpo/dpo_epoch*.pth` - Checkpoint at the end of each epoch
    - `<DATA_DIR>/models/minimind/dpo/dpo_step*.pth` - Intermediate training checkpoints

DPO training advances the model from "learning to answer" to "learning to distinguish good from bad answers." Compared to RLHF's three-model architecture (policy model + reward model + reference model), DPO requires only two models (policy model + reference model), bypassing the need for reward model training and the instability of PPO, significantly lowering the engineering barrier for alignment training. DPO's limitation is that the $\beta$ parameter is fixed, less flexible than PPO's adaptive KL penalty, and log-probability computation over long sequences can be unstable. Methods introduced in [Alignment Paradigm Evolution](./alignment-new-paradigms.md), such as KTO and GRPO, further simplify the alignment training pipeline from different perspectives.

At this point, we have completed the full pipeline of language model training. Pre-training endows the model with language capability, SFT gives it conversational ability, and DPO provides preference alignment. These three stages build upon each other progressively, with each step grounded in the foundation laid by the previous one.
