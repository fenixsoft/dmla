# logits_to_log_probs, dpo_loss 定义
# 从文档自动提取生成

import torch
import torch.nn.functional as F

def logits_to_log_probs(logits, labels):
    """
    从模型输出的 logits 计算每个 token 位置的对数概率

    Args:
        logits: 模型输出, shape [batch, seq_len, vocab_size]
        labels: 目标 token ids, shape [batch, seq_len]

    Returns:
        每个位置的对数概率, shape [batch, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO 损失

    Args:
        ref_log_probs: 参考模型的对数概率, shape [batch, seq_len]
        policy_log_probs: 策略模型的对数概率, shape [batch, seq_len]
        mask: loss 掩码, shape [batch, seq_len]
        beta: DPO 温度参数

    Returns:
        标量损失值
    """
    # 沿序列求和（仅在 mask 为 1 的位置）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # batch 中前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算隐式奖励差值
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios

    # DPO 损失 = -log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
