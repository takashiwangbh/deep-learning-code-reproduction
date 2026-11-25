"""Direct Preference Optimization 的核心损失，不依赖 trl/transformers。"""

import torch
import torch.nn.functional as F


def sequence_log_probability(logits: torch.Tensor, targets: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """计算每条回答的 log π(y|x)。调用者传入已经右移对齐的 logits/targets。

    logits: [B,T,V]；targets/mask: [B,T]。mask=1 只统计回答 token，不统计提示词。
    """
    # 常见数据加载器用 -100 填充被遮蔽的位置；先改成合法索引再 gather。
    safe_targets = targets.masked_fill(mask == 0, 0)
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, safe_targets[..., None]).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """输入均为 [B] 的完整回答 log probability，参考模型的值应先 detach。"""
    if beta <= 0:
        raise ValueError("beta must be positive")
    log_ratio = (policy_chosen - policy_rejected) - (
        reference_chosen.detach() - reference_rejected.detach()
    )
    return -F.logsigmoid(beta * log_ratio).mean()


if __name__ == "__main__":
    chosen = torch.tensor([-4.0, -3.0], requires_grad=True)
    rejected = torch.tensor([-5.0, -6.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, torch.tensor([-4.5, -4.0]),
                    torch.tensor([-5.0, -5.0]))
    loss.backward()
    print("DPO loss:", round(loss.item(), 4), "chosen gradient:", chosen.grad.tolist())
