"""运行：python smoke_test.py。小张量检查形状、梯度和因果性。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for project in ("Modern Decoder", "Mamba", "Gated DeltaNet", "Gated DeltaNet-2", "DPO"):
    sys.path.insert(0, str(ROOT / project))

import torch
import torch.nn.functional as F

from dpo import dpo_loss, sequence_log_probability
from gated_delta import HybridDeltaAttentionLM, gated_delta_scan
from gated_delta2 import GatedDelta2LanguageModel, HeadRMSNorm, gated_delta2_scan
from mamba_selective_ssm import MambaLanguageModel
from modern_decoder import ModernDecoderLM


def check_causality(model, tokens):
    model.eval()
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 23
    with torch.no_grad():
        a = model(tokens)
        b = model(changed)
        if isinstance(a, tuple):
            a, b = a[0], b[0]
    torch.testing.assert_close(a[:, :4], b[:, :4], rtol=1e-5, atol=1e-5)


def main():
    torch.manual_seed(3)
    tokens = torch.randint(0, 23, (2, 7))
    decoder = ModernDecoderLM(23, dim=16, depth=2, query_heads=4, kv_heads=2,
                              hidden=32, num_experts=3)
    logits, aux = decoder(tokens)
    assert logits.shape == (2, 7, 23)
    assert aux.ndim == 0
    (F.cross_entropy(logits[:, :-1].reshape(-1, 23), tokens[:, 1:].reshape(-1))
     + 0.01 * aux).backward()
    assert decoder.blocks[0].attention.q_proj.weight.grad is not None
    assert decoder.blocks[0].ffn.router.weight.grad is not None
    check_causality(decoder, tokens)

    mamba = MambaLanguageModel(23, dim=12, depth=2, state_dim=4)
    output = mamba(tokens)
    assert output.shape == (2, 7, 23)
    F.cross_entropy(output[:, :-1].reshape(-1, 23), tokens[:, 1:].reshape(-1)).backward()
    assert mamba.blocks[0].A_log.grad is not None
    check_causality(mamba, tokens)

    # 在同一 key 上写入新值，beta=1 且 alpha=1 时旧关联被准确替换。
    key = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    value = torch.tensor([[[[2., 0.]], [[3., 0.]]]])
    scan = gated_delta_scan(key, key, value, torch.ones(1, 2, 1), torch.ones(1, 2, 1))
    torch.testing.assert_close(scan[0, :, 0, 0], torch.tensor([2., 3.]))
    hybrid = HybridDeltaAttentionLM(23, dim=16, depth=2, heads=4, kv_heads=2, hidden=32)
    hybrid_logits = hybrid(tokens)
    assert hybrid_logits.shape == (2, 7, 23)
    hybrid_logits.mean().backward()
    assert hybrid.blocks[0].mixer.qkv.weight.grad is not None
    assert hybrid.blocks[0].mixer.output_gate.weight.grad is not None
    check_causality(hybrid, tokens)

    # 将两个逐通道门绑成同一个标量 β、把逐通道衰减绑成 α，
    # Gated Delta Rule-2 应退化为 Gated DeltaNet 的原始递推。
    q, k, v = (torch.randn(2, 4, 2, 3) for _ in range(3))
    scalar_alpha = torch.rand(2, 4, 2)
    scalar_beta = torch.rand(2, 4, 2)
    old_rule = gated_delta_scan(q, k, v, scalar_alpha, scalar_beta)
    expanded_alpha = scalar_alpha[..., None].expand_as(q)
    expanded_beta = scalar_beta[..., None].expand_as(q)
    new_rule = gated_delta2_scan(q, k, v, expanded_alpha, expanded_beta, expanded_beta)
    torch.testing.assert_close(new_rule, old_rule, atol=1e-6, rtol=1e-6)
    # 大值在 fp16 中直接平方会溢出；归一化必须使用 fp32 统计。
    normed = HeadRMSNorm(1, 4)(torch.full((1, 1, 1, 4), 1000., dtype=torch.float16))
    assert torch.isfinite(normed).all()
    q16 = F.normalize(torch.randn(1, 128, 1, 4), dim=-1).half()
    k16 = F.normalize(torch.randn(1, 128, 1, 4), dim=-1).half()
    v16 = torch.randn(1, 128, 1, 4).half()
    alpha16 = torch.full_like(q16, 0.95)
    erase16 = torch.full_like(q16, 0.5)
    write16 = torch.full_like(q16, 0.5)
    half_output = gated_delta2_scan(q16, k16, v16, alpha16, erase16, write16)
    wide_output = gated_delta2_scan(*(a.float() for a in (
        q16, k16, v16, alpha16, erase16, write16
    )))
    torch.testing.assert_close(half_output.float(), wide_output, atol=1e-3, rtol=1e-3)
    delta2 = GatedDelta2LanguageModel(23, dim=16, depth=2, heads=4, hidden=32)
    delta2_logits = delta2(tokens)
    assert delta2_logits.shape == (2, 7, 23)
    delta2_logits.mean().backward()
    assert delta2.blocks[0].mixer.erase_proj.weight.grad is not None
    assert delta2.blocks[0].mixer.write_proj.weight.grad is not None
    check_causality(delta2, tokens)

    dummy_logits = torch.zeros(2, 3, 5)
    targets = torch.tensor([[-100, 0, 0], [-100, -100, 0]])
    logp = sequence_log_probability(dummy_logits, targets,
                                    torch.tensor([[0, 1, 1], [0, 0, 1]]))
    torch.testing.assert_close(logp, torch.tensor([-2, -1.]) * torch.log(torch.tensor(5.)))
    before = dpo_loss(torch.tensor([-2.]), torch.tensor([-2.]),
                      torch.tensor([-2.]), torch.tensor([-2.]))
    after = dpo_loss(torch.tensor([-1.]), torch.tensor([-3.]),
                     torch.tensor([-2.]), torch.tensor([-2.]))
    assert after < before
    print("modern: decoder/MoE/Mamba/GatedDelta/DeltaNet-2/DPO checks passed")


if __name__ == "__main__":
    main()
