"""无需下载数据的结构与关键行为检查：python -m unittest -v checks.test_sequence"""

import unittest
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for project in (
    "Recurrent Neural Networks", "Attention Seq2Seq", "Transformer", "GPT", "BERT",
    "Llama-style Language Model", "LoRA",
):
    sys.path.insert(0, str(ROOT / project))

from attention_seq2seq import AttentionSeq2Seq
from bert import BERTForPretraining
from gpt import GPT
from llama_style import LlamaStyleLM
from lora import LoRALinear
from recurrent import RecurrentLanguageModel
from transformer import TransformerSeq2Seq


class SequenceModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_recurrent_cells_and_gradients(self) -> None:
        for cell_type in ("rnn", "lstm", "gru"):
            with self.subTest(cell_type=cell_type):
                model = RecurrentLanguageModel(17, 8, 12, cell_type)
                logits, state = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
                self.assertEqual(tuple(logits.shape), (2, 3, 17))
                self.assertEqual(tuple((state[0] if cell_type == "lstm" else state).shape), (2, 12))
                logits.sum().backward()
                self.assertIsNotNone(model.cell.input_to_gates.weight.grad if cell_type != "rnn" else model.cell.input_to_hidden.weight.grad)

    def test_additive_attention_masks_padding(self) -> None:
        model = AttentionSeq2Seq(20, 23, embedding_size=8, hidden_size=12)
        src = torch.tensor([[1, 2, 0], [3, 4, 5]])
        tgt = torch.tensor([[1, 6], [1, 7]])
        states, hidden, valid = model.encode(src)
        _, weights = model.attention(hidden, states, valid)
        self.assertAlmostEqual(weights[0, 2].item(), 0.0, places=7)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2)))
        logits = model(src, tgt)
        self.assertEqual(tuple(logits.shape), (2, 2, 23))
        logits.mean().backward()
        self.assertIsNotNone(model.attention.score.weight.grad)

    def test_transformer_decoder_is_causal(self) -> None:
        model = TransformerSeq2Seq(20, 20, d_model=16, num_heads=4, num_layers=1, d_ff=32, dropout=0)
        model.eval()
        src = torch.tensor([[2, 3, 0]])
        a = model(src, torch.tensor([[1, 4, 5]]))
        b = model(src, torch.tensor([[1, 4, 9]]))
        self.assertTrue(torch.allclose(a[:, :2], b[:, :2], atol=1e-6))
        a.sum().backward()
        self.assertIsNotNone(model.decoder[0].cross_attention.query.weight.grad)

    def test_transformer_all_masked_first_row_cannot_read_future(self) -> None:
        model = TransformerSeq2Seq(20, 20, d_model=16, num_heads=4, num_layers=2, d_ff=32, dropout=0)
        model.eval()
        src = torch.tensor([[2, 3, 0]])
        a = model(src, torch.tensor([[0, 1, 2]]))
        b = model(src, torch.tensor([[0, 1, 9]]))
        self.assertTrue(torch.allclose(a[:, :2], b[:, :2], atol=1e-6))

    def test_gpt_cache_matches_full_forward(self) -> None:
        model = GPT(23, block_size=12, d_model=16, num_heads=4, num_layers=2, dropout=0)
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        whole = model(tokens)
        first, cache = model(tokens[:, :2], use_cache=True)
        second, cache = model(tokens[:, 2:], past_key_values=cache, use_cache=True)
        self.assertTrue(torch.allclose(whole, torch.cat((first, second), dim=1), atol=1e-5))
        changed = model(torch.tensor([[1, 2, 3, 8]]))
        self.assertTrue(torch.allclose(whole[:, :3], changed[:, :3], atol=1e-6))
        self.assertEqual(tuple(model.generate(tokens[:, :2], 2).shape), (1, 4))
        whole.mean().backward()
        self.assertIsNotNone(model.blocks[0].attention.qkv.weight.grad)

    def test_bert_pretraining_outputs(self) -> None:
        model = BERTForPretraining(29, max_length=8, d_model=16, num_heads=4, num_layers=1, d_ff=32, dropout=0)
        model.eval()
        ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
        mlm, nsp = model(ids)
        self.assertEqual(tuple(mlm.shape), (2, 4, 29))
        self.assertEqual(tuple(nsp.shape), (2, 2))
        changed = ids.clone()
        changed[0, 2] = 8
        changed_mlm, _ = model(changed)
        # 与 GPT 不同，第一位置可读取后续位置，因此输出应改变。
        self.assertFalse(torch.allclose(mlm[0, 0], changed_mlm[0, 0], atol=1e-7))
        mlm[:, 1].sum().backward()
        self.assertIsNotNone(model.encoder.blocks[0].attention.query.weight.grad)

    def test_llama_style_cache_and_causality(self) -> None:
        model = LlamaStyleLM(31, d_model=16, num_heads=4, num_kv_heads=2, num_layers=2, hidden_size=32, max_length=8)
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        whole = model(tokens)
        first, cache = model(tokens[:, :2], use_cache=True)
        second, cache = model(tokens[:, 2:], past_key_values=cache, use_cache=True)
        self.assertEqual(cache[0][0].size(1), 2)  # 4 个 Q 头只缓存 2 个 KV 头
        self.assertTrue(torch.allclose(whole, torch.cat((first, second), dim=1), atol=1e-5))
        changed = model(torch.tensor([[1, 2, 3, 7]]))
        self.assertTrue(torch.allclose(whole[:, :3], changed[:, :3], atol=1e-6))
        whole.mean().backward()
        self.assertIsNotNone(model.blocks[0].attention.q_proj.weight.grad)

    def test_lora_preserves_base_and_merges(self) -> None:
        base = nn.Linear(8, 6)
        adapter = LoRALinear.from_linear(base, rank=2, alpha=4)
        x = torch.randn(3, 8)
        self.assertTrue(torch.allclose(base(x), adapter(x), atol=1e-7))
        adapter(x).sum().backward()
        self.assertFalse(adapter.base.weight.requires_grad)
        self.assertIsNotNone(adapter.B.grad)
        with torch.no_grad():
            adapter.B.add_(0.1)
        expected = adapter(x)
        adapter.merge()
        self.assertTrue(torch.allclose(expected, adapter(x), atol=1e-6))
        adapter.unmerge()
        self.assertTrue(torch.allclose(expected, adapter(x), atol=1e-6))

    def test_lora_merged_state_survives_reload(self) -> None:
        adapter = LoRALinear(8, 6, rank=2)
        with torch.no_grad():
            adapter.B.add_(0.1)
        adapter.merge()
        x = torch.randn(3, 8)
        expected = adapter(x)
        restored = LoRALinear(8, 6, rank=2)
        restored.load_state_dict(adapter.state_dict())
        self.assertTrue(restored.merged)
        self.assertTrue(torch.allclose(expected, restored(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
