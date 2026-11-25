"""Core mechanism checks for language, forecasting and recommendation baselines."""

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder: str, filename: str):
    path = ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LanguageBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_elmo_bidirectional_layer_mix(self):
        code = load("ELMo", "elmo.py")
        model = code.ELMo(20, width=8)
        ids = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
        vectors, weights = model.contextual_embeddings(ids)
        self.assertEqual(tuple(vectors.shape), (2, 4, 16))
        self.assertTrue(torch.allclose(weights.sum(), torch.tensor(1.)))
        self.assertTrue(torch.isfinite(model.language_model_loss(ids)))
        changed = ids.clone()
        changed[:, -1] = 12
        original = model.contextual_embeddings(ids)[0]
        altered = model.contextual_embeddings(changed)[0]
        self.assertFalse(torch.allclose(original[:, 1], altered[:, 1]))  # right context matters

    def test_ulmfit_stages_and_rates(self):
        code = load("ULMFiT", "ulmfit.py")
        model = code.ULMFiT(30, 3, width=8)
        model.gradual_unfreeze(0)
        self.assertFalse(any(p.requires_grad for p in model.recurrent[-1].parameters()))
        model.gradual_unfreeze(1)
        self.assertTrue(all(p.requires_grad for p in model.recurrent[-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.recurrent[-2].parameters()))
        model.gradual_unfreeze(4)
        rates = [group["lr"] for group in model.discriminative_groups(.01)]
        self.assertEqual(rates, sorted(rates))
        self.assertGreater(code.slanted_triangular_lr(10, 100), code.slanted_triangular_lr(0, 100))
        self.assertGreater(code.slanted_triangular_lr(10, 100), code.slanted_triangular_lr(100, 100))

    def test_roberta_dynamic_mask_no_nsp(self):
        code = load("RoBERTa", "roberta.py")
        ids = torch.tensor([[2, *([4, 5, 6, 7] * 10), 3]])
        torch.manual_seed(1)
        first, labels_a = code.dynamic_mask(ids, 30)
        torch.manual_seed(2)
        second, labels_b = code.dynamic_mask(ids, 30)
        self.assertFalse(torch.equal(labels_a, labels_b))
        self.assertTrue(labels_a[labels_a.ne(-100)].ge(4).all())
        self.assertEqual(int(labels_a[0, 0]), -100)
        self.assertTrue(torch.equal(first[labels_a.eq(-100)], ids[labels_a.eq(-100)]))
        self.assertTrue(torch.equal(second[labels_b.eq(-100)], ids[labels_b.eq(-100)]))
        model = code.RoBERTa(vocab=30, length=ids.shape[1], width=16)
        self.assertTrue(torch.isfinite(model.training_loss(ids)[0]))

    def test_bart_span_and_decoder_causality(self):
        code = load("BART", "bart.py")
        clean = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 10, 11, 3]])
        corrupted = code.span_infilling(clean, 2, 4)
        self.assertEqual(corrupted.shape[1], clean.shape[1] - 1)
        self.assertTrue((corrupted[:, 2] == 1).all())
        model = code.BART(vocab=32, length=8, width=16)
        decoder = clean[:, :-1]
        changed = decoder.clone()
        changed[:, -1] = 12
        self.assertTrue(torch.allclose(model(corrupted, decoder)[:, :-1],
                                       model(corrupted, changed)[:, :-1], atol=1e-5))

    def test_xlnet_query_cannot_see_target(self):
        code = load("XLNet", "xlnet.py")
        model = code.XLNet(vocab=30, length=3, width=16)
        order = torch.tensor([[1, 0, 2]])
        ids = torch.tensor([[4, 5, 6]])
        before, _ = model(ids, order)
        changed = ids.clone()
        changed[0, 1] = 9  # first factorized target
        after, _ = model(changed, order)
        self.assertTrue(torch.allclose(before[0, 1], after[0, 1], atol=1e-6))
        changed = ids.clone()
        changed[0, 0] = 9  # second factorized target
        after, _ = model(changed, order)
        self.assertTrue(torch.allclose(before[0, 0], after[0, 0], atol=1e-6))
        self.assertTrue(torch.isfinite(model.training_loss(ids, order)[0]))

    def test_electra_actual_replacements(self):
        code = load("ELECTRA", "electra.py")
        model = code.ELECTRA(vocab=30, length=8)
        clean = torch.tensor([[2, 4, 5, 6, 3, 0], [2, 7, 8, 9, 3, 0]])
        loss, details = model.training_loss(clean, mask_probability=.8)
        self.assertTrue(torch.equal(details["replaced"], details["corrupted"].ne(clean)))
        self.assertTrue(torch.equal(details["corrupted"][:, -1], clean[:, -1]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.generator.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.discriminator.parameters()))

    def test_tft_variable_weights_and_time_mask(self):
        code = load("Temporal Fusion Transformer", "tft.py")
        model = code.TemporalFusionTransformer(2, 3, 2, width=16)
        static, past = torch.randn(2, 2), torch.randn(2, 5, 3)
        future = torch.randn(2, 4, 2)
        forecast, explanation = model(static, past, future)
        self.assertEqual(tuple(forecast.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(explanation["past_selection"].sum(-1),
                                       torch.ones(2, 5), atol=1e-6))
        self.assertTrue(torch.allclose(explanation["future_selection"].sum(-1),
                                       torch.ones(2, 4), atol=1e-6))
        self.assertTrue((explanation["temporal_attention"][:, 0, 6:] == 0).all())
        changed = future.clone()
        changed[:, 2:] += 100
        self.assertTrue(torch.allclose(forecast[:, 0], model(static, past, changed)[0][:, 0], atol=1e-6))

    def test_patchtst_channel_independence(self):
        code = load("PatchTST", "patchtst.py")
        model = code.PatchTST(16, 4, width=16)
        series = torch.randn(2, 16, 3)
        first = model(series)
        changed = series.clone()
        changed[:, :, 2] += 100
        second = model(changed)
        self.assertTrue(torch.allclose(first[:, :, :2], second[:, :, :2], atol=1e-6))

    def test_din_target_attention_and_pad(self):
        code = load("DIN", "din.py")
        model = code.DIN(items=20, width=16)
        history = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        logits, weights = model(history, torch.tensor([6, 7]))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue((weights[history == 0] == 0).all())
        _, different = model(history, torch.tensor([8, 9]))
        self.assertFalse(torch.allclose(weights, different))

    def test_sasrec_strict_prefix(self):
        code = load("SASRec", "sasrec.py")
        model = code.SASRec(items=20, length=5, width=16)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        changed = ids.clone()
        changed[:, 3:] = torch.tensor([[7, 8]])
        self.assertTrue(torch.allclose(model(ids)[:, :3], model(changed)[:, :3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
