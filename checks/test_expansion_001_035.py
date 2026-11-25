"""Smoke and mechanism checks for catalogue entries 001–035."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder: str, filename: str):
    path = ROOT / folder / filename
    spec = spec_from_file_location("catalog_" + folder.replace(" ", "_").replace("-", "_"), path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FoundationChecks(unittest.TestCase):
    def test_perceptron_learns_separable_points(self):
        model = load("Perceptron", "perceptron.py").Perceptron(2)
        x = np.array([[2., 1.], [1., 2.], [-1., -2.], [-2., -1.]])
        y = np.array([1, 1, -1, -1])
        self.assertEqual(model.fit(x, y)[-1], 0)
        np.testing.assert_array_equal(model.predict(x), y)

    def test_recurrent_states_and_gate_math(self):
        x = torch.randn(2, 5, 4)
        rnn = load("Vanilla RNN", "vanilla_rnn.py").VanillaRNN(4, 8)
        lstm_module = load("LSTM", "lstm.py")
        gru_module = load("GRU", "gru.py")
        for model in (rnn, lstm_module.LSTM(4, 8), gru_module.GRU(4, 8)):
            sequence, state = model(x)
            self.assertEqual(sequence.shape, (2, 5, 8))
            self.assertTrue(torch.isfinite(sequence).all())
            sequence.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
        cell = lstm_module.LSTMCell(4, 8)
        for parameter in cell.parameters():
            torch.nn.init.zeros_(parameter)
        h, c = cell(torch.zeros(2, 4), (torch.zeros(2, 8), torch.ones(2, 8)))
        self.assertTrue(torch.allclose(c, torch.full_like(c, 0.5)))
        self.assertTrue(torch.allclose(h, 0.5 * torch.tanh(c)))
        gru_cell = gru_module.GRUCell(4, 8)
        for parameter in gru_cell.parameters():
            torch.nn.init.zeros_(parameter)
        self.assertTrue(torch.allclose(gru_cell(torch.zeros(2, 4), torch.ones(2, 8)),
                                       torch.full((2, 8), 0.5)))

    def test_siamese_shares_encoder_and_matching_loss(self):
        module = load("Siamese Network", "siamese.py")
        model = module.SiameseNetwork()
        x = torch.randn(2, 1, 16, 16)
        a, b = model(x, x)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(module.contrastive_loss(a, b, torch.ones(2)).item(), 0.0)

    def test_attention_is_causal(self):
        module = load("Self-Attention", "self_attention.py")
        model = module.MultiHeadSelfAttention(16, 4).eval()
        x = torch.randn(2, 5, 16)
        y, weights = model(x, causal=True)
        self.assertEqual(y.shape, (2, 5, 16))
        self.assertTrue(torch.equal(weights.triu(1), torch.zeros_like(weights)))
        changed = x.clone()
        changed[:, -1] += 100
        newer, _ = model(changed, causal=True)
        self.assertTrue(torch.allclose(y[:, :-1], newer[:, :-1], atol=1e-6))

    def test_residual_projection(self):
        model = load("Residual Learning", "residual_learning.py").ResidualBlock(8, 16, 2)
        y = model(torch.randn(2, 8, 16, 16))
        self.assertEqual(y.shape, (2, 16, 8, 8))


class VisionChecks(unittest.TestCase):
    def test_cnn_families_forward_and_gradients(self):
        specifications = [
            ("CNN", "cnn.py", "BasicCNN", (2, 1, 28, 28)),
            ("NiN", "nin.py", "NiN", (2, 3, 32, 32)),
            ("ResNeXt", "resnext.py", "ResNeXt", (2, 3, 32, 32)),
            ("SENet", "senet.py", "SENet", (2, 3, 32, 32)),
            ("SqueezeNet", "squeezenet.py", "SqueezeNet", (2, 3, 32, 32)),
            ("MobileNet", "mobilenet.py", "MobileNet", (2, 3, 32, 32)),
            ("ShuffleNet", "shufflenet.py", "ShuffleNet", (2, 3, 32, 32)),
            ("Xception", "xception.py", "Xception", (2, 3, 32, 32)),
            ("EfficientNet", "efficientnet.py", "EfficientNet", (2, 3, 32, 32)),
            ("ConvNeXt", "convnext.py", "ConvNeXt", (2, 3, 32, 32)),
        ]
        for folder, file, cls, shape in specifications:
            with self.subTest(folder=folder):
                model = getattr(load(folder, file), cls)()
                logits = model(torch.randn(*shape))
                self.assertEqual(logits.shape, (2, 10))
                self.assertTrue(torch.isfinite(logits).all())
                logits.sum().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_channel_shuffle_changes_group_assignment_without_losing_values(self):
        shuffle = load("ShuffleNet", "shufflenet.py").channel_shuffle
        x = torch.arange(8.).reshape(1, 8, 1, 1)
        result = shuffle(x, 2)
        self.assertEqual(sorted(result.flatten().tolist()), sorted(x.flatten().tolist()))
        self.assertNotEqual(result.flatten().tolist(), x.flatten().tolist())


class LanguageAndSeriesChecks(unittest.TestCase):
    def test_embeddings_objectives(self):
        word = load("Word2Vec", "word2vec.py").SkipGramNegativeSampling(20, 8)
        loss = word(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([[5, 6], [7, 8]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(word.target_embeddings.weight.grad)
        glove = load("GloVe", "glove.py").GloVe(20, 8)
        loss = glove(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5., 10.]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(glove.word_vectors().shape, (20, 8))

    def test_textcnn_and_bilstm(self):
        text = load("TextCNN", "textcnn.py").TextCNN(30, 8, 3)
        self.assertEqual(text(torch.randint(0, 30, (2, 7))).shape, (2, 3))
        bilstm = load("BiLSTM", "bilstm.py").BiLSTM(4, 8).eval()
        x = torch.randn(2, 5, 4)
        output = bilstm(x)
        self.assertEqual(output.shape, (2, 5, 16))
        altered = x.clone()
        altered[:, -1] += 5
        changed = bilstm(altered)
        self.assertTrue(torch.allclose(output[:, 0, :8], changed[:, 0, :8]))
        self.assertFalse(torch.allclose(output[:, 0, 8:], changed[:, 0, 8:]))

    def test_seq2seq_and_additive_alignment(self):
        source = torch.randint(0, 20, (2, 5))
        target = torch.randint(0, 25, (2, 4))
        plain = load("Seq2Seq", "seq2seq.py").Seq2Seq(20, 25, 8)
        self.assertEqual(plain(source, target).shape, (2, 4, 25))
        self.assertEqual(plain.generate(source, bos_token=1, steps=3).shape, (2, 3))
        attention = load("Bahdanau Attention", "bahdanau_attention.py").BahdanauSeq2Seq(20, 25, 8)
        valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        logits, weights = attention(source, target, valid)
        self.assertEqual(logits.shape, (2, 4, 25))
        self.assertTrue(torch.equal(weights.masked_select(~valid[:, None, :]), torch.zeros(12)))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 4)))
        changed_source = source.clone()
        changed_source[~valid] = (changed_source[~valid] + 7) % 20
        unchanged_logits, unchanged_weights = attention(changed_source, target, valid)
        self.assertTrue(torch.allclose(logits, unchanged_logits, atol=1e-6))
        self.assertTrue(torch.allclose(weights, unchanged_weights, atol=1e-6))

    def test_causal_convolutions(self):
        tcn = load("TCN", "tcn.py").TCN(4, 8, 3).eval()
        x = torch.randn(2, 9, 4)
        initial = tcn(x)
        changed = x.clone()
        changed[:, -1] += 100
        self.assertEqual(initial.shape, (2, 9, 3))
        self.assertTrue(torch.allclose(initial[:, :-1], tcn(changed)[:, :-1], atol=1e-6))
        wave = load("WaveNet", "wavenet.py").WaveNet().eval()
        tokens = torch.randint(0, 32, (2, 10))
        first = wave(tokens)
        altered = tokens.clone()
        altered[:, -1] = (altered[:, -1] + 1) % 32
        self.assertEqual(first.shape, (2, 10, 32))
        self.assertTrue(torch.allclose(first[:, :-1], wave(altered)[:, :-1], atol=1e-6))

    def test_probabilistic_and_residual_forecasting(self):
        deep = load("DeepAR", "deepar.py").DeepAR(8)
        data = torch.randn(2, 8)
        mean, scale = deep(data)
        self.assertEqual(mean.shape, (2, 7))
        self.assertTrue(torch.all(scale > 0))
        loss = deep.negative_log_likelihood(data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(deep.sample(data, 3).shape, (2, 3))
        beats = load("N-BEATS", "nbeats.py").NBeats(12, 4, width=8)
        forecast, residual = beats(torch.randn(2, 12))
        self.assertEqual(forecast.shape, (2, 4))
        self.assertEqual(residual.shape, (2, 12))
        forecast.sum().backward()


if __name__ == "__main__":
    unittest.main()
