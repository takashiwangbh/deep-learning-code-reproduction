"""生成模型扩展：验证关键算法性质，而不依赖预训练权重。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder, filename):
    spec = spec_from_file_location(folder.replace(" ", "_"), ROOT / folder / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerativeExpansionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_dae_clean_target_and_gradient(self):
        m = load("Denoising Autoencoder", "dae.py").DenoisingAutoencoder()
        x = torch.rand(2, 16)
        loss = m.loss(x)
        loss.backward()
        self.assertEqual(m(x).shape, x.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(m.encoder[0].weight.grad)

    def test_cvae_conditional_prior_and_kl(self):
        m = load("Conditional Variational Autoencoder", "cvae.py").ConditionalVAE()
        x, y = torch.rand(3, 16), torch.tensor([0, 1, 2])
        loss, reconstruction, kl = m.loss(x, y)
        loss.backward()
        self.assertEqual(m.sample(y).shape, x.shape)
        self.assertGreaterEqual(kl.item(), -1e-6)
        self.assertIsNotNone(m.prior[0].weight.grad)

    def test_vq_nearest_code_and_straight_through(self):
        mod = load("VQ-VAE", "vq_vae.py")
        quantizer = mod.VectorQuantizer(codes=2, dim=1)
        with torch.no_grad():
            quantizer.codebook.weight.copy_(torch.tensor([[0.0], [2.0]]))
        encoded = torch.tensor([[[[0.1, 1.9]]]], requires_grad=True)
        quantized, indices, loss = quantizer(encoded)
        self.assertEqual(indices.tolist(), [[[0, 1]]])
        self.assertTrue(torch.allclose(quantized.detach(), torch.tensor([[[[0.0, 2.0]]]])))
        (quantized.sum() + loss).backward()
        self.assertIsNotNone(encoded.grad)
        self.assertIsNotNone(quantizer.codebook.weight.grad)

    def test_realnvp_inverse_and_jacobian(self):
        mod = load("RealNVP", "realnvp.py")
        flow = mod.RealNVP(dim=2, layers=2)
        with torch.no_grad():
            for layer in flow.layers:
                layer.network[-1].bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.2]))
        x = torch.randn(2, 2)
        z, logdet = flow(x)
        self.assertTrue(torch.allclose(flow.inverse(z), x, atol=1e-5))
        one = x[0].detach().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda value: flow(value[None])[0][0], one)
        self.assertAlmostEqual(logdet[0].item(), torch.linalg.slogdet(jac)[1].item(), places=4)
        self.assertTrue(torch.isfinite(flow.log_prob(x)).all())

    def test_glow_inverse_and_log_density(self):
        mod = load("Glow", "glow.py")
        x = torch.randn(2, 1, 8, 8)
        flow = mod.Glow(steps=2)
        z, logdet = flow(x)
        self.assertEqual(z.shape, (2, 4, 4, 4))
        self.assertTrue(torch.allclose(flow.inverse(z), x, atol=1e-5))
        self.assertTrue(torch.isfinite(flow.log_prob(x)).all())
        self.assertEqual(logdet.shape, (2,))

    def test_pixelcnn_causal_mask(self):
        mod = load("PixelCNN", "pixelcnn.py")
        model = mod.PixelCNN(hidden=8, blocks=2)
        original = torch.zeros(1, 1, 5, 5)
        changed = original.clone()
        changed[0, 0, 2, 3] = 1
        a, b = model(original), model(changed)
        self.assertTrue(torch.allclose(a[0, 0, 2, :4], b[0, 0, 2, :4], atol=1e-6))
        self.assertTrue(torch.allclose(a[0, 0, :2], b[0, 0, :2], atol=1e-6))
        self.assertEqual(model.input.mask[0, 0, 3, 3].item(), 0)
        self.assertEqual(model.hidden[0][1].mask[0, 0, 1, 1].item(), 1)

    def test_gan_losses_and_gradients(self):
        mod = load("GAN", "gan.py")
        g, d = mod.Generator(), mod.Discriminator()
        fake = g(torch.randn(3, 8))
        loss = mod.discriminator_loss(d, torch.randn(3, 16), fake) + mod.generator_loss(d, fake)
        loss.backward()
        self.assertEqual(fake.shape, (3, 16))
        self.assertIsNotNone(g.network[0].weight.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_wgan_gp_unit_gradient(self):
        mod = load("WGAN-GP", "wgan_gp.py")
        critic = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            critic.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        penalty = mod.gradient_penalty(critic, torch.randn(3, 4), torch.randn(3, 4))
        self.assertAlmostEqual(penalty.item(), 0.0, places=6)
        g = mod.Generator(noise_dim=8, data_dim=4)
        fake = g(torch.randn(3, 8))
        loss = mod.critic_loss(critic, torch.randn(3, 4), fake) + mod.generator_loss(critic, fake)
        self.assertTrue(torch.isfinite(loss))

    def test_cgan_condition(self):
        mod = load("CGAN", "cgan.py")
        g, d = mod.ConditionalGenerator(), mod.ConditionalDiscriminator()
        z = torch.randn(2, 8)
        y = torch.tensor([0, 2])
        d_loss, g_loss = mod.losses(g, d, torch.randn(2, 16), y, z)
        (d_loss + g_loss).backward()
        self.assertEqual(g(z, y).shape, (2, 16))
        self.assertFalse(torch.allclose(g(z, y), g(z, torch.tensor([1, 1]))))

    def test_pix2pix_patch_and_l1(self):
        mod = load("pix2pix", "pix2pix.py")
        g, d = mod.UNetGenerator(width=8), mod.PatchDiscriminator(width=8)
        source, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
        d_loss, g_loss = mod.losses(g, d, source, target)
        (d_loss + g_loss).backward()
        self.assertEqual(g(source).shape, target.shape)
        self.assertGreater(d(source, target).shape[-1], 1)

    def test_cyclegan_cycle_gradients(self):
        mod = load("CycleGAN", "cyclegan.py")
        g_ab, g_ba = mod.ResnetGenerator(width=8), mod.ResnetGenerator(width=8)
        d_a, d_b = mod.PatchDiscriminator(width=8), mod.PatchDiscriminator(width=8)
        a, b = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
        d_loss, g_loss, cycle = mod.losses(g_ab, g_ba, d_a, d_b, a, b)
        (d_loss + g_loss).backward()
        self.assertGreaterEqual(cycle.item(), 0)
        self.assertIsNotNone(g_ab.network[0].weight.grad)

    def test_srgan_fourfold_and_content_gradient(self):
        mod = load("SRGAN", "srgan.py")
        g, d, feature = mod.SRGenerator(width=8), mod.SRDiscriminator(width=8), mod.FeatureExtractor()
        low, high = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 16, 16)
        d_loss, g_loss, content = mod.losses(g, d, feature, low, high)
        (d_loss + g_loss).backward()
        self.assertEqual(g(low).shape, high.shape)
        self.assertGreaterEqual(content.item(), 0)
        self.assertIsNotNone(g.first.weight.grad)

    def test_stylegan2_modulation_and_regularizers(self):
        mod = load("StyleGAN2", "stylegan2.py")
        layer = mod.ModulatedConv2d(4, 5, 3)
        with torch.no_grad():
            layer.affine.weight.normal_(0, 0.2)
        x, s = torch.randn(2, 4, 4, 4), torch.randn(2, 3)
        y = layer(x, s)
        self.assertEqual(y.shape, (2, 5, 4, 4))
        self.assertFalse(torch.allclose(y[0], layer(x[:1], s[1:2])[0]))
        g, d = mod.StyleGenerator(width=8), mod.StyleDiscriminator(width=8)
        latent, real = torch.randn(2, 16), torch.randn(2, 3, 8, 8)
        d_loss, g_loss = mod.adversarial_losses(g, d, real, latent)
        path_penalty, _ = mod.path_length_penalty(g, latent)
        (d_loss + g_loss + path_penalty).backward()
        self.assertEqual(g(latent).shape, real.shape)
        self.assertIsNotNone(g.mapping[0].weight.grad)

    def test_ddim_skip_and_determinism(self):
        mod = load("DDIM", "ddim.py")
        schedule = mod.DDIM(total_steps=20)
        clean, noise = torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
        t, s = 19, 6
        current = schedule.q_sample(clean, torch.full((2,), t), noise)
        jumped = schedule.step(current, noise, t, s, eta=0)
        expected = schedule.q_sample(clean, torch.full((2,), s), noise)
        self.assertTrue(torch.allclose(jumped, expected, atol=1e-5))
        self.assertTrue(torch.allclose(schedule.step(current, noise, t, -1, eta=0), clean, atol=1e-5))
        denoiser = mod.TinyDenoiser(total_steps=20)
        self.assertTrue(torch.equal(schedule.sample(denoiser, noise, 5), schedule.sample(denoiser, noise, 5)))
        with self.assertRaises(ValueError):
            schedule.step(current, noise, t, s, eta=2.0)


if __name__ == "__main__":
    unittest.main()
