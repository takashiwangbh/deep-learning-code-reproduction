"""不下载数据的最小正确性检查：形状、损失、梯度和关键数学性质。"""

import unittest

import torch
from torch.nn import functional as F

from autoencoder import Autoencoder, reconstruction_loss
from clip import CLIP, contrastive_loss
from dcgan import Discriminator, Generator, discriminator_loss, generator_loss
from ddpm import DDPM, TinyUNet
from dit import DiT
from flow_matching import FlowMatching
from gcn import GCN, normalize_adjacency
from latent_diffusion import LatentDiffusion
from vae import VAE, elbo_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_autoencoder(self):
        model = Autoencoder(input_dim=16, hidden_dim=12, latent_dim=3)
        x = torch.randn(2, 1, 4, 4)
        reconstructed, z = model(x)
        self.assertEqual(reconstructed.shape, (2, 16))
        self.assertEqual(z.shape, (2, 3))
        reconstruction_loss(reconstructed, x).backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_vae_reparameterization_and_kl(self):
        model = VAE(input_dim=16, hidden_dim=12, latent_dim=3)
        x = torch.rand(2, 1, 4, 4)
        logits, mu, logvar = model(x)
        total, reconstruction, kl = elbo_loss(logits, x, mu, logvar)
        self.assertEqual(logits.shape, (2, 16))
        self.assertGreaterEqual(kl.item(), 0)
        self.assertAlmostEqual(total.item(), reconstruction.item() + kl.item(), places=5)
        total.backward()
        self.assertIsNotNone(model.to_mu.weight.grad)
        self.assertIsNotNone(model.to_logvar.weight.grad)
        self.assertEqual(model.sample(2).shape, (2, 16))
        _, _, zero_kl = elbo_loss(logits, x, torch.zeros_like(mu), torch.zeros_like(logvar))
        self.assertAlmostEqual(zero_kl.item(), 0)

    def test_dcgan(self):
        generator = Generator(latent_dim=8, base_channels=4)
        discriminator = Discriminator(base_channels=4)
        fake = generator(torch.randn(2, 8))
        self.assertEqual(fake.shape, (2, 3, 64, 64))
        self.assertLessEqual(fake.abs().max().item(), 1)
        real_logits = discriminator(torch.randn_like(fake))
        fake_logits = discriminator(fake.detach())
        discriminator_loss(real_logits, fake_logits).backward()
        self.assertIsNotNone(discriminator.net[0].weight.grad)
        generator_loss(discriminator(fake)).backward()
        self.assertIsNotNone(generator.net[0].weight.grad)

    def test_ddpm_equation_and_reverse(self):
        model = DDPM(TinyUNet(image_channels=1, base_channels=8), timesteps=4)
        x0, noise = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        t = torch.tensor([0, 3])
        expected = model.alpha_bars[t].sqrt().reshape(-1, 1, 1, 1) * x0
        expected += (1 - model.alpha_bars[t]).sqrt().reshape(-1, 1, 1, 1) * noise
        self.assertTrue(torch.allclose(model.q_sample(x0, t, noise), expected))
        loss = model.training_loss(x0)
        loss.backward()
        self.assertIsNotNone(model.denoiser.stem.weight.grad)
        output = model.p_sample(x0, torch.zeros(2, dtype=torch.long))
        self.assertEqual(output.shape, x0.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.sample((2, 1, 8, 8)).shape, (2, 1, 8, 8))

    def test_latent_ddpm_does_not_clip_to_pixel_range(self):
        class ZeroDenoiser(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        diffusion = DDPM(ZeroDenoiser(), timesteps=4, clip_denoised=False)
        xt = torch.full((2, 1, 2, 2), 3.0)
        x0 = diffusion.p_sample(xt, torch.zeros(2, dtype=torch.long))
        self.assertGreater(x0.min().item(), 2.0)

    def test_dit_patchify_and_gradient(self):
        model = DiT(image_size=8, patch_size=2, in_channels=2, hidden_dim=32, depth=2, heads=4)
        x = torch.randn(2, 2, 8, 8)
        predicted = model(x, torch.tensor([1, 2]))
        self.assertEqual(predicted.shape, x.shape)
        self.assertTrue(torch.equal(predicted, torch.zeros_like(predicted)))  # adaLN-Zero
        F.mse_loss(predicted, torch.randn_like(predicted)).backward()
        self.assertGreater(model.final.linear.weight.grad.abs().sum().item(), 0)

    def test_flow_matching_path_and_euler(self):
        noise = torch.zeros(2, 1, 2, 2)
        data = torch.ones_like(noise)
        halfway, velocity = FlowMatching.interpolation(noise, data, torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.equal(halfway, torch.full_like(data, 0.5)))
        self.assertTrue(torch.equal(velocity, data))

        class UnitVelocity(torch.nn.Module):
            def forward(self, x, t):
                return torch.ones_like(x)

        flow = FlowMatching(UnitVelocity())
        result = flow.sample(tuple(noise.shape), steps=4, initial_noise=noise)
        self.assertTrue(torch.allclose(result, data))

    def test_latent_diffusion_two_stages_and_text_attention(self):
        model = LatentDiffusion(
            image_size=16, image_channels=1, latent_channels=2,
            base_channels=8, vocab_size=16, context_dim=16,
            max_text_length=4, heads=2, timesteps=4,
        )
        images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
        tokens = torch.tensor([[1, 2, 0], [3, 4, 5]])
        total, pixel, kl = model.autoencoder_loss(images)
        self.assertGreaterEqual(kl.item(), 0)
        self.assertAlmostEqual(total.item(), pixel.item() + 1e-4 * kl.item(), places=5)
        total.backward()
        self.assertIsNotNone(model.autoencoder.encoder[0].weight.grad)
        self.assertIsNotNone(model.autoencoder.decoder[0].weight.grad)

        model.zero_grad(set_to_none=True)
        model.freeze_autoencoder()
        model.training_loss(images, tokens).backward()
        self.assertIsNone(model.autoencoder.encoder[0].weight.grad)
        self.assertGreater(model.conditioner.token.weight.grad.abs().sum().item(), 0)
        self.assertGreater(
            model.diffusion.denoiser.cross_attention.to_k.weight.grad.abs().sum().item(), 0
        )

        latent = torch.randn(1, 2, 4, 4)
        t = torch.tensor([2])
        visible = torch.tensor([[True, True, False]])
        with torch.no_grad():
            context_a, mask_a = model.conditioner(torch.tensor([[1, 2, 0]]), visible)
            context_b, _ = model.conditioner(torch.tensor([[1, 2, 7]]), visible)
            context_c, _ = model.conditioner(torch.tensor([[1, 8, 0]]), visible)
            denoiser = model.diffusion.denoiser
            result_a = denoiser(latent, t, context_a, mask_a)
            result_b = denoiser(latent, t, context_b, mask_a)
            result_c = denoiser(latent, t, context_c, mask_a)
        self.assertTrue(torch.allclose(result_a, result_b, atol=1e-6))
        self.assertFalse(torch.allclose(result_a, result_c, atol=1e-6))

        generated = model.sample(tokens)
        self.assertEqual(generated.shape, images.shape)
        self.assertTrue(torch.isfinite(generated).all())
        self.assertLessEqual(generated.abs().max().item(), 1)

    def test_clip_pairing_and_text_causality(self):
        model = CLIP(image_size=16, patch_size=8, vocab_size=20, context_length=6,
                     width=16, depth=1, heads=2, projection_dim=8)
        image = torch.randn(2, 3, 16, 16)
        tokens = torch.randint(0, 20, (2, 6))
        lengths = torch.tensor([3, 5])
        logits = model(image, tokens, lengths)
        self.assertEqual(logits.shape, (2, 2))
        contrastive_loss(logits).backward()
        self.assertIsNotNone(model.image.patch.weight.grad)
        self.assertIsNotNone(model.text.token.weight.grad)
        changed = tokens.clone()
        changed[0, 3:] = (changed[0, 3:] + 1) % 20
        with torch.no_grad():
            first = model.text(tokens, lengths)[0]
            second = model.text(changed, lengths)[0]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_gcn_self_loops_and_gradient(self):
        adjacency = torch.tensor([[0., 1, 0], [1, 0, 0], [0, 0, 0]])
        normalized = normalize_adjacency(adjacency)
        self.assertAlmostEqual(normalized[2, 2].item(), 1)
        self.assertAlmostEqual(normalized[2, :2].abs().sum().item(), 0)
        model = GCN(4, 6, 2, dropout=0)
        logits = model(torch.randn(3, 4), adjacency)
        self.assertEqual(logits.shape, (3, 2))
        F.cross_entropy(logits[:2], torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.conv1.linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
