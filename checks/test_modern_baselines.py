"""Mechanism checks for ten added generative, RL, SSL, and graph baselines."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder, name):
    path = ROOT / folder / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules to resolve annotations.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BaselineTests(unittest.TestCase):
    def test_score_sde(self):
        module = load("Score-SDE", "score_sde.py")
        sde = module.VPSDE()
        clean = torch.randn(3, 2)
        noisy, std = sde.marginal(clean, torch.zeros(3), torch.randn_like(clean))
        self.assertTrue(torch.allclose(noisy, clean))
        self.assertTrue(torch.allclose(std, torch.zeros_like(std)))
        network = module.ScoreNetwork()
        loss = module.score_matching_loss(network, sde, clean)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(module.reverse_sample(network, sde, clean, steps=3).shape), (3, 2))
        class ZeroScore(nn.Module):
            def forward(self, state, time):
                return torch.zeros_like(state)

        start = torch.tensor([[1., -1.]])
        torch.manual_seed(7)
        actual = module.reverse_sample(ZeroScore(), sde, start, steps=1, epsilon=.5)
        torch.manual_seed(7)
        expected = start + .5 * 0.5 * sde.beta(torch.ones(1))[:, None] * start
        expected = expected + sde.beta(torch.ones(1)).sqrt()[:, None] * (.5 ** .5) * torch.randn_like(start)
        self.assertTrue(torch.allclose(actual, expected))

    def test_classifier_free_guidance(self):
        module = load("Classifier-Free Guidance", "cfg.py")
        model = module.ConditionalNoisePredictor()
        labels = torch.tensor([0, 2, 1])
        dropped, mask = module.drop_conditions(labels, model.null_condition, 1.0)
        self.assertTrue(mask.all() and (dropped == model.null_condition).all())
        noisy, time = torch.randn(3, 4), torch.ones(3, dtype=torch.long)
        self.assertTrue(torch.allclose(module.guided_prediction(model, noisy, time, labels, 0),
                                       model(noisy, time, None)))
        self.assertTrue(torch.allclose(module.guided_prediction(model, noisy, time, labels, 1),
                                       model(noisy, time, labels), atol=1e-6))

    def test_controlnet_zero_injection(self):
        module = load("ControlNet", "controlnet.py")
        base = module.TinyDenoiser(hidden=8)
        model = module.ControlNet(base)
        noisy, time = torch.randn(2, 3, 8, 8), torch.rand(2)
        first = model(noisy, time, torch.randn(2, 1, 8, 8))
        second = model(noisy, time, torch.randn(2, 1, 8, 8))
        self.assertTrue(torch.allclose(first, base(noisy, time), atol=1e-6))
        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        first.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.locked.parameters()))
        self.assertIsNotNone(model.zero_residual.weight.grad)

    def test_reinforce_returns(self):
        module = load("REINFORCE", "reinforce.py")
        returns = module.discounted_returns(torch.tensor([1., 2., 3.]), gamma=.5)
        self.assertTrue(torch.allclose(returns, torch.tensor([2.75, 3.5, 3.])))
        policy = module.Policy(3, 2)
        log_prob = policy.act(torch.randn(3))[1]
        (-log_prob).backward()
        self.assertIsNotNone(policy.network[0].weight.grad)

    def test_double_dqn_selection_vs_evaluation(self):
        module = load("Double DQN", "double_dqn.py")
        class FixedQ(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer("values", torch.tensor(values, dtype=torch.float))

            def forward(self, states):
                return self.values.expand(states.shape[0], -1)

        online, target = FixedQ([2., 1.]), FixedQ([0., 10.])
        desired = module.double_dqn_target(online, target, torch.tensor([1.]),
                                           torch.zeros(1, 2), torch.tensor([0.]), gamma=.5)
        self.assertTrue(torch.allclose(desired, torch.tensor([1.])))

    def test_td3_delayed_actor_and_clipping(self):
        from copy import deepcopy
        module = load("TD3", "td3.py")
        actor, critic = module.Actor(3, 1, hidden=8), module.TwinCritic(3, 1, hidden=8)
        target_actor, target_critic = deepcopy(actor), deepcopy(critic)
        aopt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        qopt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        batch = (torch.randn(4, 3), torch.randn(4, 1).tanh(),
                 torch.randn(4, 1), torch.randn(4, 3), torch.zeros(4, 1))
        self.assertIsNone(module.update(actor, critic, target_actor, target_critic,
                                        aopt, qopt, batch, step=1)["actor"])
        result = module.update(actor, critic, target_actor, target_critic,
                               aopt, qopt, batch, step=2)
        self.assertIsNotNone(result["actor"])
        smooth = module.smoothed_target_action(target_actor, batch[3], noise_std=100)
        self.assertTrue((smooth.abs() <= 1).all())

    def test_alphazero_search_and_outcomes(self):
        module = load("AlphaZero", "alphazero.py")
        game = module.TicTacToe()
        game = game.play(0).play(3).play(1).play(4).play(2)
        self.assertEqual(game.winner(), 1)
        self.assertEqual(game.outcome(), -1.0)  # side to move is losing
        network = module.PolicyValueNet(16)
        start = module.TicTacToe().play(0)
        probabilities, root = module.MCTS(network).search(start, simulations=6)
        self.assertEqual(float(probabilities[0]), 0.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertEqual(root.visits, 6)
        states, policies, outcomes = module.self_play(network, simulations=2)
        self.assertEqual(tuple(states.shape[1:]), (9,))
        self.assertTrue(torch.allclose(policies.sum(-1), torch.ones(len(policies))))
        self.assertTrue(((outcomes >= -1) & (outcomes <= 1)).all())

    def test_barlow_vicreg(self):
        barlow = load("Barlow Twins", "barlow_twins.py")
        vicreg = load("VICReg", "vicreg.py")
        varied = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        correlation = barlow.cross_correlation(varied, varied)
        self.assertTrue(torch.allclose(correlation, torch.eye(2), atol=1e-4))
        collapsed = torch.zeros_like(varied)
        self.assertGreater(barlow.barlow_loss(collapsed, collapsed),
                           barlow.barlow_loss(varied, varied))
        self.assertGreater(vicreg.variance_penalty(collapsed), vicreg.variance_penalty(varied))
        self.assertTrue(torch.isfinite(vicreg.vicreg_loss(varied, varied)))

    def test_graphormer_distances(self):
        module = load("Graphormer", "graphormer.py")
        adjacency = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
        edge_types = torch.tensor([[[0, 1, 0], [1, 0, 2], [0, 2, 0]]])
        distances, paths = module.shortest_path_features(adjacency, edge_types)
        self.assertEqual(int(distances[0, 0, 2]), 2)
        self.assertEqual(paths[0, 0, 2, :2].tolist(), [1, 2])
        model = module.Graphormer(4, 2)
        features = torch.randn(1, 3, 4)
        logits = model(features, adjacency, edge_types)
        self.assertEqual(tuple(logits.shape), (1, 2))
        logits.sum().backward()
        permutation = [2, 1, 0]
        reordered = model(features[:, permutation], adjacency[:, permutation][:, :, permutation],
                          edge_types[:, permutation][:, :, permutation])
        self.assertTrue(torch.allclose(logits, reordered, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
