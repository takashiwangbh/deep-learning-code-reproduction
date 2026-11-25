"""运行：python smoke_test.py。"""

import torch

from dqn import AtariQNetwork, DQNAgent, ReplayBuffer, SmallQNetwork
from ppo import ActorCritic, clipped_policy_loss, generalized_advantage_estimate, ppo_update


def main():
    torch.manual_seed(3)
    q = AtariQNetwork(6)
    assert q(torch.randn(2, 4, 84, 84)).shape == (2, 6)

    agent = DQNAgent(SmallQNetwork(4, 2), SmallQNetwork(4, 2))
    replay = ReplayBuffer(16)
    for i in range(12):
        replay.add(torch.randn(4), i % 2, 1.0, torch.randn(4), i % 3 == 0)
    before = agent.online.net[0].weight.detach().clone()
    loss = agent.update(replay.sample(8))
    assert loss >= 0 and not torch.equal(before, agent.online.net[0].weight)
    assert all(p.grad is None for p in agent.target.parameters())
    assert len(replay) == 12
    graph_state = torch.randn(4, requires_grad=True)
    replay.add(graph_state, 0, 0.0, graph_state * 2, False)
    assert not replay.memory[-1].state.requires_grad
    assert not replay.memory[-1].next_state.requires_grad

    rewards = torch.tensor([1.0, 2.0])
    values = torch.zeros(2)
    dones = torch.tensor([0.0, 1.0])
    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, torch.tensor(100.0), gamma=1.0, gae_lambda=1.0
    )
    torch.testing.assert_close(advantage, torch.tensor([3.0, 2.0]))  # 终止后不 bootstrap
    torch.testing.assert_close(returns, advantage)
    clipped = clipped_policy_loss(torch.log(torch.tensor([2.0])),
                                  torch.zeros(1), torch.ones(1))
    torch.testing.assert_close(clipped, torch.tensor(-1.2))

    model = ActorCritic(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randn(8, 4)
    with torch.no_grad():
        distribution, values = model(observations)
        actions = distribution.sample()
        old_logp = distribution.log_prob(actions)
    before = model.policy_head.weight.detach().clone()
    ppo_update(model, optimizer, observations, actions, old_logp,
               torch.arange(8).float(), values + 1)
    assert not torch.equal(before, model.policy_head.weight)
    print("reinforcement: DQN replay/target update and PPO GAE/clipping checks passed")


if __name__ == "__main__":
    main()
