"""Small AlphaZero mechanics on tic-tac-toe: policy/value net, PUCT MCTS, self-play."""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TicTacToe:
    board: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3, dtype=torch.int8))
    player: int = 1  # absolute side to move: +1 or -1

    def canonical(self):
        return self.board.float().flatten() * self.player

    def legal_moves(self):
        return torch.where(self.board.flatten() == 0)[0].tolist()

    def winner(self):
        lines = [*self.board, *self.board.T, self.board.diagonal(),
                 self.board.flip(1).diagonal()]
        for line in lines:
            total = int(line.sum())
            if abs(total) == 3:
                return 1 if total > 0 else -1
        return 0

    def outcome(self):
        """Terminal payoff from side-to-move's perspective, or None if unfinished."""
        winner = self.winner()
        if winner:
            return float(winner * self.player)
        if not self.legal_moves():
            return 0.0
        return None

    def play(self, move: int):
        if self.outcome() is not None or move not in self.legal_moves():
            raise ValueError("Illegal move or finished game")
        next_board = self.board.clone()
        next_board.flatten()[move] = self.player
        return TicTacToe(next_board, -self.player)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, 9)
        self.value_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, canonical_boards: torch.Tensor):
        hidden = self.trunk(canonical_boards)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class Node:
    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0  # value from player-to-move at this node
        self.children: dict[int, Node] = {}

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, network: PolicyValueNet, exploration: float = 1.5):
        self.network = network
        self.exploration = exploration

    @torch.no_grad()
    def expand(self, node: Node, game: TicTacToe):
        logits, value = self.network(game.canonical().unsqueeze(0))
        legal = game.legal_moves()
        masked_logits = logits[0, legal]
        probabilities = masked_logits.softmax(dim=-1)
        node.children = {move: Node(float(probability))
                         for move, probability in zip(legal, probabilities)}
        return float(value[0])

    def select(self, node: Node):
        root = math.sqrt(node.visits + 1)
        return max(node.children.items(), key=lambda pair: (
            -pair[1].mean_value if pair[1].visits else 0.0
        ) + self.exploration * pair[1].prior * root / (1 + pair[1].visits))

    def search(self, position: TicTacToe, simulations: int = 16):
        if position.outcome() is not None or simulations < 1:
            raise ValueError("Search needs a live game and at least one simulation")
        root = Node()
        self.expand(root, position)
        for _ in range(simulations):
            game = position
            node = root
            visited = [node]
            while node.children:
                move, node = self.select(node)
                game = game.play(move)
                visited.append(node)
            payoff = game.outcome()
            value = self.expand(node, game) if payoff is None else payoff
            for item in reversed(visited):
                item.visits += 1
                item.value_sum += value
                value = -value  # players alternate along each edge
        visits = torch.zeros(9)
        for move, child in root.children.items():
            visits[move] = child.visits
        return visits / visits.sum(), root


def self_play(network: PolicyValueNet, simulations: int = 8):
    """One small game; targets are search visit probabilities and final results."""
    game = TicTacToe()
    records = []
    while game.outcome() is None:
        probabilities, _ = MCTS(network).search(game, simulations)
        records.append((game.canonical(), probabilities, game.player))
        move = int(torch.multinomial(probabilities, 1))
        game = game.play(move)
    winner = game.winner()
    states = torch.stack([record[0] for record in records])
    policies = torch.stack([record[1] for record in records])
    outcomes = torch.tensor([winner * record[2] for record in records], dtype=torch.float32)
    return states, policies, outcomes


def training_loss(network: PolicyValueNet, states: torch.Tensor,
                  target_policies: torch.Tensor, outcomes: torch.Tensor):
    logits, values = network(states)
    policy_loss = -(target_policies * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, outcomes)
    return policy_loss + value_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    network = PolicyValueNet(32)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    states, policies, outcomes = self_play(network, simulations=4)
    loss = training_loss(network, states, policies, outcomes)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("AlphaZero self-play positions/loss:", len(states), round(loss.item(), 4))
