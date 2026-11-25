"""仅用 NumPy 实现两层 MLP 和反向传播：python mlp_numpy.py。"""

import numpy as np


class TwoLayerMLP:
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(1 / input_dim), (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": rng.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, num_classes)),
            "b2": np.zeros(num_classes),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        p = self.parameters
        z1 = x @ p["w1"] + p["b1"]       # [N, H]
        a1 = np.tanh(z1)                    # 非线性；没有它，两层线性层仍是一层线性层
        logits = a1 @ p["w2"] + p["b2"]  # [N, C]
        return logits, (x, a1)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        """对平均交叉熵逐层应用链式法则。y 是 [N] 的类别编号。"""
        logits, (x, a1) = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)  # 防止 exp 溢出
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        n = len(x)
        loss = -np.log(probabilities[np.arange(n), y] + 1e-12).mean()

        # softmax + cross entropy 的合成导数是 p - one_hot(y)。
        dlogits = probabilities.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n
        grads = {
            "w2": a1.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        da1 = dlogits @ self.parameters["w2"].T
        dz1 = da1 * (1 - a1**2)            # tanh'(z) = 1 - tanh(z)^2
        grads["w1"] = x.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return float(loss), grads

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].argmax(axis=1)


def train_xor(steps: int = 2500, learning_rate: float = 0.2) -> tuple[TwoLayerMLP, float]:
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 0])
    model = TwoLayerMLP(2, 8, 2)
    for _ in range(steps):
        loss, grads = model.loss_and_gradients(x, y)
        for name, value in model.parameters.items():
            value -= learning_rate * grads[name]
    return model, loss


if __name__ == "__main__":
    model, loss = train_xor()
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    print(f"XOR loss={loss:.4f}, predictions={model.predict(x).tolist()}")
