"""SGD、动量法和 Adam 的完整参数更新；不调用框架优化器。"""

import numpy as np


class SGD:
    def __init__(self, learning_rate: float, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]):
        for name, parameter in parameters.items():
            v = self.velocity.setdefault(name, np.zeros_like(parameter))
            v *= self.momentum
            v += gradients[name]
            parameter -= self.learning_rate * v


class Adam:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.epsilon = epsilon
        self.first_moment: dict[str, np.ndarray] = {}
        self.second_moment: dict[str, np.ndarray] = {}
        self.step_number = 0

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]):
        self.step_number += 1
        for name, parameter in parameters.items():
            gradient = gradients[name]
            m = self.first_moment.setdefault(name, np.zeros_like(parameter))
            v = self.second_moment.setdefault(name, np.zeros_like(parameter))
            m *= self.beta1
            m += (1 - self.beta1) * gradient
            v *= self.beta2
            v += (1 - self.beta2) * gradient**2
            m_hat = m / (1 - self.beta1**self.step_number)  # 消除初始零值偏差
            v_hat = v / (1 - self.beta2**self.step_number)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


if __name__ == "__main__":
    for optimizer in (SGD(0.1, momentum=0.9), Adam(0.1)):
        parameters = {"x": np.array([5.0])}
        for _ in range(100):
            optimizer.step(parameters, {"x": 2 * (parameters["x"] - 2)})
        print(type(optimizer).__name__, parameters["x"].item())
