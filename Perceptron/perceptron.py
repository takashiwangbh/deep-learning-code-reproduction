"""Rosenblatt binary perceptron: every update is visible here."""

import numpy as np


class Perceptron:
    def __init__(self, n_features: int, learning_rate: float = 1.0):
        self.weight = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.weight + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(x) >= 0, 1, -1)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 20) -> list[int]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2 or x.shape[1] != len(self.weight) or x.shape[0] != len(y):
            raise ValueError("x must be [N,D] and y must be [N]")
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("labels must be -1 or +1")
        errors = []
        for _ in range(epochs):
            mistakes = 0
            for sample, label in zip(x, y):
                if label * (sample @ self.weight + self.bias) <= 0:
                    self.weight += self.learning_rate * label * sample
                    self.bias += self.learning_rate * label
                    mistakes += 1
            errors.append(mistakes)
            if mistakes == 0:
                break
        return errors


if __name__ == "__main__":
    points = np.array([[2, 1], [1, 2], [-1, -2], [-2, -1]], dtype=float)
    labels = np.array([1, 1, -1, -1])
    model = Perceptron(2)
    history = model.fit(points, labels)
    print("mistakes by epoch:", history, "predictions:", model.predict(points))
