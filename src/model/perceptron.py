import numpy as np


class Perceptron:
    def __init__(self, n_features: int, bias: bool):
        self.n_features = n_features
        self.weight = np.random.normal(size=(self.n_features,))

        self.bias = np.random.normal(size=(self.n_features,)) if bias else None

    def forward(self):
        pass

    def update(self):
        pass


if __name__ == "__main__":
    n_features = 4
    p = Perceptron(n_features)
    print(p.weight)
    print(p.bias)
