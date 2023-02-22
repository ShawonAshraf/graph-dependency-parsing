import numpy as np
from typing import Dict, List
from tqdm.auto import tqdm, trange
from .features import ProcessedInstance


class Perceptron:
    def __init__(self, feature_dict: Dict, normalise=False, train=True, pretrained_weights=None):
        self.feature_dict = feature_dict
        self.weights = {}
        self.normalise = normalise

        # only init weights for training, otherwise load pretrained weights
        if train:
            self.init_weights()
        else:
            self.weights = pretrained_weights

    def init_weights(self):
        for feature in self.feature_dict.keys():
            if feature not in self.weights.keys():
                # assign 0 as a starter value
                self.weights[feature] = 0.0 if not self.normalise else np.random.normal()

    # for one sentence
    # creates and returns the score matrix
    # based on weights
    def forward(self, sentence_features):
        # the adjacency matrix for the graph should be
        # a square matrix
        # all tokens against each other
        score_matrix = np.array(shape=(
            len(sentence_features),
            len(sentence_features)
        ))

        for i, token_features in enumerate(sentence_features):
            for j, tokf in enumerate(token_features):
                if tokf in self.weights.keys():
                    score_matrix[i][j] += self.weights[tokf]

        return score_matrix

    def train(self, epochs):
        for e in trange(epochs):
            pass

    def predict(self):
        pass

    def save(self):
        pass


if __name__ == "__main__":
    p = Perceptron({}, normalise=True)
    print(p.weights)
