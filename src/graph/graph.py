import numpy as np
from data.sentence import Sentence


class Graph:
    def __init__(self, sentence: Sentence) -> None:
        self.tokens = sentence.tokens

        self.matrix = None
        self.__construct_adjacency_matrix()

    def __construct_adjacency_matrix(self):
        n_tokens = len(self.tokens)

        # populate a matrix of 1's
        # multiply with np.inf
        # shape = n_tokens + 1 (for ROOT as an extra index at 0,0)
        self.matrix = np.ones(shape=(n_tokens + 1, n_tokens + 1)) * np.Inf

        for idx, token in enumerate(self.tokens):
            head_idx = token.head - 1
            self.matrix[idx][head_idx] = 1.0
