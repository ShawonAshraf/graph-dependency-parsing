import numpy as np
from tqdm.auto import tqdm, trange

from .perceptron import Perceptron
from .features import ProcessedInstance
from .eval import uas
from typing import List, Dict, Callable


class Parser:
    def __init__(self,
                 perceptron: Perceptron,
                 decoder_fn: Callable) -> None:
        self.perceptron = perceptron

        self.decoder_fn = decoder_fn

    # a forward pass for the parser
    # parse tokens from a single sentence
    def parse(self, sentence_features: List[List[str]]):
        score_matrix = self.perceptron.forward(sentence_features)
        print(score_matrix)
        # construct a graph and pass to decoder
        graph = self.decoder_fn(score_matrix)

        # get heads from the graph
        n_tokens = len(sentence_features)
        heads = np.ones(shape=(n_tokens,))
        for node in graph.nodes:
            _id = node.node_id
            incoming = list(node.incoming.keys())

            # ignore root
            if _id == 0:
                continue

            # take the first key
            heads[_id - 1] = incoming[0]

        assert heads.shape[0] == n_tokens
        return heads

    # train only the perceptron
    # the decoder isn't trainable
    # since it gets the information from the
    # perceptron
    def train(self,
              epochs: int,
              train_instances: List[ProcessedInstance],
              dev_instances: List[ProcessedInstance]):
        print("\n================ Training Parser ===============\n")
        self.perceptron.train(epochs, train_instances)
        # TODO: log uas
        # shuffle train_instances
        np.random.shuffle(train_instances)
