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
    def parse(self, token_features: List[str]):
        score_matrix = self.perceptron.forward(token_features)
        # construct a graph and pass to decoder
        graph = self.decoder_fn(score_matrix)

        # get heads from the graph
        heads = np.ones(shape=(len(token_features, )))
        for node in graph.nodes:
            _id = node.id
            incoming = node.incoming.keys()
            # take the first key
            heads[_id - 1] = incoming[0]

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
        for e in range(epochs):
            print(f"epoch : {e + 1}")

            # train pass
            self.perceptron.train(epochs=epochs, processed_instances=train_instances)
            # eval
            train_preds = 0
