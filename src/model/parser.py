import numpy as np

from .perceptron import Perceptron
from .features import ProcessedInstance
from typing import List, Dict, Callable


class Parser:
    def __init__(self, processed_instances: List[ProcessedInstance],
                 perceptron: Perceptron,
                 decoder_fn: Callable) -> None:
        self.features = processed_instances.features
        self.labels = processed_instances.labels

        self.perceptron = perceptron

        self.decoder_fn = decoder_fn


    # a forward pass for the parser
    def parse(self):
        pass

    # train only the perceptron
    # the decoder isn't trainable
    # since it gets the information from the
    # perceptron
    def train(self):
        pass
    
