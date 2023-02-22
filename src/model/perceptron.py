import numpy as np
from typing import Dict, List
from tqdm.auto import trange
from .features import ProcessedInstance
import gzip
import pickle


# for scoring arcs
class Perceptron:
    def __init__(self, feature_dict: Dict, train=True):
        self.feature_dict = feature_dict
        self.weights = {}

        # only init weights for training,
        # otherwise load pretrained weights ( to be called by the user )
        if train:
            self.init_weights()

    def init_weights(self):
        for feature in self.feature_dict.keys():
            if feature not in self.weights.keys():
                # assign 0 as a starter value
                self.weights[feature] = 0.0

    # sorts the weight dict
    def sort(self):
        self.weights = dict(
            sorted(
                self.weights.items(), reverse=True, key=lambda x: x[1]
            )
        )

    # feature list is the list containing all features for a token
    def score(self, feature_list: List[str]):
        s = 0.0
        for f in feature_list:
            if f in self.weights.keys():
                s += self.weights[f]
        return s



    def train(self, epochs, processed_instances: List[ProcessedInstance]):
        pass

    # update the corresponding token feature weights
    def update(self, factor: float, sentence_features: List[List[str]]) -> None:
        for token_features in sentence_features:
            for tokf in token_features:
                if tokf in self.weights.keys():
                    self.weights[tokf] += factor

    # save the weight dict to the disc
    # out_path must also include file name
    def save(self, out_path: str) -> None:
        try:
            with gzip.open(out_path, "wb") as fp:
                pickle.dump(self.weights, fp)
        except FileExistsError as e:
            print(e)
        except FileNotFoundError as e:
            print(e)

    # load saved weights
    def load(self, file_path) -> None:
        try:
            with gzip.open(file_path, "rb") as fp:
                self.weights = pickle.load(fp)
        except FileExistsError as e:
            print(e)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    p = Perceptron({}, normalise=True)
    print(p.weights)
