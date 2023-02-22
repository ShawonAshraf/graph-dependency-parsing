import numpy as np
from typing import Dict, List
from tqdm.auto import trange
from .features import ProcessedInstance
import gzip
import pickle


# for scoring arcs
class Perceptron:
    def __init__(self, feature_dict: Dict, normalise=False, train=True):
        self.feature_dict = feature_dict
        self.weights = {}
        self.normalise = normalise

        # only init weights for training,
        # otherwise load pretrained weights ( to be called by the user )
        if train:
            self.init_weights()

    def init_weights(self):
        for feature in self.feature_dict.keys():
            if feature not in self.weights.keys():
                # assign 0 as a starter value
                self.weights[feature] = 0.0 if not self.normalise else np.random.normal()

    # for one sentence
    # creates and returns the score matrix
    # based on weights
    def forward(self, sentence_features) -> np.ndarray:
        # the adjacency matrix for the graph should be
        # a square matrix
        # rows are heads
        # cols are dependents
        # idx 0,0 is for root
        score_matrix = np.ones(shape=(
            len(sentence_features) + 1,
            len(sentence_features) + 1
        ))
        score_matrix *= -np.Inf  # default value

        # length of the sentences features
        s = len(sentence_features)

        # again, rows are heads
        for i in range(s):
            # features for the current token in the sentence
            current_token_features = sentence_features[i]

            # cols are dependents
            for j in range(i + 1, s):
                total = 0.0
                for tokf in current_token_features:
                    if tokf in self.weights.keys():
                        total += self.weights[tokf]

                # add the total to the score matrix
                score_matrix[i][j] = total

        return score_matrix

    def train(self, epochs, processed_instances: List[ProcessedInstance]):
        for e in trange(epochs):
            for _, pi in enumerate(processed_instances):
                sentence_features = pi.features
                labels = pi.labels

                # actual heads
                targets = [l[1] for l in labels]

                score_matrix = self.forward(sentence_features)

                # square matrix so shape index doesn't matter
                for idx, target in enumerate(targets):
                    predicted_head_idx = np.argmax(score_matrix[:, idx])
                    target_idx = target - 1

                    # update weights
                    factor = 1.0 if predicted_head_idx == target_idx else -1.0
                    self.update(factor, sentence_features)

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
