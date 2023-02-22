import numpy as np
from typing import Dict, List
from tqdm.auto import trange, tqdm
from .features import extract_feature_permutation
import gzip
import pickle
from data.sentence import Sentence
from data.conll06_token import Conll06Token


# for scoring arcs
class Perceptron:
    def __init__(self, feature_dict: Dict, train=True):
        self.feature_dict = feature_dict
        self.weights = {}

        # only init weights for training,
        # otherwise load pretrained weights ( to be called by the user )
        if train:
            self.init_weights()

    def make_gold_features(self, sentence: Sentence):
        features = set()
        for tokens in sentence.tokens:
            for tok in tokens:
                head = tok.head
                features.update(extract_feature_permutation(head, tok, tokens))

            self.init_weights(features)

    # creates all possible feature permutations be consideing tokens as heads to each other
    def feature_permutations(self, sentence: Sentence):
        features = set()
        for tokens in sentence.tokens:
            for head in tokens:
                for token in tokens:
                    features.update(extract_feature_permutation(head, token, tokens))

        self.init_weights(features)

    def init_weights(self, feature_set):
        for feature in feature_set:
            if feature not in self.weights.keys():
                # assign 0 as a starter value
                self.weights[feature] = 0.0

    # feature list is the list containing all features for a token
    def score(self, feature_list: List[str]):
        s = 0.0
        for f in feature_list:
            if f in self.weights.keys():
                s += self.weights[f]
        return s

    def train(self, epochs: int, sentences: List[Sentence]):
        for e in trange(epochs):
            for idx, sentence in tqdm(enumerate(sentences)):
                tokens = sentence.tokens

                # create all possible feature permutations
                for token in tokens:
                    pred, features = self.predict(token, tokens)
                    # update weights for the actual head
                    self.update(features[pred], features[token.head])

    def predict(self, token: Conll06Token, tokens: List[Conll06Token]):
        scores = list()
        features = list()

        # score all probable heads against other tokens
        for probable_head in tokens:
            feats = extract_feature_permutation(probable_head, token, tokens)
            score = self.score(feats)
            features.append(feats)
            scores.append(score)

        pred = np.argmax(scores)
        return pred, features

    def update(self, pred_features, actual_features) -> None:

        def update_weight(feature, factor):
            if feature in self.weights.keys():
                self.weights[f] += factor

        for f in actual_features:
            update_weight(f, 1.0)
        for f in pred_features:
            update_weight(f, -1.0)

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
