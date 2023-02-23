import numpy as np
from tqdm.auto import tqdm, trange

from .perceptron import Perceptron
from .features import extract_feature_permutation
from .eval import uas
from typing import List, Dict, Callable
from data.sentence import Sentence
from graph.graph import construct_graph
import gzip
import pickle


class Parser:
    def __init__(self,
                 perceptron: Perceptron,
                 decoder_fn: Callable) -> None:
        self.perceptron = perceptron

        self.decoder_fn = decoder_fn

        # for logging
        self.uas_train_scores_over_epochs = list()
        self.uas_dev_scores_over_epochs = list()

    # a forward pass for the parser
    # parse tokens from a single sentence
    def parse(self, sentence: Sentence):
        n_tokens = len(sentence.tokens)
        score_matrix = np.zeros(
            (n_tokens, n_tokens)
        )

        features = dict()
        # update the score matrix
        for i, tok in enumerate(sentence.tokens):
            features[i] = dict()
            for j, head in enumerate(sentence.tokens):
                feat = extract_feature_permutation(head, tok, sentence.tokens)
                score = self.perceptron.score(feat)
                score_matrix[i][j] = score
                features[i][j] = feat

        heads = self.decoder_fn(score_matrix)

        return heads, features

    # train only the perceptron
    # the decoder isn't trainable
    # since it gets the information from the
    # perceptron
    def train(self,
              epochs: int,
              train_set: List[Sentence],
              dev_set: List[Sentence]):
        print("\n================ Training Parser ===============\n")
        for e in trange(epochs):
            self.perceptron.train(epochs=1, sentences=train_set)

            # eval on train
            avg_train_uas, _ = self.eval(train_set)
            self.uas_train_scores_over_epochs.append(avg_train_uas)

            # eval on dev
            avg_dev_uas, _ = self.eval(dev_set)
            self.uas_train_scores_over_epochs.append(avg_dev_uas)

            # log
            print(f"Epoch :: {e + 1}/{epochs} train_uas :: {avg_train_uas} dev_uas :: {avg_dev_uas}")

    def eval(self, sentences: List[Sentence]):
        scores = list()

        for sentence in sentences:
            preds, _ = self.parse(sentence)
            gold = [tok.head for tok in sentence.tokens]

            uas_score = np.count_nonzero(preds == gold)

            scores.append(uas_score)

        return np.mean(scores), preds

    def save_scores(self) -> None:
        with gzip.open("scores.pickle", "wb") as fp:
            pickle.dump(self.uas_train_scores_over_epochs)
            pickle.dump(self.uas_train_scores_over_epochs)

    def generate_tree(self, sentences: List[Sentence]) -> List[Sentence]:
        tree_sents = sentences.copy()
        for idx, sentence in tqdm(enumerate(tree_sents), desc="generating_trees"):
            head, _ = self.parse(sentence)
            tokens = sentence.tokens

            for h, t in zip(head, tokens):
                t.head = h

        return tree_sents
