from typing import List
from data.sentence import Sentence
import numpy as np


# for one sentence
def count_correct_heads(gold: Sentence, pred: Sentence):
    gold_tokens = gold.tokens
    pred_tokens = pred.tokens

    # get heads of all tokens
    gold_heads = np.array([tok.head for tok in gold_tokens])
    pred_heads = np.array([tok.head for tok in pred_tokens])

    n_matches = np.count_nonzero(gold_heads == pred_heads)

    return n_matches


def uas(gold: List[Sentence], pred: List[Sentence]):
    # count all the tokens
    n_tokens_gold = sum([len(s.tokens) for s in gold])
    n_tokens_pred = sum([len(s.tokens) for s in pred])

    # both must be equal
    assert n_tokens_gold == n_tokens_pred

    # count how many heads match between gold and pred
    scores = np.zeros(shape=(n_tokens_gold,), dtype=np.float32)
    for idx in range(len(gold)):
        scores[idx] = count_correct_heads(gold[idx], pred[idx])

    # return uas score
    return np.sum(scores) / float(n_tokens_gold)


def las(gold: List[Sentence], pred: List[Sentence]):
    pass
