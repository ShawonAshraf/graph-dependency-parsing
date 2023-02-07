from typing import List

import numpy as np
from data.sentence import Sentence


# count the number of heads that match
# for one sentence


def score_heads(gold: Sentence, pred: Sentence):
    gold_tokens = gold.tokens
    pred_tokens = pred.tokens

    # get heads of all tokens
    gold_heads = np.array([tok.head for tok in gold_tokens])
    pred_heads = np.array([tok.head for tok in pred_tokens])

    n_matches = np.count_nonzero(gold_heads == pred_heads)

    return n_matches


# count head and label matches
# for one sentence


def score_head_and_label(gold: Sentence, pred: Sentence):
    gold_tokens = gold.tokens
    pred_tokens = pred.tokens

    # get heads of all tokens
    gold_heads = [tok.head for tok in gold_tokens]
    pred_heads = [tok.head for tok in pred_tokens]

    # labels - rel
    gold_labels = [tok.rel for tok in gold_tokens]
    pred_labels = [tok.rel for tok in pred_tokens]

    # iterate and match
    scores = np.zeros(shape=(len(gold_heads), ))
    for i in range(len(gold_heads)):
        if gold_heads[i] == pred_heads[i] and gold_labels[i] == pred_labels[i]:
            scores[i] += 1.0

    return np.sum(scores)


# unlabeled attachment


def uas(gold: List[Sentence], pred: List[Sentence]):
    # count all the tokens
    n_tokens_gold = sum([len(s.tokens) for s in gold])
    n_tokens_pred = sum([len(s.tokens) for s in pred])

    # both must be equal
    assert n_tokens_gold == n_tokens_pred

    # count how many heads match between gold and pred
    scores = np.zeros(shape=(len(gold),), dtype=np.float32)
    for idx in range(len(gold)):
        scores[idx] = score_heads(gold[idx], pred[idx])

    # return uas score
    return np.sum(scores) / float(n_tokens_gold)


# labeled attachment


def las(gold: List[Sentence], pred: List[Sentence]):
    # count all the tokens
    n_tokens_gold = sum([len(s.tokens) for s in gold])
    n_tokens_pred = sum([len(s.tokens) for s in pred])

    # both must be equal
    assert n_tokens_gold == n_tokens_pred

    # label and head scores
    scores = np.zeros(shape=(len(gold), ), dtype=np.float32)
    for i in range(len(gold)):
        scores[i] = score_head_and_label(gold[i], pred[i])

    return np.sum(scores) / float(n_tokens_gold)
