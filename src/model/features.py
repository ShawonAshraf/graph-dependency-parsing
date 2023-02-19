# from data.conll06_token import Conll06Token
from dataclasses import dataclass
from typing import List, Tuple

from data.sentence import Sentence


# feature class for a single sentence
@dataclass
class Feature:
    features: List[str]
    labels: List[Tuple]


# for a single sentence
# returns a list of feature strings


def extract_feature(sentence: Sentence) -> List[str]:
    tokens = sentence.tokens

    features: List[str] = list()

    for idx, tok in enumerate(tokens):
        # check if head is root (_id = 0)
        if tok.head == 0:
            hpos = "ROOT"
            # hpos+1
            # HROOT means the head was root
            hpos_plus_1 = "<HROOT>"
        else:
            hpos = tokens[tok.head - 1].form
            # hpos+1
            # if end of sentence is reached, add EOS
            hpos_plus_1 = tokens[tok.head - 1].form if not tok.head == len(tokens) else "<EOS>"

        # dependent
        dpos = tok.form
        # dpos+1
        # if the last token in the sentence add EOS
        dpos_plus_1 = tokens[idx + 1].form if not idx == len(tokens) - 1 else "<EOS>"

        # add all to a string
        feature_str = f"hpos={hpos},dpos={dpos},hpos+1={hpos_plus_1},dpos+1={dpos_plus_1}"
        features.append(feature_str)

    return features


# for a single sentence
def get_labels(sentence: Sentence) -> List[Tuple]:
    # a list of (rel, head) tuples
    labels: List[Tuple] = list()

    tokens = sentence.tokens
    for tok in tokens:
        labels.append((tok.rel, tok.head))

    return labels


def preprocess(sentences: List[Sentence]):
    feature_dict = dict()
    all_features: List[Feature] = list()

    feature_counter = 0

    for idx, sentence in enumerate(sentences):
        features = extract_feature(sentence)
        labels = get_labels(sentence)

        for feat in features:
            if feat not in feature_dict.keys():
                feature_dict[feat] = feature_counter + 1
                feature_counter += 1  # inc after each entry
            else:
                continue

        all_features.append(Feature(features, labels))

    return feature_dict, all_features
