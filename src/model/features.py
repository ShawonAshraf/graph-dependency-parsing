from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from data.sentence import Sentence


# feature class for a single sentence
@dataclass
class ProcessedInstance:
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
            hform = "ROOT"
            # hform+1
            # HROOT means the head was root
            hform_plus_1 = "<HROOT>"
        else:
            hform = tokens[tok.head - 1].form
            # hform+1
            # if end of sentence is reached, add EOS
            hform_plus_1 = tokens[tok.head].form if not tok.head == len(tokens) else "<EOS>"

        # dependent
        dform = tok.form
        # dform+1
        # if the last token in the sentence add EOS
        dform_plus_1 = tokens[idx + 1].form if not idx == len(tokens) - 1 else "<EOS>"

        # add all to a string
        feature_str = f"hform={hform},dform={dform},hform+1={hform_plus_1},dform+1={dform_plus_1}"
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
    all_features: List[ProcessedInstance] = list()

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

        all_features.append(ProcessedInstance(features, labels))

    return feature_dict, all_features


# features from one sentence
def vectorize_feature_list(fdict: Dict, features: List[str]) -> np.ndarray:
    vector = np.ones(shape=(len(features, ))) * -1.0

    for idx, feat in enumerate(features):
        if feat in fdict.keys():
            vector[idx] = fdict[feat]

    return vector


# create rep for features from all the sentences
def create_vector_representation(fdict: Dict, preprocessed: List[ProcessedInstance]) -> np.ndarray:
    feature_rep = list()
    for pi in preprocessed:
        v = vectorize_feature_list(fdict, pi.features)
        feature_rep.append(v)

    return np.array(feature_rep, dtype=object)


# encode the labels
# convert rel labels to digits
def encode_labels(preprocessed: List[ProcessedInstance]) -> List[Tuple]:
    encoded = list()

    all_labels = {'IM', 'P', 'QMOD', 'OPRD', 'INTJ', 'CONJ', 'SBJ', 'PRT',
                  'APPO', 'GAP', 'PMOD', 'EXTR', 'OBJ', 'LGS',
                  'COORD', 'ADV', 'NMOD', 'ROOT', 'AMOD', 'SUB',
                  'VC', 'PRN', 'DEP', 'GAP-SBJ'}

    label_dict = {}
    for idx, label in enumerate(all_labels):
        label_dict[label] = idx

    # encode
    for idx, pi in enumerate(preprocessed):
        enc = [(label_dict[label[0]], label[1]) for label in pi.labels]
        encoded.append(enc)

    return encoded
