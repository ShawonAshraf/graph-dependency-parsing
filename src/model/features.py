from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm.auto import tqdm

import numpy as np
from data.sentence import Sentence


# feature class for a single sentence
@dataclass
class ProcessedInstance:
    features: List[List[str]]
    labels: List[Tuple]


# for a single sentence
# returns a list of feature strings


def extract_feature(sentence: Sentence) -> List[List[str]]:
    tokens = sentence.tokens

    features = list()

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

        # create indivudal features as strings and
        # and add them to the features list
        features.append([
            f"hform={hform}",
            f"dform = {dform}",
            f"hform+1={hform_plus_1}",
            f"dform+1={dform_plus_1}"

        ])

    return features


# for a single sentence
def get_labels(sentence: Sentence) -> List[Tuple]:
    # a list of (rel, head) tuples
    labels: List[Tuple] = list()

    tokens = sentence.tokens
    for tok in tokens:
        labels.append((tok.rel, tok.head))

    return labels


# gets features from sentences
def preprocess(sentences: List[Sentence]):
    feature_dict = dict()
    all_features: List[ProcessedInstance] = list()

    # counts index for feature dict keys
    fc = 0
    for _, sentence in tqdm(enumerate(sentences), desc="preprocess"):
        sentence_features = extract_feature(sentence)
        labels = get_labels(sentence)

        for token_features in sentence_features:
            for tokf in token_features:
                if tokf not in feature_dict.keys():
                    feature_dict[tokf] = fc + 1
                    fc += 1

        all_features.append(ProcessedInstance(sentence_features, labels))

    return feature_dict, all_features


# features from one sentence
def vectorize_feature_list(fdict: Dict, sentence_features: List[List[str]]) -> np.ndarray:
    vector = np.ones(shape=(len(sentence_features, ), len(sentence_features[0]))) * -1.0

    for i, token_features in enumerate(sentence_features):
        for j, tokf in enumerate(token_features):
            if tokf in fdict.keys():
                vector[i][j] = fdict[tokf]

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
def encode_labels(preprocessed: List[ProcessedInstance]) -> List[List[Tuple]]:
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
