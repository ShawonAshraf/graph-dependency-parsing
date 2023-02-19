# from data.conll06_token import Conll06Token
from data.sentence import Sentence
from typing import List


# for a single sentence
# returns a list of feature strings
def extract_feature(sentence: Sentence):
    tokens = sentence.tokens

    features: List[str] = list()

    for _, tok in enumerate(tokens):
        hpos = tokens[tok.head - 1].form
        dpos = tok.form

        # hpos+1
        hpos_plus_1 = tokens[tok.head].form

        # dpos+1
        dpos_plus_1 = tokens[tok._id - 1].form

        feature_str = f"hpos={hpos},dpos={dpos},hpos+1={hpos_plus_1},dpos+1={dpos_plus_1}"
        features.append(feature_str)

    return features
