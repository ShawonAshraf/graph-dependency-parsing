from typing import List

from data.conll06_token import Conll06Token


def extract_feature_permutation(head, token: Conll06Token, tokens: List[Conll06Token]) -> List[str]:
    if head._id == 0:
        hform = "ROOT"
        hform_plus_1 = "<HROOT>"
    else:
        hform = tokens[head._id - 1].form
        hform_plus_1 = tokens[head._id].form if not head._id == len(tokens) else "<EOS>"

    dform = token.form
    dform_plus_1 = tokens[token._id].form if not token._id == len(tokens) else "<EOS>"

    return [
        f"hform={hform}",
        f"dform = {dform}",
        f"hform+1={hform_plus_1}",
        f"dform+1={dform_plus_1}"
    ]
