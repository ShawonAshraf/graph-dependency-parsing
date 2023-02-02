from dataclasses import dataclass


@dataclass
class Conll06Token:
    _id: int
    form: str
    lemma: str
    pos: str
    xpos: str
    morph: str
    head: int
    rel: str
    placeholder_1: str
    placeholder_2: str
