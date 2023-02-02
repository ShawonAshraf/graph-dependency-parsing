from dataclasses import dataclass
from typing import Any

@dataclass
class Conll06Token:
    _id: Any
    form: str
    lemma: str
    pos: str
    xpos: str
    morph: str
    head: Any
    rel: str
    placeholder_1: str
    placeholder_2: str

    # convert str values of _id and head to int
    # the file reader return strings
    def __post_init__(self):
        self._id = int(self._id)
        self.head = int(self.head)
