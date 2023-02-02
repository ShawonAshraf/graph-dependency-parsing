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

    # create a representation like the conllu06 format
    def __str__(self):
        attributes = self.__dict__
        s = ""
        for attrib, value in attributes.items():
            # check if a token property has new line
            v = str(value)
            if "\n" in v:
                s += v
            else:
                s += v + "\t"
        return s
