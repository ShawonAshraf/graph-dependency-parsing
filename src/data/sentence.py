from dataclasses import dataclass
from typing import List

from .conll06_token import Conll06Token


@dataclass
class Sentence:
    tokens: List[Conll06Token]
