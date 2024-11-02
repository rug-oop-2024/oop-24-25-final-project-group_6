
from pydantic import BaseModel, Field
from pydantic.fields import PrivateAttr
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature():
    _type: str
    name: str

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value: str):
        if value not in ["numerical", "categorical"]:
            raise ValueError("Type can only be either numerical or " +
                             "categorical")
        else:
            self._type = value

    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name

    def __str__(self):
        return self.name + ": " + self.type
