from autoop.core.ml.artifact import Artifact

from abc import abstractmethod, ABC
from copy import deepcopy
from typing import List
import os
import pickle

import numpy as np
from pydantic import BaseModel
from pydantic.fields import PrivateAttr


class Model(BaseModel, ABC):
    """
    Abstract class for training models based on supervised learning

    This class serves as a blueprint for training models that use supervised
    learning machine learning algorithms.

    Subclasses have to use the fit() and predict() methods for training and
    predicting their training models.

    It uses a private attribute _params to keep track of important
    values that have to be stored within the subclasses itself. It also
    creates a deepcopy to prevent leakage.
    """
    _params: dict = PrivateAttr(default_factory=dict)
    _hyperparameters: dict = PrivateAttr(default_factory=dict)
    _type: str = PrivateAttr(default_factory=str)

    def __init__(self, type):
        self._type = type

    @property
    def parameters(self) -> dict:
        """
        Getter function for the private _params attribute

        :returns:
            Deepcopy of private _params attribute to prevent leakage
        """
        return deepcopy(self._params)

    @property
    def type(self) -> str:
        """
        Getter function for the private _type attribute

        :returns:
            private _type attribute
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Setter function for private _type attribute

        Ensures only 'classification' or 'regression' are valid types for the
        model.

        :param value: The type of the model ('regression' or 'classification')

        :raises ValueError: If value is not one of the allowed types.
        """

        allowed_types: List[str] = ["regression", "classification"]

        if value in allowed_types:
            self._type = value
        else:
            raise ValueError(f"Invalid model type: '{value}'." +
                             " Allowed types are: {', '.join(allowed_types)}.")

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains models based on observations and ground_truth

        :param observations: np.ndarray
            The input data for the training model

        :param ground_truth: np.ndarray
            The accepted labels corresponding to the input data

        :returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes predictions based on provided observations

        :param observations: np.ndarray
            The input data for the training model

        :returns:
            np.ndarray: Predicted values based on the input data.
        """
        pass

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(
            name=name,
            asset_path=os.path.abspath(__file__),
            meta_data={},
            tags=[],
            data=pickle.dumps(self.parameters),
            type=self.type,
            # model might get a version attribute for saving the same model
            version="1.0.0"
        )
