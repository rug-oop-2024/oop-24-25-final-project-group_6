from autoop.core.ml.artifact import Artifact

from abc import abstractmethod, ABC
from copy import deepcopy
import numpy as np
import os
import pickle
from typing import List


class Model(ABC):
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
    _params: dict = dict
    _hyperparameters: dict = dict
    _type: str = str

    def __str__(self) -> str:
        """
        Human readable representation of the model
        """
        return "yo"

    def __init__(self, type: str) -> None:
        """
        Initializer method for the Method class
        """
        self.type = type

    def get_name(self) -> str:
        """
        Gets the name of the class
        Returns:
            str: The class name
        """
        return self.__class__.__name__

    @property
    def parameters(self) -> dict:
        """
        Getter function for the private _params attribute

        :returns:
            Deepcopy of private _params attribute to prevent leakage
        """
        return deepcopy(self._params)

    @property
    def hyperparameters(self) -> dict:
        """
        Getter function for the private _hyperparameters attribute

        :returns:
            Deepcopy of private _hyperparameters attribute to prevent leakage
        """
        return deepcopy(self._hyperparameters)

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
            raise ValueError(f"Invalid model type: '{value}'."
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
        """
        Makes the model class into a artifact.

        Args:
            name (str): The name of the model
        Returns:
            Artifact: The data of the model stored in an artifact.
        """
        return Artifact(
            name=name,
            asset_path=os.path.abspath(__file__),
            metadata={"model_name": self.__class__.__name__},
            tags=["model"],
            data=pickle.dumps(self.parameters),
            type=self.type,
            version="1_0_0"
        )
