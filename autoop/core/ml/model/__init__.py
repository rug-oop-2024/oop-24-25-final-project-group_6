"""
This package provides utilities for the model.py class.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.classification_mdl import RandomForest
from autoop.core.ml.model.classification.classification_mdl import KNN
from autoop.core.ml.model.classification.classification_mdl import DecisionTree
from autoop.core.ml.model.regression.regression_mdl import LinearRegression
from autoop.core.ml.model.regression.regression_mdl import Ridge
from autoop.core.ml.model.regression.regression_mdl import (
    DecisionTreeRegressor
)


REGRESSION_MODELS = [
    "LinearRegression",
    "Ridge",
    "DecisionTreeRegressor"
]

CLASSIFICATION_MODELS = [
    "RandomForest",
    "KNN",
    "DecisionTree"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge()
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    elif model_name == "RandomForest":
        return RandomForest()
    elif model_name == "KNN":
        return KNN()
    elif model_name == "DecisionTree":
        return DecisionTree()
    else:
        return f"Model {model_name} is not implemented."
