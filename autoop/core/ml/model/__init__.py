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
    "linear regression",
    "ridge",
    "decision tree regressor"
]

CLASSIFICATION_MODELS = [
    "random forest",
    "KNN",
    "decision tree"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "linear regression":
        return LinearRegression()
    elif model_name == "ridge":
        return Ridge()
    elif model_name == "decision tree regressor":
        return DecisionTreeRegressor()
    elif model_name == "random forest":
        return RandomForest()
    elif model_name == "KNN":
        return KNN()
    elif model_name == "decision tree":
        return DecisionTree()
    else:
        return f"Model {model_name} is not implemented."
