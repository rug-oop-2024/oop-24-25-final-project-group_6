from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

import numpy as np
from autoop.core.ml.model.model import Model


class LinearRegression(Model):
    """Linear Regression model for predicting continuous outcomes."""

    def __init__(self) -> None:
        super().__init__(type="regression")
        self.model = SklearnLinearRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the linear regression model.

        Args:
            observations (np.ndarray): The input data for training the model.
            ground_truth (np.ndarray): The true continuous values corresponding to the input data.
        """
        self.model.fit(observations, ground_truth)
        self._params = {"coef_": self.model.coef_, "intercept_": self.model.intercept_}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts continuous outcomes for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted continuous values.
        """
        prediction = self.model.predict(observations)
        return prediction


class Ridge(Model):
    """Ridge Regression model for predicting continuous outcomes with L2 regularization."""

    def __init__(self):
        super().__init__(type="regression")
        self.model = SklearnRidge()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the ridge regression model.

        Args:
            observations (np.ndarray): The input data for training the model.
            ground_truth (np.ndarray): The true continuous values corresponding to the input data.
        """
        self.model.fit(observations, ground_truth)
        self._params = {"coef_": self.model.coef_, "intercept_": self.model.intercept_}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts continuous outcomes for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted continuous values.
        """
        return self.model.predict(observations)


class DecisionTreeRegressor(Model):
    """Decision Tree Regressor model for predicting continuous outcomes."""

    def __init__(self) -> None:
        super().__init__(type="regression")
        self.model = SklearnDecisionTreeRegressor()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the decision tree regressor model.

        Args:
            observations (np.ndarray): The input data for training the model.
            ground_truth (np.ndarray): The true continuous values corresponding to the input data.
        """
        self.model.fit(observations, ground_truth)
        self._params = {
            "feature_importances_": self.model.feature_importances_,
            "max_depth": self.model.get_depth(),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts continuous outcomes for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted continuous values.
        """
        prediction = self.model.predict(observations)
        return prediction
