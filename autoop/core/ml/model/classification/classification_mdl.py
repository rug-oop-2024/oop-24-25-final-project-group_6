from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
import numpy as np

from autoop.core.ml.model.model import Model


class RandomForest(Model):
    """Random Forest model for classification."""

    def __init__(self) -> None:
        super().__init__(type="classification")
        self.model = RandomForestClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the Random Forest model."""
        self.model.fit(observations, ground_truth)
        self._params = {
            "feature_importances_": self.model.feature_importances_,
            "n_estimators": self.model.n_estimators,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts classes for the given observations."""
        return self.model.predict(observations)


class KNN(Model):
    """K-Nearest Neighbors (KNN) model for classification."""

    def __init__(self) -> None:
        super().__init__(type="classification")
        self.model = SklearnKNN()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the KNN model.

        Args:
            observations (np.ndarray): The input data for training the model.
            ground_truth (np.ndarray): The true labels corresponding to the
            input data.
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts classes for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted class labels.
        """
        prediction = self.model.predict(observations)
        return prediction


class DecisionTree(Model):
    """Decision Tree model for classification."""

    def __init__(self) -> None:
        super().__init__(type="classification")
        self.model = SklearnDecisionTree()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the decision tree model.

        Args:
            observations (np.ndarray): The input data for training the model.
            ground_truth (np.ndarray): The true labels corresponding to the
            input data.
        """
        self.model.fit(observations, ground_truth)
        self._params = {
            "feature_importances_": self.model.feature_importances_,
            "max_depth": self.model.get_depth(),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts classes for the given observations.

        Args:
            observations (np.ndarray): The input data for making predictions.

        Returns:
            np.ndarray: The predicted class labels.
        """
        prediction = self.model.predict(observations)
        return prediction
