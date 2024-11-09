from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric score.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed metrix score.
        """
        pass

    def get_name(self):
        """Get the name of a metric.

        Returns:
            _type_: The name of the metric.
        """
        return f"{self.__class__.__name__}"


# classification metric
class Accuracy(Metric):
    """Class for computing the accuracy metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the accuracy score.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed accuracy score.
        """
        row_matches = np.all(y_ground == y_pred, axis=1)
        correct_val = np.sum(row_matches)
        return correct_val / len(y_ground)


class Precision(Metric):
    """Class for computing the precision metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the precision score.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed precision score.
        """
        y_ground_flat = y_ground.flatten()
        y_pred_flat = y_pred.flatten()

        true_pos = np.sum((y_pred_flat == 1) & (y_ground_flat == 1))
        false_pos = np.sum((y_pred_flat == 1) & (y_ground_flat == 0))
        return true_pos / (true_pos + false_pos) if (true_pos + false_pos) != \
            0 else 0.0


class Recall(Metric):
    """Class for computing the recall metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the recall score.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed recall score.
        """
        y_ground_flat = y_ground.flatten()
        y_pred_flat = y_pred.flatten()

        true_pos = np.sum((y_pred_flat == 1) & (y_ground_flat == 1))
        false_neg = np.sum((y_pred_flat == 0) & (y_ground_flat == 1))
        return true_pos / (true_pos + false_neg) if (true_pos + false_neg) != \
            0 else 0.0


# regression metric


class MeanSquaredError(Metric):
    """Class for computing the mean squared error (MSE) metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean squared error.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed mean squared error.
        """
        return np.mean((y_ground - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """Class for computing the mean absolute error (MAE) metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean absolute error.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The computed mean absolute error.
        """
        return np.mean(np.abs(y_ground - y_pred))


class RSquared(Metric):
    """Class for computing the R-squared (coefficient of determination)
    metric."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the R-squared score.

        Args:
            y_ground (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The R-squared score.
        """
        total_var = np.var(y_ground)
        unexplained_var = np.var(y_ground - y_pred)
        return 1 - (unexplained_var / total_var)


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
]


def get_metric(name: str) -> Metric:
    """Get a metric instance by name.

    Args:
        name (str): The name of the metric to retrieve.

    Raises:
        ValueError: Error occurs if the provided name does not exist in
        Metrics.

    Returns:
        Metric: An instance of the requested metric.
    """
    if name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    else:
        return RSquared()
