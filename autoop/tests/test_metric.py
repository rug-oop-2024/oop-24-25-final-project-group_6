from autoop.core.ml.metric import (
    Metric,
    Accuracy,
    Precision,
    Recall,
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared
)

import numpy as np
from typing import List
import unittest


class TestMetric(unittest.TestCase):
    """
    Class that unit tests metrics.
    """
    def _test_metric(
            self,
            metric: Metric,
            task_type: str,
            expected_results: List[int]
            ) -> None:
        """
        Private method that handles the testing of metrics.
        Args:
            metric (Metric): The metric that is used
            task_type (str): Task type of categorical or continuous
            expected_results (List[int]): The results the metric expects.
        """
        metric: Metric = metric()
        if task_type == "categorical":
            result_perfect = metric(np.array([1, 1, 1, 1, 1]),
                                    np.array([1, 1, 1, 1, 1]))

            result_ok = metric(np.array([1, 0, 1, 1, 0]),
                               np.array([1, 0, 0, 1, 1]))

            result_failure = metric(np.array([1, 1, 1, 1, 1]),
                                    np.array([0, 0, 0, 0, 0]))

            self.assertEqual(result_perfect, expected_results[0])
            self.assertAlmostEqual(result_ok, expected_results[1], delta=0.1)
            self.assertEqual(result_failure, expected_results[2])

        if task_type == "continuous":
            result_perfect = metric(np.array([0, 100, 200, 300, 400]),
                                    np.array([0, 100, 200, 300, 400]))

            result_ok = metric(np.array([0, 100, 200, 300, 400]),
                               np.array([1, 105, 210, 320, 440]))

            result_failure = metric(np.array([0, 100, 200, 300, 400]),
                                    np.array([0, 0, 0, 0, 0]))

            self.assertAlmostEqual(result_perfect, expected_results[0],
                                   delta=0.1)
            self.assertAlmostEqual(result_ok, expected_results[1],
                                   delta=0.1)
            self.assertAlmostEqual(result_failure, expected_results[2],
                                   delta=0.1)

    def test_accuracy_metric(self) -> None:
        """
        Tests the accuracy metric.
        """
        self._test_metric(Accuracy, "categorical", [1, 0.6, 0])

    def test_precision_metric(self) -> None:
        """
        Tests the precision metric.
        """
        self._test_metric(Precision, "categorical", [1, 0.66, 0])

    def test_recall_metric(self) -> None:
        """
        Tests the recall metric.
        """
        self._test_metric(Recall, "categorical", [1, 0.66, 0])

    def test_mean_squared_error(self) -> None:
        """
        Tests the mean squared error metric.
        """
        self._test_metric(MeanSquaredError, "continuous",
                          [0.0, 425.2, 60000.0])

    def test_mean_absolute_error(self) -> None:
        """
        Tests the mean absolute error metric.
        """
        self._test_metric(MeanAbsoluteError, "continuous", [0.0, 15.2, 200.0])

    def test_r_squared_error(self) -> None:
        """
        Tests the r squared error metric.
        """
        self._test_metric(RSquared, "continuous", [1.0, 0.97874, 0.0])
