from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.regression_mdl import LinearRegression
from autoop.core.ml.metric import MeanSquaredError

import pandas as pd
import unittest

from sklearn.datasets import fetch_openml


class TestPipeline(unittest.TestCase):
    """
    Testing class for testing the pipeline class
    """
    def setUp(self) -> None:
        """
        Method that is ran before any actual tests are ran.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=LinearRegression(),
            input_features=list(filter
                                (lambda x: x.name != "age", self.features)
                                ),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self) -> None:
        """
        Tests the init method of the pipeline class.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self) -> None:
        """
        Tests the preprocessing method of the pipeline class.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self) -> None:
        """
        Tests the split data method of the pipeline class.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(
            self.pipeline._train_X[0].shape[0],
            int(0.8 * self.ds_size)
        )
        self.assertEqual(
            self.pipeline._test_X[0].shape[0],
            self.ds_size - int(0.8 * self.ds_size)
        )

    def test_train(self) -> None:
        """
        Tests the train method of the pipeline class.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self) -> None:
        """
        Tests the evaluate method of the pipeline class.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate(
            self.pipeline._train_X,
            self.pipeline._train_y,
            data_type="training"
        )
        self.assertIsNotNone(self.pipeline._metrics_results_train)
        self.assertIsNotNone(self.pipeline._prediction_train)
        self.pipeline._evaluate(
            self.pipeline._test_X,
            self.pipeline._test_y,
            data_type="evaluation"
        )
        self.assertIsNotNone(self.pipeline._metrics_results_train)
        self.assertIsNotNone(self.pipeline._prediction_test)
        self.assertEqual(len(self.pipeline._metrics_results_test), 1)
