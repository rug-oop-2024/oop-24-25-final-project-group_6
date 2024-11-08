from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """A machine learning that handles the data preprocessing, model training, 
    evluation and artifacts management. 
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ) -> None:
        """Initialize the Pipeline with the provided metrics, dataset, model, input features,
        target feature, and data split ratio.

        Args:
            metrics (List[Metric]): The metric used to evaluate the accuracy of the prediction of the model.
            dataset (Dataset): The dataset used for training and predicting.
            model (Model): The model used to train and predict.
            input_features (List[Feature]): The input features.
            target_feature (Feature): The target feature.
            split (float, optional): The data split ratio. Defaults to 0.8.

        Raises:
            ValueError: If the target feature type does not match the model type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """Returns a string representation of the Pipeline."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Returns the model used in the pipeline.

        Returns:
            Model: The model retrieved by the user.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the pipeline execution to be saved.

        Returns:
            List[Artifact]: List of artifacts including encoders and scalers.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """Registers an artifact with the provided name.

        Args:
            name (str): The name of the artifact.
            artifact (Artifact): The Artifact object to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Applies predefined transformations to prepare the input and 
        target data for use. Also saves these transformations for later use.
        """
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """Splits the data into training and testing sets based on the chosen split ratio.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenate a list of numpy arrays column-wise.

        Args:
            vectors (List[np.array]): List of numpy arrays.

        Returns:
            np.array: A single concatenated numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, x, y, data_type: str) -> None:
        """Evaluates the model on the given data and records the evaluation metrics
        for both training and testing data.

        Args:
            x: Input data for evaluation.
            y: Ground truth labels.
            data_type (str): Indicates whether the evaluation is on 
            'training' or 'evaluation' data.
        """
        X = self._compact_vectors(x)
        Y = y
        metric_result = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            metric_name = metric.get_name()
            result = metric(Y, predictions)
            metric_result.append((metric_name, result))

        if data_type == "training":
            self._metrics_results_train = metric_result
            self._prediction_train = predictions
        elif data_type == "evaluation":
            self._metrics_results_test = metric_result
            self._prediction_test = predictions

    def execute(self) -> dict:
        """Executes the entire pipeline process including preprocessing, splitting, training,
        and evaluation.

        Returns:
            dict: A dictionary containing metrics, predictions, and ground truth for both
                  training and evaluation datasets.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        self._evaluate(self._train_X, self._train_y, data_type="training")
        self._evaluate(self._test_X, self._test_y, data_type="evaluation")

        return {
            "metrics_train": self._metrics_results_train,
            "prediction_train": self._prediction_train,
            "metrics_test": self._metrics_results_test,
            "prediction_test": self._prediction_test,
            "ground_truth_test": self._output_vector,
            "ground_truth_train": self._train_y
        }
