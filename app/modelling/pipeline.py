import streamlit as st
from typing import List
import pickle as pkl

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types

from autoop.core.ml.metric import METRICS, get_metric

from autoop.core.ml.pipeline import Pipeline


def select_dataset_split() -> float:
    """Prompt the user to select the train-test split ratio using a slider.

    Returns:
        float: The selected split ratio.
    """
    st.write("Please select dataset split: ")
    split_ratio: float = st.slider(
        "Select the training/test split ratio: ",
        min_value=0.1,
        max_value=0.9,
        value=0.8,
        step=0.1
    )
    return float(split_ratio)


def select_metrics() -> List[Metric]:
    """Prompts the user to select one or more evaluation 
    metrics from a predefined list.


    Returns:
        List[Metric]: A list of selected Metric objects.
    """
    st.write("Select Metrics")
    selected_metrics_names: str = st.multiselect("Choose metrics to evaluate "
                                                 + "the model:",
                                                 options=METRICS)

    selected_metrics: List[Metric] = [get_metric(metric) for metric in
                                      selected_metrics_names]
    return selected_metrics


def display_pipeline_summary(
        selected_dataset: Dataset,
        selected_feature: Feature,
        split_ratio: float,
        selected_metrics: List[Metric],
        selected_model: Model
        ) -> None:
    """Displays a summary of the pipeline configuration, 
    including dataset information, feature column details, split ratio, 
    selected metrics, and model information.

    Args:
        selected_dataset (Dataset): The chosen dataset for the pipeline.
        selected_feature (Feature): The target feature for the pipeline.
        split_ratio (float): The ratio of data used for training.
        selected_metrics (List[Metric]): The metrics selected for evaluation.
        selected_model (Model): The machine learning model chosen for the pipeline.
    """
    st.markdown("---")
    st.header("🛠️ Pipeline Summary")

    st.subheader("📊 Dataset Information")
    if selected_dataset:
        st.write(f"**Dataset**: {selected_dataset.name}")
    else:
        st.error("❌ No dataset selected. Selecting a dataset is required to " +
                 "proceed.")

    st.subheader("**📂 Feature column**")
    if selected_feature:
        st.write(f"**Name**: {selected_feature.name}")
        st.write(f"**Type**: {selected_feature.type}")
    else:
        st.error("❌ No feature column selected. Selecting a training" +
                 " - feature column is required to proceed.")

    st.subheader("🧮 Training - Test split")
    if split_ratio:
        test_split: float = 1 - float(split_ratio)
        st.markdown("**Training data ratio**: {:.1f}"
                    .format(float(split_ratio)))
        st.markdown("**Test data ratio**: {:.1f}".format(test_split))
    else:
        st.error("❌ No training - test split selected. Selecting a training" +
                 " - test split is required to proceed.")

    st.subheader("🎯 Metrics")
    if selected_metrics:
        st.write(", ".join([m.__class__.__name__ for m in selected_metrics]))
    else:
        st.error("❌ No metric selected.")

    st.subheader("🔍 Model Information")
    if selected_model:
        st.markdown(f"**Model Name:** {selected_model.__class__.__name__}")
        st.markdown(f"**Model Type:** {selected_model.type.capitalize()}")
    else:
        st.error("❌ No model selected. Selecting a model is required " +
                 "to proceed.")
    st.markdown("---")


def train_pipeline(
        selected_dataset: Dataset,
        split_ratio: float,
        metrics: List[Metric],
        model: Model,
        target_feature: Feature
        ) -> None:
    """Train a machine learning pipeline using the selected dataset, 
    model, split ratio, metrics, and target feature.

    Args:
        selected_dataset (Dataset): The dataset for training.
        split_ratio (float): The proportion of data used for training.
        metrics (List[Metric]): A list of evaluation metrics.
        model (Model): The machine learning model to be trained.
        target_feature (Feature): The target feature for prediction.
    """
    try:
        pipeline = Pipeline(
            metrics=metrics,
            dataset=selected_dataset,
            model=model,
            input_features=detect_feature_types(selected_dataset),
            target_feature=target_feature,
            split=split_ratio
        )
    except Exception as e:
        st.error(e)

    st.header("🚀 Pipeline results")

    results = pipeline.execute()
    st.subheader("📊 Training Metrics")
    for metric_results in results["metrics_train"]:
        st.markdown(f"**{metric_results[0]}**: {metric_results[1]}")

    st.subheader("🧪 Test Metrics")
    for metric_results in results["metrics_test"]:
        st.markdown(f"**{metric_results[0]}**: {metric_results[1]}")


def save_pipeline(
        dataset: Dataset,
        feature_column: Feature,
        split_ratio: float,
        metrics: Metric,
        model: Model,
        automl: AutoMLSystem
        ) -> bool:
    """Saves the trained pipeline configuration as an artifact.

    Args:
        dataset (Dataset): The dataset used in the pipeline.
        feature_column (Feature): The target feature column.
        split_ratio (float): The split ratio used for training and testing.
        metrics (Metric): The evaluation metrics.
        model (Model): The machine learning model used in the pipeline.
        automl (AutoMLSystem): The AutoML system instance for registering the artifact.

    Returns:
        bool: "True" if the pipeline was saved successfully and "False" otherwise.
    """
    st.subheader("💾 Save Pipeline")
    pipeline_name = st.text_input("Enter the name for your pipeline:")
    version = st.text_input("Enter the version for your pipeline:")

    if st.button("Save Pipeline"):
        if pipeline_name and version:
            try:
                pipeline_artifact: Artifact = Artifact(
                    name=pipeline_name,
                    type="pipeline",
                    version=version,
                    asset_path=pipeline_name + ".pkl",
                    data=pkl.dumps({
                        "dataset": dataset,
                        "feature_column": feature_column,
                        "split_ratio": split_ratio,
                        "selected_metrics": metrics,
                        "selected_model": model
                    })
                )
                automl.registry.register(pipeline_artifact)
            except Exception as e:
                st.error(e)
                return False
            st.success(f"Pipeline '{pipeline_name}' version '{version}' " +
                       "successfully saved !")
            return True
        else:
            st.error("Please enter the name and version.")
            return False

    st.markdown("---")
