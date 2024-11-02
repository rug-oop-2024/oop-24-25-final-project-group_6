import streamlit as st
import pandas as pd
from typing import List, Dict, Optional

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model.model import Model
from autoop.functional.feature import detect_feature_types

from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model

from autoop.core.ml.pipeline import Pipeline


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

available_datasets = automl.registry.list(type="dataset")


def select_dataset(datasets: List[Dataset]) -> Dataset:
    dataset_contents: Dict = {}

    for dataset in datasets:
        dataset_contents[dataset.name] = dataset

    selected_dataset_name = st.selectbox(
        label="Select a dataset:",
        options=list(dataset_contents.keys())
    )

    return dataset_contents[selected_dataset_name]


def select_features(dataset: Dataset) -> Feature:
    feature_columns: List[Feature] = detect_feature_types(dataset)

    selected_feature_column: Feature = st.selectbox("Select a feature column:", options=feature_columns)

    return selected_feature_column


def select_model(feature: Feature) -> Model:
    if feature.type == "categorical":
        selected_model = st.selectbox("Choose a model:", options=CLASSIFICATION_MODELS)
    else:
        selected_model = st.selectbox("Choose model:", options=REGRESSION_MODELS)
    selected_model = get_model(selected_model)
    return selected_model


def select_dataset_split() -> float:
    st.write("Please select dataset split: ")
    split_ratio: float = st.slider("Select the training/test split ratio: ", min_value=0.1, max_value=0.9, value=0.8, step= 0.1)
    return split_ratio


def select_metrics() -> List[Metric]:
    st.write("Select Metrics")
    selected_metrics_names: str = st.multiselect("Choose metrics to evaluate the model:", options=METRICS)
    selected_metrics: List[Metric] = [get_metric(metric) for metric in selected_metrics_names]
    return selected_metrics


def display_pipeline_summary(
        selected_dataset: Dataset,
        split_ratio: float,
        selected_metrics: List[Metric],
        selected_model: Model
        ) -> None:
    st.header("Pipeline Summary")

    st.subheader("Selected Dataset:")
    st.write(selected_dataset)

    st.subheader("Train/Test Split Ratio:")
    st.write(split_ratio)

    st.subheader("Selected Metrics:")
    st.write(selected_metrics)

    st.subheader("Selected Models:")
    st.write(selected_model)


def train_pipeline(
        selected_dataset: Dataset,
        split_ratio: float,
        metrics: List[Metric],
        model: Model,
        target_feature: Feature
        ) -> None:

    pipeline = Pipeline(
        metrics=metrics,
        dataset=selected_dataset,
        model=model,
        input_features=detect_feature_types(selected_dataset),
        target_feature=target_feature,
        split=split_ratio
    )

    results = pipeline.execute()
    st.write(results["metrics_train"])
    st.write(results["prediction_train"])
    st.write(results["metrics_test"])
    st.write(results["prediction_test"])


def save_pipeline():
    st.write("Save Pipeline")
    pipeline_name = st.text_input("Enter the name for your pipeline:")
    version = st.text_input("Enter the version for your pipeline:")

    if st.button("Save Pipeline"):
        if pipeline_name and version:
            st.success(f"Pipeline '{pipeline_name}' version '{version}' successfully saved !")
        else:
            st.error("Please enter the name and version.")


selected_dataset: Optional[Dataset] = select_dataset(available_datasets)

selected_feature: Optional[Feature] = None

if isinstance(selected_dataset, Dataset):
    selected_feature = select_features(selected_dataset)
else:
    st.write("No dataset selected.")

selected_model: Optional[List[Model]] = select_model(selected_feature)

split_ratio: float = select_dataset_split()

selected_metrics: Optional[List[Metric]] = select_metrics()

display_pipeline_summary(
    selected_dataset.name,
    str(split_ratio),
    selected_metrics,
    selected_model
)

if st.button("Train pipeline"):
    train_pipeline(
        selected_dataset,
        split_ratio,
        selected_metrics,
        selected_model,
        selected_feature
    )
