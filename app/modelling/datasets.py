import streamlit as st

from typing import List, Dict
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


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

    selected_feature_column: Feature = st.selectbox("Select a feature column:",
                                                    options=feature_columns)

    return selected_feature_column
