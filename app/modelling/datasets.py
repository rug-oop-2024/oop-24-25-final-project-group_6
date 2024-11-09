from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

from typing import List, Dict

import streamlit as st


def select_dataset(datasets: List[Dataset]) -> Dataset:
    """
    Function for selecting datasets from a list of datasets. The selection of
    the dataset is displayed in a selectionbox in streamlit.

    Args:
        datasets (List[Dataset]): A list of datasets from which a dataset has
        to be selected from.

    Returns:
        Dataset: The selected dataset from the streamlit selectbox.
    """
    dataset_contents: Dict = {}

    for dataset in datasets:
        dataset_contents[dataset.name] = dataset

    selected_dataset_name = st.selectbox(
        label="Select a dataset:",
        options=list(dataset_contents.keys())
    )

    return dataset_contents[selected_dataset_name]


def select_features(features: List[Feature]) -> Feature:
    """
    Function for selecting features from a list of features. The selection of
    the feature is displayed in a selection box in streamlit.

    Args:
        features (List[Feature]): A list of features from which a feature has
        to be selected from.

    Returns:
        Dataset: The selected feature from the streamlit selectbox.
    """
    feature_columns: List[Feature] = detect_feature_types(features)

    selected_feature_column: Feature = st.selectbox("Select a feature column:",
                                                    options=feature_columns)

    return selected_feature_column
