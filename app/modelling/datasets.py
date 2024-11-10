from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

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


def select_target_column(features: List[Feature]) -> Feature:
    """
    Function for selecting features from a list of features. The selection of
    the feature is displayed in a selection box in streamlit.

    Args:
        features (List[Feature]): A list of features from which a feature has
        to be selected from.

    Returns:
        Feature: The selected feature from the streamlit selectbox.
    """
    selected_target_column: Feature = st.selectbox("Select a target column:",
                                                   options=features)

    return selected_target_column


def select_input_columns(features: List[Feature]) -> Feature:
    """
    Function for selecting input features from a list of features. The
    selection of the feature is displayed in a multiselection box in streamlit.

    Args:
        features (List[Feature]): A list of features from which input features
        have to selected from.

    Returns:
        Feature: The selected features from the streamlit multiselectbox.
    """
    selected_input_columns: Feature = st.multiselect("Select input columns:",
                                                     options=features)

    return selected_input_columns
