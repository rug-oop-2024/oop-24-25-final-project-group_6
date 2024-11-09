from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset
from autoop.functional.preprocessing import preprocess_features
from autoop.functional.feature import detect_feature_types

import streamlit as st


def predict(model: Model, dataset: Dataset, transformers: dict) -> None:
    """
    Function for predicting variables. Needs to be implemented.

    Args:
        model (Model): The model to predict from.
        dataset (Dataset): The data to predict with.
    """
    try:
        predictions = model.predict(dataset.read())
        st.write(predictions)
    except Exception as e:
        st.error("Wrong file format, please provide a valid format.")
        st.error(e)
