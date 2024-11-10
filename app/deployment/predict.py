from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset

import streamlit as st


def predict(model: Model, dataset: Dataset) -> None:
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
        st.error(e)
