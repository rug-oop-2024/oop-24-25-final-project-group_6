from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset
from autoop.functional.preprocessing import preprocess_features
from autoop.functional.feature import detect_feature_types

import streamlit as st


def predict(model: Model, dataset: Dataset) -> None:
    """
    Function for predicting variables. Needs to be implemented.

    Args:
        model (Model): The model to predict from.
        dataset (Dataset): The data to predict with.
    """
    try:
        data = preprocess_features(detect_feature_types(dataset), dataset)
        for d in data:
            if d[2]["type"] == "OneHot":
                return None
        st.write(data)
        model.predict(dataset.read())
    except Exception as e:
        st.error(e)

    st.dataframe(dataset.read())
