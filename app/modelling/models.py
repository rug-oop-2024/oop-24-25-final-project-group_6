from autoop.core.ml.feature import Feature
from autoop.core.ml.model.model import Model
from autoop.core.ml.model import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    get_model
)

import streamlit as st


def select_model(feature: Feature) -> Model:
    """
    Function for selecting a model, depending on the type of feature.

    Args:
        feature (Feature): The type of feature either categorical or numerical.
    Returns:
        Model: The model that is selected from the streamlit selection box.
    """
    if feature.type == "categorical":
        selected_model = st.selectbox("Choose a model:",
                                      options=CLASSIFICATION_MODELS)
    else:
        selected_model = st.selectbox("Choose model:",
                                      options=REGRESSION_MODELS)
    selected_model = get_model(selected_model)
    return selected_model
