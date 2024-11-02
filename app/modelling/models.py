import streamlit as st
from autoop.core.ml.feature import Feature
from autoop.core.ml.model.model import Model

from autoop.core.ml.model import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    get_model
)


def select_model(feature: Feature) -> Model:
    if feature.type == "categorical":
        selected_model = st.selectbox("Choose a model:",
                                      options=CLASSIFICATION_MODELS)
    else:
        selected_model = st.selectbox("Choose model:",
                                      options=REGRESSION_MODELS)
    selected_model = get_model(selected_model)
    return selected_model
