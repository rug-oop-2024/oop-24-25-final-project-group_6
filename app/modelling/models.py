from autoop.core.ml.feature import Feature
from autoop.core.ml.model.model import Model
from autoop.core.ml.metric import (
    Metric
)
from autoop.core.ml.model import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    get_model
)

from typing import List

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


def is_valid_target_for_prediction(
    metric_results: List[Metric],
    target_feature: Feature
) -> str | None:
    """
    Method that tests whether a target column is valid for prediction

    Returns:
        str | None: Returns string if there is an error, returns None if there
        is not error.
    """
    if target_feature.type == "numerical":
        for result in metric_results:
            metric = result[0]
            metric_result = result[1]

            match metric:
                case "RSquared":
                    if abs(metric_result) < 0.6:
                        return "RSquared lower than 0.6"

    elif target_feature.type == "categorical":
        for result in metric_results:
            metric = result[0]
            metric_result = result[1]

            match metric:
                case "Accuracy":
                    if abs(metric_result) < 0.5:
                        return "Accuracy lower than 0.5"
                case "Precision":
                    if abs(metric_result) < 0.5:
                        return "Precision lower than 0.5"
                case "Recall":
                    if abs(metric_result) < 0.5:
                        return "Recall lower than 0.5"
