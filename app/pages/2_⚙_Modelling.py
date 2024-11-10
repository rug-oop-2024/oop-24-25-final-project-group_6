from typing import List, Optional

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model.model import Model
from autoop.functional.feature import detect_feature_types
from app.core.system import AutoMLSystem
from app.modelling.pipeline import (
    select_dataset_split,
    select_metrics,
    display_pipeline_summary,
    train_pipeline,
    save_pipeline
)
from app.modelling.datasets import (
    select_dataset,
    select_target_column,
    select_input_columns
)
from app.modelling.models import select_model
import streamlit as st


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Writes helper text

    Args:
        text (str): The text that is written
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

available_datasets = automl.registry.list(type="dataset")


def modelling_page(available_datasets: List[Dataset]) -> None:
    """
    Function to display the modelling page.

    Args:
        available_datasets (List[Dataset]): The datasets that can be used for
        modelling.
    """
    selected_dataset: Optional[Dataset] = select_dataset(available_datasets)

    feature_columns: List[Feature] = detect_feature_types(selected_dataset)

    selected_target = select_target_column(feature_columns)

    input_features = [feature for feature in
                      feature_columns
                      if feature != selected_target]

    selected_input_columns = select_input_columns(input_features)

    selected_model: Optional[List[Model]] = select_model(selected_target)

    split_ratio: float = select_dataset_split()

    selected_metrics: Optional[List[Metric]] = select_metrics(selected_target)

    display_pipeline_summary(
        selected_dataset,
        selected_target,
        selected_input_columns,
        split_ratio,
        selected_metrics,
        selected_model
    )

    if st.button("Train pipeline"):
        st.session_state.train = True

    if "train" in st.session_state:
        try:
            pipeline, is_valid_target_column = train_pipeline(
                selected_dataset,
                split_ratio,
                selected_metrics,
                selected_model,
                selected_target,
                selected_input_columns
            )
            if is_valid_target_column is None:
                save_pipeline(
                    pipeline=pipeline,
                    automl=automl,
                    dataset=selected_dataset,
                    metrics=selected_metrics
                )
            else:
                st.markdown(
                    f"""
                    <div style="color: #B22222; border: 2px solid #7C0A02;
                    padding: 15px;
                    background-color: #7C0A02; border-radius: 8px;">
                        <h2 style="text-align: center; color: #FFFDD3;">🚫 Error
                        : Target column not valid for prediction</h2>
                        <p style="text-align: center; font-size: 16px; color:
                        #FFFDD3;">
                            {is_valid_target_column}. Therefore
                            the target column is not valid for prediction.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(e)


if available_datasets:
    modelling_page(available_datasets)
else:
    st.markdown(
        """
        <div style="color: #B22222; border: 2px solid #7C0A02; padding: 15px;
        background-color: #7C0A02; border-radius: 8px;">
            <h2 style="text-align: center; color: #FFFDD3;">🚫 Error: No
            Dataset Saved</h2>
            <p style="text-align: center; font-size: 16px; color: #FFFDD3;">
                Required <strong>dataset save</strong> is not available.
                Please ensure that a dataset has been saved.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
