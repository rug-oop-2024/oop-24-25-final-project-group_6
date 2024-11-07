from typing import List, Optional

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model.model import Model
from app.core.system import AutoMLSystem
from app.modelling.pipeline import (
    select_dataset_split,
    select_metrics,
    display_pipeline_summary,
    train_pipeline,
    save_pipeline
)
from app.modelling.datasets import select_dataset, select_features
from app.modelling.models import select_model

import streamlit as st


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning " +
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

available_datasets = automl.registry.list(type="dataset")

selected_dataset: Optional[Dataset] = select_dataset(available_datasets)

selected_feature: Optional[Feature] = None

if isinstance(selected_dataset, Dataset):
    selected_feature = select_features(selected_dataset)
else:
    st.write("No dataset selected.")

selected_model: Optional[List[Model]] = select_model(selected_feature)

split_ratio: float = select_dataset_split()

selected_metrics: Optional[List[Metric]] = select_metrics()

display_pipeline_summary(
    selected_dataset,
    selected_feature,
    split_ratio,
    selected_metrics,
    selected_model
)

if st.button("Train pipeline"):
    st.session_state.train = True

if "train" in st.session_state:
    train_pipeline(
        selected_dataset,
        split_ratio,
        selected_metrics,
        selected_model,
        selected_feature
    )

    save_pipeline(
        selected_dataset,
        selected_feature,
        split_ratio,
        selected_metrics,
        selected_model,
        automl
    )
