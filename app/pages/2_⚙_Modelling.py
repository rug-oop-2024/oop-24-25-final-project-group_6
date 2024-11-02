import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model



st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

def select_model(task_type: str):
    st.write("Select Model")
    if task_type == "classification":
        selected_model = st.selectbox("Choose a model:", options=CLASSIFICATION_MODELS)
    else:
        selected_model = st.selectbox("Choose model:", options=REGRESSION_MODELS)
    return selected_model


def select_dataset_split(dataset: Dataset):
    st.write("Please select dataset split: ")
    split_ratio = st.slider("Select the training/test split ratio: ", min_value=0.1, max_value=0.9, value=0.8, step= 0.1)
    return split_ratio

def select_metrics():
    st.write("Select Metrics")
    selected_metrics = st.multiselect("Choose metrics to evaluate the model:", options=METRICS)
    return selected_metrics

def display_pipeline_summary(selected_dataset, split_ratio, selected_metrics, selected_model):
    st.write("Pipeline Summary")
    st.write("Selected Dataset:")
    st.write(selected_dataset)
    st.write("Train/Test Split Ratio:")
    st.write(split_ratio)
    st.write("Selected Metrics:")
    st.write(selected_metrics)
    st.write("Selected Models:")
    st.write(selected_model)


def train_pipeline(selected_dataset, split_ratio, selected_metrics, selected_model):
    # model = get_model(selected_model)
    # metrics = [get_metric[metric] for metric in selected_metrics]
    pass


def save_pipeline():
    st.write("Save Pipeline")
    pipeline_name = st.text_input("Enter the name for your pipeline:")
    version = st.text_input("Enter the version for your pipeline:")
    
    if st.button("Save Pipeline"):
        if pipeline_name and version:
            st.success(f"Pipeline '{pipeline_name}' version '{version}' successfully saved !")
        else:
            st.error("Please enter the name and version.")