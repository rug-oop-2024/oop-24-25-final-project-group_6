import streamlit as st
import pickle as pkl
import pandas as pd
from typing import List

from app.core.system import AutoMLSystem
from app.deployment.load import select_pipeline
from app.deployment.predict import predict
from app.modelling.pipeline import display_pipeline_summary
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

available_pipelines: List[Dataset] = automl.registry.list(type="pipeline")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.title("🛠 Pipeline manager")
write_helper_text("In this section, you can design a machine learning " +
                  "pipeline to train a model on a dataset.")

pipeline_artifact = select_pipeline(available_pipelines)
pipeline_data = pkl.loads(pipeline_artifact.data)

display_pipeline_summary(
    selected_dataset=pipeline_data["dataset"],
    selected_feature=pipeline_data["feature_column"],
    split_ratio=pipeline_data["split_ratio"],
    selected_metrics=pipeline_data["selected_metrics"],
    selected_model=pipeline_data["selected_model"]
)

st.header("Predictions")

st.write("Upload a CSV file for predictions:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
dataframe = pd.read_csv(uploaded_file)

if st.button("Predict"):
    predict(pipeline_data["selected_model"], dataframe)
