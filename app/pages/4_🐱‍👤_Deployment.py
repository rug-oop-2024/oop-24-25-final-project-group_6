import streamlit as st
import pickle as pkl
import pandas as pd
from typing import List

from app.core.system import AutoMLSystem
from app.deployment.load import select_pipeline
from app.deployment.predict import predict
from app.modelling.pipeline import display_pipeline_summary, train_pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import Metric
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import get_model

automl = AutoMLSystem.get_instance()

available_pipelines: List[Artifact] = automl.registry.list(type="pipeline")


def write_helper_text(text: str) -> None:
    """
    Writes helper text

    Args:
        text (str): The text that is written
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.title("🛠 Pipeline manager")
write_helper_text("In this section, you can design a machine learning "
                  "pipeline to train a model on a dataset.")


def deployment_page(available_pipelines: List[Artifact]) -> None:
    """
    Function that displays the deployment page.
    """
    pipeline_artifact = select_pipeline(available_pipelines)

    pipeline_data: dict = pkl.loads(pipeline_artifact.data)
    pipeline_artifacts: List[Artifact] = pipeline_data["artifacts"]

    metrics = None
    dataset = None
    target_feature = None
    split = None
    model = None

    for artifact in pipeline_artifacts:
        if artifact.name == "metrics_list":
            metrics: List[Metric] = pkl.loads(artifact.data)
        elif artifact.type == "dataset":
            dataset: List[Metric] = artifact
        elif artifact.name == "pipeline_config":
            pipeline_data = pkl.loads(artifact.data)
            target_feature: Feature = pipeline_data["target_feature"]
            split: float = pipeline_data["split"]
        elif artifact.name.startswith("pipeline_model"):
            model = get_model(artifact.metadata["model_name"])

    display_pipeline_summary(
        selected_dataset=dataset,
        selected_feature=target_feature,
        split_ratio=split,
        selected_metrics=metrics,
        selected_model=model
    )

    train_pipeline(
        selected_dataset=dataset,
        split_ratio=split,
        metrics=metrics,
        model=model,
        target_feature=target_feature
    )

    st.header("Predictions")

    st.write("Upload a CSV file for predictions:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        prediction_dataset = Dataset.from_dataframe(
            dataframe,
            uploaded_file.name,
            uploaded_file.name
        )

        if st.button("Predict"):
            predict(
                model=model,
                dataset=prediction_dataset
            )


if available_pipelines:
    deployment_page(available_pipelines)
else:
    st.markdown(
        """
        <div style="color: #B22222; border: 2px solid #7C0A02; padding: 15px;
        background-color: #7C0A02; border-radius: 8px;">
            <h2 style="text-align: center; color: #FFFDD3;">🚫 Error: No
            Pipeline Saved</h2>
            <p style="text-align: center; font-size: 16px; color: #FFFDD3;">
                Required <strong>pipeline save</strong> is not available.
                Please ensure that the pipeline has completed successfully and
                has been saved.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
