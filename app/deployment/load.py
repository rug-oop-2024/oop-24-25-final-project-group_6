from typing import List, Dict

from autoop.core.ml.artifact import Artifact

import streamlit as st


def select_pipeline(pipelines: List[Artifact]) -> Artifact:
    pipeline_contents: Dict = {}

    for pipeline in pipelines:
        pipeline_contents[pipeline.name] = pipeline

    selected_pipeline_box = st.selectbox(
        label="Select a pipeline:",
        options=[pipeline + " (version: " +
                 pipeline_contents[pipeline].version + ")"
                 for pipeline in pipeline_contents.keys()]
    )

    selected_pipeline_name = selected_pipeline_box.split(" (")[0]

    return pipeline_contents[selected_pipeline_name]
