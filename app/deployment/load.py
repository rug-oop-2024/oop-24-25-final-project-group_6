from autoop.core.ml.artifact import Artifact

from typing import List, Dict

import streamlit as st


def select_pipeline(pipelines: List[Artifact]) -> Artifact:
    """
    Function for selecting a saved pipelines from a streamlit selection box.

    Args:
        pipelines (List[Artifact]): All saved pipelines
    Returns:
        Artifact: The selected pipeline.
    """
    pipeline_contents: Dict = {}

    for pipeline in pipelines:
        pipeline_contents[pipeline.name] = pipeline

    selected_pipeline_box = st.selectbox(
        label="Select a pipeline:",
        options=[f"{pipeline} (version: {pipeline_contents[pipeline].version})"
                 for pipeline in pipeline_contents.keys()]
    )

    selected_pipeline_name = selected_pipeline_box.split(" (")[0]

    return pipeline_contents[selected_pipeline_name]
