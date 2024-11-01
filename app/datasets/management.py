from autoop.core.ml.dataset import Dataset
from app.core.system import AutoMLSystem

import pandas as pd
import os


def create(file) -> Dataset:
    dataframe = pd.read_csv(file)

    datasets_dir = os.path.dirname(os.path.realpath(__file__))
    app_dir = os.path.dirname(datasets_dir)
    project_dir = os.path.dirname(app_dir)
    path_to_saved_file = project_dir + "\\assets\\objects\\" + file.name

    dataset: Dataset = Dataset.from_dataframe(
        name=file.name,
        asset_path=path_to_saved_file,
        data=dataframe,
        version="1.0.0"
    )

    return dataset


def save(dataset: Dataset) -> None:
    automl = AutoMLSystem.get_instance()

    automl.registry.register(dataset)
