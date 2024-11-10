from autoop.core.ml.dataset import Dataset
from app.core.system import AutoMLSystem

import pandas as pd


def create(file: str) -> Dataset:
    """
    Creates a dataset from a UploadedFile class or a file location.

    Args:
        file: The path the data is stored in, must be in csv format.
    """
    dataframe = pd.read_csv(file)

    dataset: Dataset = Dataset.from_dataframe(
        name=file.name,
        asset_path=file.name,
        data=dataframe,
        version="1.0.0"
    )

    return dataset


def save(dataset: Dataset) -> None:
    """
    Saves a dataset in the automl registry.

    Args:
        dataset (Dataset): The dataset that is getting saved.
    """
    automl = AutoMLSystem.get_instance()

    automl.registry.register(dataset)
