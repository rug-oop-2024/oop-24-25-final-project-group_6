
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data_frame: pd.DataFrame = dataset.read()
    feature_types_list: List[Feature] = []

    for col in data_frame.columns:
        if data_frame[col].dtype in ['int64', 'float64']:
            feature_types_list.append(Feature("numerical",
                                              data_frame[col].name))
        else:
            feature_types_list.append(Feature("categorical",
                                              data_frame[col].name))

    return feature_types_list


if __name__ == "__main__":
    detect_feature_types(Dataset)
