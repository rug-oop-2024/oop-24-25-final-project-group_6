from autoop.core.ml.artifact import Artifact

import io

import pandas as pd


class Dataset(Artifact):
    """
    Dataset class which inherits from Artifact. The dataset class handles
    data.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Dataset class which inherits from Artifact. The dataset class handles
        data.

        *args (list): The arguments given for the dataset
        **kwargs (dict): The keywords arguments given for the dataset.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1_0_0"
                       ) -> "Dataset":
        """
        Static method of Dataset that from a dataframe creates it to a Dataset.

        Args:
            data (pd.DataFrame): The dataframe that will be recreated to a
            Dataset class.
            name (str): The name of the dataset.
            asset_path (str): The OS path the dataset is saved in.
            version (str): The version of the dataset.

        Returns:
            Dataset: The dataset that is created from the dataframe.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the data from the dataset in streams of bytes. Returns
        pandas dataframe of the data.

        Returns:
            pd.DataFrame: Pandas dataframe made from the data.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the data provided from a dataframe to streams of bytes.

        Returns:
            bytes: The data saved in bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
