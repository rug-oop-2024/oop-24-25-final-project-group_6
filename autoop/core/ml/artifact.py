from typing import List, Optional
from pydantic import BaseModel, Field, root_validator
from pydantic import PrivateAttr
import base64


class Artifact(BaseModel):
    """
    An artifact is an abstract object refering to an asset and includes
    information about this specific assert (such as models, data, etc.)

    :param tags: Tags provide additional information about what the model provides. May be used for filtering content in the app later. If we find no use case it may be deprecated later.
    :type tags: Optional[List[str]]

    :param meta_data: Data that provides information about other data,
    but not the content itself.
    :type meta_data: Optional[List[str]]

    :param name: The name of the Artifact
    :type name: Optional[str]

    :param type: Provides information about what type of asset the artifact
    refers to. Such as "model:torch" for torch model assets.
    :type type: Optional[str]

    :param id: For maintaining referential identity, such that if there are
    similair assets that it is obvious they are distinct.
    :type id: Optional[str]

    :param asset_path: The path of the asset the artifact is refering to.
    :type asset_path: Optional[str]

    :param data: The contents of the asset the artifact is refering to. For datasets it is the data in bytes format, for models the parameters and
    hyperparameters in bytes format.
    :type data: Optional[str]

    :param version: The version of the asset the artifat is refering to.
    :type version: Optional[str]
    """
    tags: Optional[List[str]] = Field(default_factory=list)
    meta_data: Optional[dict] = Field(default_factory=dict)

    name: Optional[str] = Field(default_factory=str)
    type: Optional[str] = Field(default_factory=str)
    id: Optional[str] = Field(default_factory=None)

    _asset_path: Optional[str] = PrivateAttr(default_factory=str)
    _data: Optional[bytes] = PrivateAttr(default_factory=bytes)
    _version: Optional[str] = PrivateAttr(default="1.0.0")

    def __init__(self, **kwargs) -> None:
        """
        Initialisor method of
        """
        super().__init__(**kwargs)

    @root_validator(pre=True)
    def set_id(cls, values):
        """
        Sets the id attribute.

        Method that is ran, before the actual fields of pydantic are validated.
        This method is used tp make modifactions to the raw data of id before
        fields are valided.
        """
        asset_path = values.get("_asset_path", "")
        version = values.get("_version", "1.0.0")
        values['id'] = f"{base64.b64encode(asset_path.encode()).decode()}:{version}"
        return values

    @property
    def data(self) -> str:
        return self._data

    @data.setter
    def data(self, value: str) -> None:
        if isinstance(value, bytes):
            self._data = value
        else:
            print("Is ran")
            raise ValueError(f"Invalid data type: '{value}'. "
                             "Please store data in type bytes.")

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @asset_path.setter
    def asset_path(self, value: str) -> None:
        if True:
            self._asset_path = value
        else:
            raise ValueError(f"Invalid path: '{value}'.")

    @property
    def version(self) -> str:
        return self._version

    def _is_version(self, value: str) -> bool:
        try:
            version_indices: List[str] = value.split(".")
            if len(version_indices) != 3:
                return False
            return all(part.isdigit() for part in version_indices)
        except ValueError:
            return False

    @version.setter
    def version(self, value: str) -> None:
        if self._is_version(value):
            self._version = value
        else:
            raise ValueError(f"Invalid version format: '{value}'. "
                             "Please use the 'x.y.z' format, e.g., '1.0.0'.")

    def read(self) -> bytes:
        return self.data
