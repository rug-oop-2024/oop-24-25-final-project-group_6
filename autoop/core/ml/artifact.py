from typing import List
from copy import deepcopy
import base64


class Artifact():
    """
    An artifact is an abstract object refering to an asset and includes
    information about this specific assert (such as models, data, etc.)

    The artifact class is immutable

    :param tags: Tags provide additional information about what the model
    provides. May be used for filtering content in the app later. If we find
    no use case it may be deprecated later.

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

    :param data: The contents of the asset the artifact is refering to. For
    datasets it is the data in bytes format, for models the parameters and
    hyperparameters in bytes format.
    :type data: Optional[str]

    :param version: The version of the asset the artifat is refering to.
    :type version: Optional[str]
    """
    _tags: List[str]

    _name: str
    _type: str
    _id: str

    _asset_path: str
    _metadata: dict
    _data: bytes
    _version: str

    def __init__(
            self,
            name: str,
            data: bytes,
            type: str = "",
            asset_path: str = "placeholder",
            metadata: dict = {},
            version: str = "1.0.0",
            tags: list = []
    ) -> None:
        """
        Initializer method of artifact class
        """
        self._name = None
        self._tags = None
        self._type = None
        self._id = None
        self._asset_path = None
        self._metadata = None
        self._data = None
        self._version = None

        self.name = name
        self.data = data
        self.type = type
        self.asset_path = asset_path
        self.metadata = metadata
        self.version = version
        self.tags = tags

        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        path = f"{encoded_path}_{self.version}"
        self.id = path.strip("=")

    @property
    def id(self) -> str:
        """
        Getter function for the private attribute id. Is for making the id
        attribute read-only as a artifact is immutable.

        Returns:
            str: Private attribute attribute
        """
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        """
        Setter function for the private attribute id.

        Args:
            value (str): The value that has to be considered to be applied to
            id
        """
        if self._id is not None:
            raise AttributeError("Artifact class is immutable. ID can only"
                                 "be set once.")
        if not isinstance(value, str):
            raise TypeError("ID must be a string.")

        self._id = value

    @property
    def tags(self) -> List[str]:
        """
        Getter function for the private attribute tags. Is for making the tags
        attribute read-only as a artifact is immutable.

        Returns:
            str: Private attribute attribute
        """
        return self._tags

    @tags.setter
    def tags(self, value: List[str]) -> List[str]:
        """
        Setter function for the private attribute tags.

        Args:
            value List[str]: The value that gets applied to the private tags
            attribute.
        """
        if self._tags is not None:
            raise AttributeError("Artifact class is immutable. Tags can only"
                                 "be set once.")
        if not isinstance(value, list):
            raise TypeError("Tags must be a list.")

        self._tags = value

    @property
    def name(self) -> str:
        """
        Getter function for the private attribute name. Is for making the name
        attribute read-only as a artifact is immutable.

        Returns:
            str: Private name attribute
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Setter function for the private attribute name.

        Args:
            value (str): The value that has to be considered to be applied to
            name
        """
        if self._name is not None:
            raise AttributeError("Artifact class is immutable. Name can only"
                                 "be set once.")
        if not isinstance(value, str):
            raise TypeError("Name must be a string.")

        self._name = value

    @property
    def type(self) -> str:
        """
        Getter function for the private attribute type.

        Returns:
            str: Private type attribute.
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Setter function for the private attribute type.

        Args:
            value (str): The value that has to be considered to be applied to
            type
        """
        if self._type is not None:
            raise AttributeError("Artifact class is immutable. Type can only"
                                 "be set once.")
        if not isinstance(value, str):
            raise TypeError("Type must be a string.")

        self._type = value

    @property
    def metadata(self) -> dict:
        """
        Getter method for the private metadata attribute.

        Returns:
            dict: A deepcopy of metadata to prevent leakage.
        """
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, value: dict) -> None:
        """
        Setter method for the private attribute metadata. Only accepts
        dictonaries.

        Args:
            value (dict): The value that has to be set for metadata.
        """
        if self._metadata is not None:
            raise AttributeError("Artifact class is immutable. metadata can"
                                 "only be set once.")
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict.")
        self._metadata = value

    @property
    def data(self) -> bytes:
        """
        Getter method for the private attribute data.

        Returns:
            bytes: Data in bytes.
        """
        return self._data

    @data.setter
    def data(self, value: bytes) -> None:
        """
        Setter method for the private attribute data. Only accepts type of
        bytes.

        Args:
            value (bytes): The value data has to set to in bytes.
        """
        if self._data is not None:
            raise AttributeError("Artifact class is immutable. data can "
                                 "only be set once.")
        if isinstance(value, bytes):
            self._data = value
        else:
            raise ValueError(f"Invalid data type: '{value}'. "
                             "Please store data in type bytes.")

    @property
    def asset_path(self) -> str:
        """
        Getter method for the private attribute asset_path.
        """
        return self._asset_path

    @asset_path.setter
    def asset_path(self, value: str) -> None:
        """
        Setter method for the private attribute asset_path.

        Args:
            value (str): The value that asset_path gets set to.
        """
        if self._asset_path is not None:
            raise AttributeError("Artifact class is immutable. asset_path can"
                                 "only be set once.")
        if not isinstance(value, str):
            raise TypeError("asset_path must be a string.")
        self._asset_path = value

    @property
    def version(self) -> str:
        """
        Getter method for the private attribute version.
        """
        return self._version

    def _is_version(self, value: str) -> bool:
        """
        Private method for checking whether the value version is a valid
        version.

        Args:
            value (str): Value that will be checked whether it is a version.
        """
        try:
            version_indices: List[str] = value.split(".")
            if len(version_indices) != 3:
                return False
            return all(part.isdigit() for part in version_indices)
        except ValueError:
            return False

    @version.setter
    def version(self, value: str) -> None:
        """
        Setter method for the private attribute version. Only accepts versions
        in proper version formatting.

        Args:
            value (str): The value that version has to be set to.
        """
        if self._version is not None:
            raise AttributeError("Artifact class is immutable. version can"
                                 "only be set once.")
        if not self._is_version(value):
            raise ValueError(f"Invalid version format: '{value}'. "
                             "Please use the 'x.y.z' format, e.g., '1.0.0'.")
        self._version = value

    def __str__(self) -> str:
        """
        Returns:
            str: Human readable representation of artifact
        """
        return (f"Artifact(name={self._name}, "
                f"type={self._type}, "
                f"version={self._version}, "
                f"id={self._id}, "
                f"asset_path={self._asset_path}, "
                f"metadata={self._metadata}, "
                f"tags={self._tags})")

    def __repr__(self) -> str:
        """
        Returns:
            str: Reusable string representation of the Artifact object.
        """
        return (f"Artifact(name={self._name}, "
                f"type={self._type}, "
                f"version={self._version}, "
                f"id={self._id}, "
                f"asset_path={self._asset_path}, "
                f"metadata={self._metadata}, "
                f"tags={self._tags}, "
                f"data={self._data})")

    def read(self) -> bytes:
        """
        Returns:
            bytes: the data in bytes.
        """
        return self._data
