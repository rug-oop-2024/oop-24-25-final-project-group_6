from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage

from typing import List


class ArtifactRegistry():
    """
    Class for registering and handling artifacts.
    """
    def __init__(
            self,
            database: Database,
            storage: Storage
    ) -> None:
        """
        Initializer method for ArtifactRegistry class.

        Args:
            database (Database): The database of the registry
            storage (Storage): The storage of the registry
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Method used for registering artifacts in the storage and the metadata
        of the artifact in a database

        Args:
            artifact (Artifact): The artifact that has to be registered.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        metadata = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, metadata)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Gets all the selected type artifacts from the artifact registery.

        Args:
            type (str): The type of artifact you want to get.
        Returns:
            List[Artifact]: All the artifacts from the selected artifact type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            if type == "dataset":
                artifact = Dataset(
                    name=data["name"],
                    version=data["version"],
                    asset_path=data["asset_path"],
                    tags=data["tags"],
                    metadata=data["metadata"],
                    data=self._storage.load(data["asset_path"]),
                )
            else:
                artifact = Artifact(
                    name=data["name"],
                    version=data["version"],
                    asset_path=data["asset_path"],
                    tags=data["tags"],
                    metadata=data["metadata"],
                    data=self._storage.load(data["asset_path"]),
                    type=data["type"],
                )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get a specific artifact from the artifact registery using the artifact
        id.

        Args:
            artifact_id (str): The artifact id you are referring to.
        Returns:
            Artifact: The artifact that gets returned from the id.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete a specific artifact using the artifact id.

        Args:
            artifact_id (str): The artifact id you are referring to.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Class for automatically handling machine learning.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the class AutoMLSystem.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Get an instance of AutoMLSystem.

        Returns:
            AutoMLSystem: The automlsystem instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Getter method for the private registry variable.

        Returns:
            ArtifactRegistry: Registry for handling artifacts.
        """
        return self._registry
