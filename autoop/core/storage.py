from abc import ABC, abstractmethod
from glob import glob
import os
from typing import List


class NotFoundError(Exception):
    """
    Custom error class that is called when a path is not found
    """
    def __init__(self, path: str) -> None:
        """
        Initialize NotFoundError class
        Args:
            path (str): The path of the OS
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract or interface class for classes that store data on a framework.
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Local storage class which stores data within the project of an operation
    system.
    """
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initalize LocalStorage class
        Args:
            base_path: The OS path where the data is stored
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data given a key.
        Args:
            data (bytes): The data to save
            key (str): The dictonary and file to save the data in.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data given a key.
        Args:
            key (str): The dictonary and file to save the data in.
        Returns:
            bytes: The data that is loaded.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data given a key.
        Args:
            key (str): The dictonary and file to save the data in.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all paths under a given prefix.
        Args:
            prefix (str): Prefix to list.
        Returns:
            List[str]: The paths returned under a given prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys
                if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Private method to look whether the path is exists
        Args:
            prefix (str): The path to the file.
        Raises:
            NotFoundError
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Privated method that joins a base_path and a path together.
        Args:
            path (str): A path to a file.
        Returns:
            str: The joined path
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
