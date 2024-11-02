import streamlit as st
from typing import Callable

from app.core.system import AutoMLSystem
from app.datasets.management import create, save
from autoop.core.ml.dataset import Dataset