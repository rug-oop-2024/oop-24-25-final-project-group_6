from autoop.core.ml.model.model import Model
from autoop.core.ml.dataset import Dataset
from autoop.functional.preprocessing import preprocess_features
from autoop.functional.feature import detect_feature_types
import streamlit as st
import pandas as pd


def predict(model: Model, dataset: Dataset):
    try:
        data = preprocess_features(detect_feature_types(dataset), dataset)
        for d in data:
            if d[2]["type"] == "OneHot"
        st.write(data)
        model.predict(dataset.read())
    except Exception as e:
        st.error(e)

    st.dataframe(dataset.read())
