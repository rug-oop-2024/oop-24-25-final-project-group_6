import streamlit as st
from typing import List, Dict

from app.core.system import AutoMLSystem
from app.datasets.management import create, save
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.title("Dataset manager")

datasets_list: List[Dataset] = automl.registry.list(type="dataset")
datasets_dict: Dict = {}
for dataset in datasets_list:
    datasets_dict[dataset.name] = dataset.read()

st.write("Upload a CSV file to view the dataset:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

dataset = None

if uploaded_file is not None:
    dataset = create(uploaded_file)

if isinstance(dataset, Dataset):
    st.write("Download the dataset as CSV:")

    st.download_button(
        label="Download Dataset as CSV",
        data=dataset.data,
        file_name="uploaded_dataset.csv",
        mime="text/csv"
    )

    st.button(
        label="Save Dataset",
        on_click=save,
        args=(dataset,)
    )

st.write("Your datasets")

selected_dataset = st.selectbox("Select a dataset:", options=list(datasets_dict.keys()))

st.subheader("Selected dataset")
st.dataframe(datasets_dict[selected_dataset])
