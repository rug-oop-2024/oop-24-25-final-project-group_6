import streamlit as st
from typing import List, Dict

from app.core.system import AutoMLSystem
from app.datasets.management import create, save
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.title("Dataset manager")

available_datasets: List[Dataset] = automl.registry.list(type="dataset")
dataset_contents: Dict = {}
for dataset in available_datasets:
    dataset_contents[dataset.name] = dataset.read()

st.write("Upload a CSV file to view the dataset:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

dataset = None

if uploaded_file is not None:
    dataset = create(uploaded_file)
    st.success("File uploaded succesfully!")

if isinstance(dataset, Dataset):
    st.write("Download or Save Dataset")

    st.download_button(
        label="Download Dataset as CSV",
        data=dataset.data,
        file_name="uploaded_dataset.csv",
        mime="text/csv"
    )

    if st.button("Save Dataset"):
        save(dataset)
        st.success("Dataset saved successfully!")

st.write("Your datasets")
st.write("Select a dataset from your saved datasets to view it.")

selected_dataset = st.selectbox("Select a dataset:",
                                options=list(dataset_contents.keys()))


if selected_dataset is not None:
    st.subheader(f"Dataset: {selected_dataset}")
    st.dataframe(dataset_contents[selected_dataset])
else:
    st.write("No dataset selected or available.")
