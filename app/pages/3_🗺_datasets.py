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
    dataset_contents[dataset.name] = dataset

st.write("Upload a CSV file to view the dataset:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

dataset = None

if uploaded_file is not None:
    dataset = create(uploaded_file)
    st.success("File uploaded succesfully!")

if isinstance(dataset, Dataset):
    st.subheader("Save or Download dataset")
    st.markdown('<h5>Download Dataset</h5>', unsafe_allow_html=True)

    download = st.download_button(
        label="Download Dataset as CSV",
        data=dataset.data,
        file_name="uploaded_dataset.csv",
        mime="text/csv"
    )

    if download:
        st.success("Dataset downloaded successfully!")

    st.markdown('<h5>Save Dataset</h5>', unsafe_allow_html=True)

    if st.button("Save Dataset"):
        save(dataset)
        st.session_state.is_saved = True
        st.rerun()

    if "is_saved" in st.session_state:
        if st.session_state.is_saved:
            st.success("Dataset saved successfully!")
else:
    st.session_state.is_saved = False

st.subheader("Your datasets")
st.write("Select a dataset from your saved datasets to view it.")

selected_dataset_name = st.selectbox(
    "Select a dataset:",
    options=list(dataset_contents.keys())
)

if selected_dataset_name is not None:
    st.subheader(f"Dataset: {selected_dataset_name}")

    selected_dataset = dataset_contents[selected_dataset_name]
    st.dataframe(selected_dataset.read())

    delete_button = st.button("Delete dataset")

    if delete_button:
        automl.registry.delete(selected_dataset.id)
else:
    st.write("No dataset selected or available.")
