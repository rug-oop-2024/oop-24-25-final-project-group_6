from autoop.core.ml.model.model import Model
import streamlit as st
import pandas as pd


def predict(model: Model, dataframe: pd.DataFrame):
    try:
        model.predict(dataframe.to_numpy())
    except Exception as e:
        st.error(e)

    st.dataframe(dataframe)
