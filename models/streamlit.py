# app.py
import streamlit as st
import pickle

# Load the saved model
with open("model_rf_streamlit.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Download Trained RandomForest Model")

# Convert model to bytes for download
with open("model_rf_streamlit.pkl", "rb") as f:
    model_bytes = f.read()

# Download button
st.download_button(
    label="📥 Download Model",
    data=model_bytes,
    file_name="model_rf_streamlit.pkl",
    mime="application/octet-stream"
)
