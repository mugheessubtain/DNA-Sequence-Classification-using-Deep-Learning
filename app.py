import streamlit as st
import torch
from model import DNAClassifier
from utils import encode_sequence

# Load trained model
model = DNAClassifier()
model.load_state_dict(torch.load("dna_model.pth"))
model.eval()

st.set_page_config(page_title="DNA Classifier", page_icon="🧬", layout="centered")
st.title("🧬 DNA Sequence Classifier")

seq_input = st.text_area("Enter DNA sequence (A, C, G, T only):")

if st.button("Predict"):
    if seq_input:
        encoded = encode_sequence(seq_input, max_len=100)
        X = torch.tensor([encoded])
        with torch.no_grad():
            output = model(X)
            pred = torch.argmax(output, dim=1).item()
            label = "Disease-Related Gene" if pred == 1 else "Normal Gene"
            st.success(f"Prediction: {label}")
    else:
        st.error("Please enter a DNA sequence.")
