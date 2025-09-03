import numpy as np
import pandas as pd

# Encode DNA sequence into integers (A=0, C=1, G=2, T=3)
def encode_sequence(seq, max_len=100):
    mapping = {"A":0, "C":1, "G":2, "T":3}
    encoded = [mapping.get(c, 0) for c in seq]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # padding
    return encoded[:max_len]

# Load dataset from CSV
def load_dataset(path="data/dna_sequences.csv", max_len=100):
    df = pd.read_csv(path)
    X = [encode_sequence(seq, max_len) for seq in df["sequence"]]
    y = df["label"].values
    return np.array(X), y
