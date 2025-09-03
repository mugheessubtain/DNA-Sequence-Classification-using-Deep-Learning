# 🧬 DNA Sequence Classifier (AI + Bioinformatics)

This project applies **Deep Learning (PyTorch)** to classify DNA sequences into categories (e.g., disease-related vs. normal genes). It demonstrates how **AI can be applied in Bioinformatics and Genomics**.

## 🚀 Features
- DNA sequence preprocessing & encoding
- PyTorch deep learning classifier
- Training & evaluation pipeline
- Streamlit UI for predictions
- Extendable to larger genomic datasets

## ⚙️ Tech Stack
- Python, PyTorch, Scikit-learn
- Pandas, NumPy, Matplotlib
- Streamlit (UI)

## 📂 Project Structure

dna-sequence-classifier/ │── app.py │── train.py │── model.py │── utils.py │── requirements.txt │── README.md │── data/dna_sequences.csv

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py

🎯 Future Work

Use larger genomic datasets (Kaggle/UCI Bioinformatics repo)

Try advanced models (LSTMs, Transformers, DNA-BERT)
