import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_dataset
from model import DNAClassifier

# Load dataset
X, y = load_dataset("data/dna_sequences.csv", max_len=100)
X = torch.tensor(X)
y = torch.tensor(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model, loss, optimizer
model = DNAClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.2f}")

# Save model
torch.save(model.state_dict(), "dna_model.pth")
print("✅ Model saved as dna_model.pth")
