import torch
import torch.nn as nn

class DNAClassifier(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, num_classes=2):
        super(DNAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        return self.fc2(x)
      
