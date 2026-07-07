"""ANN classifier helpers."""

import torch
from torch import nn


class SLIDNet(nn.Module):
    """Simple feed-forward network for SLID features."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def save_ann(model, path):
    """Save ANN weights."""
    torch.save(model.state_dict(), path)
