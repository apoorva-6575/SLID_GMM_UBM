"""
Artificial Neural Network (ANN) Classifier Module.

This module implements a PyTorch Multi-Layer Perceptron (MLP) that takes
the score vectors (log-likelihoods) from the Language-specific GMMs and
classifies the audio into the final spoken language.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class LanguageANN(nn.Module):
    """
    PyTorch Neural Network for Language Classification based on GMM scores.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.2):
        """
        Initialize the ANN architecture.

        Parameters
        ----------
        input_dim : int
            Number of input features (equal to the number of language GMMs).
        num_classes : int
            Number of output classes (languages).
        hidden_dims : list or tuple
            List of integers defining the size of each hidden layer.
        dropout : float
            Dropout probability to prevent overfitting.
        """
        super(LanguageANN, self).__init__()
        
        layers = []
        
        # Build hidden layers dynamically
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
            These are the GMM log-likelihood scores.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch_size, num_classes).
            (Note: Softmax is applied internally by CrossEntropyLoss during training).
        """
        return self.network(x)


def train_ann(model, train_loader, epochs=50, learning_rate=0.001):
    """
    Standard PyTorch training loop for the ANN.

    Parameters
    ----------
    model : LanguageANN
        The neural network instance.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of (gmm_scores, language_labels).
    epochs : int
        Number of times to iterate over the entire dataset.
    learning_rate : float
        Learning rate for the Adam optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    print(f"Training ANN for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass (predictions)
            outputs = model(inputs)
            
            # 3. Calculate loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass (calculate gradients)
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
    print("ANN training complete.")
    return model


def save_ann(model, filepath):
    """Save the PyTorch model weights."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"ANN saved to {path}")


def load_ann(model, filepath):
    """Load PyTorch model weights into an existing architecture."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
        
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    return model
