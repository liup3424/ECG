import torch
import torch.nn as nn
import copy
from typing import Optional
from torch.utils.data import DataLoader

class ECGNet(nn.Module):
    def __init__(self, num_classes: int = 2, output_sec: int = 20):
        """
        Initialize ECG classification network
        Args:
            num_classes: Number of output classes
            output_sec: Number of output time steps
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
        )
        self.block3 = nn.Sequential(
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_sec)
        self.fc = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 1, T)
        Returns:
            Output tensor of shape (B, output_sec, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.temporal_pool(x)
        x = self.fc(x)
        return x.permute(0, 2, 1)

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                epochs: int = 100,
                early_stop_rounds: int = 10,
                model_path: Optional[str] = None,
                device: Optional[torch.device] = None) -> nn.Module:
    """
    Train the model
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        epochs: Number of epochs
        early_stop_rounds: Number of rounds for early stopping
        model_path: Path to save the best model
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred.reshape(-1, pred.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred.reshape(-1, pred.size(-1)), yb.reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if model_path:
                torch.save(best_model_wts, model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_rounds:
                print("Early stopping.")
                break

    model.load_state_dict(best_model_wts)
    return model