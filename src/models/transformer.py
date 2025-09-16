"""
Transformer model for ECG classification
"""
import math
import torch
import torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Positional encoding for transformer
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        Args:
            x: Input tensor (B, L, D)
        Returns:
            Output tensor with positional encoding
        """
        return x + self.pe[:x.size(1)]

class ECGTransformer(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, num_classes: int = 2, dropout: float = 0.1,
                 output_length: int = 20):
        """
        Transformer model for ECG classification
        Args:
            input_dim: Input dimension (number of channels)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout probability
            output_length: Length of output sequence
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to match desired sequence length
        self.output_pooling = nn.AdaptiveAvgPool1d(output_length)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor (B, 1, T)
            src_mask: Source mask for transformer
        Returns:
            Output tensor (B, output_length, num_classes)
        """
        # Reshape input and project to model dimension
        x = x.transpose(1, 2)  # (B, T, 1)
        x = self.input_projection(x)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer_encoder(x, src_mask)  # (B, T, d_model)
        
        # Pool to desired output length
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.output_pooling(x)  # (B, d_model, output_length)
        x = x.transpose(1, 2)  # (B, output_length, d_model)
        
        # Apply classification layer
        x = self.classifier(x)  # (B, output_length, num_classes)
        
        return x