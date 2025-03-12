# models/predictor.py
import torch
import torch.nn as nn

class AestheticPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        # Final prediction layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.predictor(x).squeeze(-1)