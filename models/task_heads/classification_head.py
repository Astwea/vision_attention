"""Classification head for image classification tasks."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for image classification."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension from aggregator
            num_classes: Number of classes
            hidden_dim: Hidden dimension for classifier
            dropout: Dropout rate
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Global features (B, input_dim)
            
        Returns:
            Classification logits (B, num_classes)
        """
        return self.classifier(x)

