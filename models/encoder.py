# models/encoder.py
import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Tuple

class FashionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_large",
        pretrained: bool = True,
        embedding_dim: int = 768,
        num_attributes: Dict[str, int] = None  # Dict mapping attribute name to number of classes
    ):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Use average pooling
        )
        
        # Get feature dimension from backbone
        backbone_dim = self.backbone.num_features
        
        # Project to embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )
        
        # Split into mean and variance
        self.embedding_dim = embedding_dim
        
        # Attribute prediction heads (optional)
        self.attribute_heads = nn.ModuleDict()
        if num_attributes:
            for attr_name, num_classes in num_attributes.items():
                self.attribute_heads[attr_name] = nn.Linear(embedding_dim, num_classes)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attributes: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass
        Args:
            x: Input image tensor
            return_attributes: Whether to return attribute predictions
            
        Returns:
            embedding: Sampled embedding
            mu: Mean of embedding distribution
            logvar: Log variance of embedding distribution
            attributes: Dict of attribute predictions (optional)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to embedding distribution parameters
        h = self.projector(features)
        mu, logvar = torch.chunk(h, 2, dim=1)
        
        # Sample embedding using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Predict attributes if requested
        attributes = None
        if return_attributes and self.attribute_heads:
            attributes = {}
            for attr_name, head in self.attribute_heads.items():
                attributes[attr_name] = head(z)
        
        return z, mu, logvar, attributes
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to embedding (deterministic, for inference)"""
        features = self.backbone(x)
        h = self.projector(features)
        mu, _ = torch.chunk(h, 2, dim=1)
        return mu