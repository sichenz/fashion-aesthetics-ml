# models/aesthetic_predictor.py
"""
Aesthetic Score Predictor
Architecture: CLIP ViT-B/32 (frozen) → MLP Head → scalar aesthetic score

Based on the LAION aesthetic predictor approach, but trainable on custom data
so the score reflects YOUR consumer preference signals rather than generic aesthetics.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class AestheticPredictor(nn.Module):
    """
    Predicts aesthetic scores from images using a frozen CLIP backbone
    and a trainable MLP head.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dims: list = None,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Load CLIP vision model
        print(f"Loading CLIP backbone: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_dim = self.clip.visual_projection.in_features  # 768 for ViT-B/32

        # Freeze CLIP backbone
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()
            print("  CLIP backbone frozen.")

        self.freeze_backbone = freeze_backbone

        # Build MLP aesthetic head
        layers = []
        in_dim = self.clip_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features (frozen)."""
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.clip.vision_model(pixel_values=pixel_values)
                # Use the [CLS] token embedding before projection
                features = outputs.pooler_output
        else:
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            features = outputs.pooler_output
        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → aesthetic score.
        Args:
            pixel_values: (B, 3, 224, 224) normalized image tensor
        Returns:
            scores: (B,) predicted aesthetic scores
        """
        features = self.extract_features(pixel_values)
        scores = self.head(features).squeeze(-1)
        return scores

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inference-mode prediction."""
        self.eval()
        with torch.no_grad():
            return self.forward(pixel_values)
