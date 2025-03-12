# models/generator.py
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch.nn as nn
import torch.nn.functional as F

class AestheticConditionedGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3-medium",
        lora_rank: int = 16,
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.device = device
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(device)
        
        # Add LoRA adapters to UNet
        self.add_lora_layers()
    
    def add_lora_layers(self):
        """Add LoRA layers to the UNet model"""
        unet = self.pipeline.unet
        lora_attn_procs = {}
        
        # Add LoRA adapters to each attention layer
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None
            if name.endswith("attn1.processor"):
                cross_attention_dim = unet.config.cross_attention_dim
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=unet.config.hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.lora_rank
            )
            
        # Set attention processors
        unet.set_attn_processor(lora_attn_procs)
        
        # Freeze all parameters except LoRA
        for param in self.pipeline.parameters():
            param.requires_grad = False
            
        # Enable training for LoRA parameters
        for param in self.pipeline.unet.parameters():
            if param.ndim == 1:  # Exclude biases and 1D params
                param.requires_grad = False
    
    def get_lora_parameters(self):
        """Get trainable LoRA parameters"""
        for name, param in self.pipeline.unet.named_parameters():
            if param.requires_grad:
                yield param
    
    def save_lora_weights(self, save_path: str):
        """Save LoRA weights"""
        self.pipeline.unet.save_attn_procs(save_path)
    
    def load_lora_weights(self, load_path: str):
        """Load LoRA weights"""
        self.pipeline.unet.load_attn_procs(load_path)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ):
        """Generate images using the pipeline"""
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
        
        return images

# Aesthetic Embedding Conditioning
class AestheticEmbeddingConditioner(nn.Module):
    """Module to condition the diffusion model on aesthetic embeddings"""
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        cross_attention_dim: int = 1024
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cross_attention_dim = cross_attention_dim
        
        # Projection from aesthetic embedding to cross attention dim
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attention_dim)
        )
    
    def forward(self, aesthetic_embedding: torch.Tensor) -> torch.Tensor:
        """Project aesthetic embedding to cross attention space"""
        return self.projector(aesthetic_embedding)