# inference.py
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from models.encoder import FashionEncoder
from models.predictor import AestheticPredictor
from models.generator import AestheticConditionedGenerator
from utils.data_utils import load_config, get_transforms, set_seed

def predict_aesthetic_score(encoder, predictor, image_path, device):
    """Predict aesthetic score for an image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform image
    _, transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get embedding
    with torch.no_grad():
        embedding, _, _, _ = encoder(image_tensor)
        score = predictor(embedding).item()
    
    return score

def generate_fashion_designs(config, num_samples=5, target_rating=4.5, style=None, color=None):
    """Generate fashion designs with specified attributes"""
    # Load config
    config = load_config(config)
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load encoder
    encoder = FashionEncoder(
        model_name=config['encoder']['model_name'],
        pretrained=False,
        embedding_dim=config['encoder']['embedding_dim']
    ).to(device)
    
    # Load encoder weights
    encoder_path = os.path.join(config['paths']['checkpoints'], 'best_encoder.pt')
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path))
    else:
        raise ValueError(f"Encoder weights not found at {encoder_path}")
    
    # Load predictor
    predictor = AestheticPredictor(
        embedding_dim=config['encoder']['embedding_dim'],
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)
    
    # Load predictor weights
    predictor_path = os.path.join(config['paths']['checkpoints'], 'best_predictor.pt')
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
    else:
        raise ValueError(f"Predictor weights not found at {predictor_path}")
    
    # Load generator
    generator = AestheticConditionedGenerator(
        model_id=config['generator']['model_id'],
        lora_rank=config['generator']['lora_rank'],
        device=device
    )
    
    # Load LoRA weights
    lora_path = os.path.join(config['paths']['checkpoints'], 'generator_lora', 'final')
    if os.path.exists(lora_path):
        generator.load_lora_weights(lora_path)
    else:
        print(f"Warning: LoRA weights not found at {lora_path}, using base model")
    
    # Set encoder and predictor to eval mode
    encoder.eval()
    predictor.eval()
    
    # Build prompt
    prompt_parts = []
    prompt_parts.append(f"A high-quality fashion item")
    
    if style:
        prompt_parts.append(f"in {style} style")
    
    if color:
        prompt_parts.append(f"in {color} color")
    
    prompt_parts.append(f"with aesthetic rating of {target_rating:.1f}")
    
    prompt = " ".join(prompt_parts)
    
    # Generate images
    images = generator.generate(
        prompt=prompt,
        negative_prompt="low quality, bad design, poor lighting, watermark, logo, text",
        num_images=num_samples,
        guidance_scale=config['inference']['guidance_scale']
    )
    
    # Create output directory
    output_dir = os.path.join(config['paths']['processed_data'], 'generated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images and predict scores
    saved_paths = []
    scores = []
    
    for i, img in enumerate(images):
        # Save image
        img_path = os.path.join(output_dir, f"generated_{i}.jpg")
        img.save(img_path)
        saved_paths.append(img_path)
        
        # Predict score
        score = predict_aesthetic_score(encoder, predictor, img_path, device)
        scores.append(score)
    
    # Display results
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i, (img_path, score) in enumerate(zip(saved_paths, scores)):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Score: {score:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"generated_grid.jpg"))
    plt.show()
    
    return saved_paths, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--target_rating", type=float, default=4.5, help="Target aesthetic rating")
    parser.add_argument("--style", type=str, default=None, help="Style attribute (e.g., casual, formal)")
    parser.add_argument("--color", type=str, default=None, help="Color attribute")
    args = parser.parse_args()
    
    generate_fashion_designs(
        args.config,
        num_samples=args.num_samples,
        target_rating=args.target_rating,
        style=args.style,
        color=args.color
    )

if __name__ == "__main__":
    main()