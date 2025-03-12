# train_generator.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from diffusers import DDPMScheduler, StableDiffusionPipeline

from models.encoder import FashionEncoder
from models.predictor import AestheticPredictor
from models.generator import AestheticConditionedGenerator
from utils.data_utils import load_config, get_dataloaders, set_seed

def train_generator(config_path, encoder_path=None, predictor_path=None):
    # Load config
    config = load_config(config_path)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['generator']['gradient_accumulation_steps'],
        mixed_precision="fp16"
    )
    
    # Set seed
    accelerate_set_seed(config['project']['seed'])
    
    # Load encoder
    encoder = FashionEncoder(
        model_name=config['encoder']['model_name'],
        pretrained=False,
        embedding_dim=config['encoder']['embedding_dim']
    )
    
    # Load encoder weights
    if encoder_path:
        encoder.load_state_dict(torch.load(encoder_path))
    else:
        # Try to load from default path
        default_path = os.path.join(config['paths']['checkpoints'], 'best_encoder.pt')
        if os.path.exists(default_path):
            encoder.load_state_dict(torch.load(default_path))
        else:
            raise ValueError(f"Encoder weights not found at {default_path}")
    
    # Load predictor
    predictor = AestheticPredictor(
        embedding_dim=config['encoder']['embedding_dim'],
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    )
    
    # Load predictor weights
    if predictor_path:
        predictor.load_state_dict(torch.load(predictor_path))
    else:
        # Try to load from default path
        default_path = os.path.join(config['paths']['checkpoints'], 'best_predictor.pt')
        if os.path.exists(default_path):
            predictor.load_state_dict(torch.load(default_path))
        else:
            raise ValueError(f"Predictor weights not found at {default_path}")
    
    # Freeze encoder and predictor
    for param in encoder.parameters():
        param.requires_grad = False
    
    for param in predictor.parameters():
        param.requires_grad = False
    
    # Initialize generator
    generator = AestheticConditionedGenerator(
        model_id=config['generator']['model_id'],
        lora_rank=config['generator']['lora_rank']
    )
    
    # Get dataloaders
    train_loader, _ = get_dataloaders(config)
    
    # Prepare models for training
    encoder, predictor = accelerator.prepare(encoder, predictor)
    
    # Set up optimizer
    optimizer = AdamW(
        generator.get_lora_parameters(),
        lr=config['generator']['learning_rate']
    )
    
    # Prepare optimizer
    optimizer = accelerator.prepare(optimizer)
    
    # Get noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['generator']['model_id'],
        subfolder="scheduler"
    )
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(0, config['generator']['max_train_steps']),
        desc="Training generator"
    )
    
    # Save directory
    save_dir = os.path.join(config['paths']['checkpoints'], 'generator_lora')
    os.makedirs(save_dir, exist_ok=True)
    
    # Set models to train/eval mode
    generator.pipeline.unet.train()
    encoder.eval()
    predictor.eval()
    
    # Start training
    while global_step < config['generator']['max_train_steps']:
        for batch in train_loader:
            # Skip step if we've reached max training steps
            if global_step >= config['generator']['max_train_steps']:
                break
            
            # Get images and ratings
            images = batch['image'].to(accelerator.device)
            ratings = batch['rating'].to(accelerator.device)
            
            # Convert images to latents
            with torch.no_grad():
                # Get aesthetic embeddings
                embeddings, _, _, _ = encoder(images)
                
                # Get aesthetic scores
                aesthetic_scores = predictor(embeddings)
                
                # Encode images to latents
                latents = generator.pipeline.vae.encode(
                    images.to(dtype=generator.pipeline.vae.dtype)
                ).latent_dist.sample() * generator.pipeline.vae.config.scaling_factor
            
            # Set prompt embedding based on aesthetic score
            batch_size = images.shape[0]
            
            # Create random noise
            noise = torch.randn_like(latents)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Construct prompt embeddings
            prompts = [f"A fashion item with aesthetic rating {score.item():.1f}" for score in aesthetic_scores]
            
            # Get text embeddings
            with torch.no_grad():
                text_inputs = generator.pipeline.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=generator.pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(accelerator.device)
                
                text_embeddings = generator.pipeline.text_encoder(text_inputs.input_ids)[0]
            
            # Get unconditional embeddings for classifier-free guidance
            with torch.no_grad():
                uncond_tokens = [""] * batch_size
                uncond_input = generator.pipeline.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=text_inputs.input_ids.shape[1],
                    return_tensors="pt"
                ).to(accelerator.device)
                
                uncond_embeddings = generator.pipeline.text_encoder(uncond_input.input_ids)[0]
            
            # Concatenate conditional and unconditional embeddings
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Forward pass through UNet
            with accelerator.accumulate(generator.pipeline.unet):
                # Get model prediction
                model_pred = generator.pipeline.unet(
                    noisy_latents,
                    timesteps,
                    text_embeddings
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Log info
            progress_bar.set_postfix(loss=loss.detach().item())
            
            # Save checkpoint
            if global_step % 500 == 0:
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(generator.pipeline.unet)
                unwrapped_unet.save_attn_procs(
                    os.path.join(save_dir, f"step_{global_step}")
                )
    
    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(generator.pipeline.unet)
    unwrapped_unet.save_attn_procs(
        os.path.join(save_dir, "final")
    )
    
    print("Generator training completed!")
    return os.path.join(save_dir, "final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to encoder weights")
    parser.add_argument("--predictor_path", type=str, default=None, help="Path to predictor weights")
    args = parser.parse_args()
    
    train_generator(args.config, args.encoder_path, args.predictor_path)