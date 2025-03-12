# train_encoder.py
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.encoder import FashionEncoder
from utils.data_utils import load_config, get_dataloaders, get_unlabeled_dataloader, set_seed
from utils.training_utils import train_one_epoch, validate, get_lr_scheduler

def train_encoder(config_path):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)
    unlabeled_loader = get_unlabeled_dataloader(config)
    
    # Initialize model
    model = FashionEncoder(
        model_name=config['encoder']['model_name'],
        pretrained=config['encoder']['pretrained'],
        embedding_dim=config['encoder']['embedding_dim']
    ).to(device)
    
    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['encoder']['lr'],
        weight_decay=0.01
    )
    
    # Define scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Define loss functions
    reconstruction_criterion = nn.MSELoss()
    
    # KL divergence loss
    def kl_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(config['paths']['checkpoints'], 'best_encoder.pt')
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    for epoch in range(config['encoder']['epochs']):
        print(f"Epoch {epoch + 1}/{config['encoder']['epochs']}")
        
        # Train on labeled data
        model.train()
        labeled_loss = 0
        
        for batch in tqdm(train_loader, desc="Training on labeled data"):
            # Move data to device
            images = batch['image'].to(device)
            
            # Forward pass
            z, mu, logvar, _ = model(images)
            
            # Compute losses
            kl = kl_loss(mu, logvar)
            
            # Total loss
            loss = kl
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            labeled_loss += loss.item()
        
        labeled_loss /= len(train_loader)
        
        # Train on unlabeled data
        model.train()
        unlabeled_loss = 0
        
        for batch in tqdm(unlabeled_loader, desc="Training on unlabeled data"):
            # Move data to device
            images = batch['image'].to(device)
            
            # Forward pass
            z, mu, logvar, _ = model(images)
            
            # Compute losses
            kl = kl_loss(mu, logvar)
            
            # Total loss
            loss = kl
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            unlabeled_loss += loss.item()
        
        unlabeled_loss /= len(unlabeled_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move data to device
                images = batch['image'].to(device)
                
                # Forward pass
                z, mu, logvar, _ = model(images)
                
                # Compute losses
                kl = kl_loss(mu, logvar)
                
                # Total loss
                loss = kl
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Print metrics
        print(f"Labeled Loss: {labeled_loss:.4f}, Unlabeled Loss: {unlabeled_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    print("Training completed!")
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_encoder(args.config)