# train_predictor.py
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.encoder import FashionEncoder
from models.predictor import AestheticPredictor
from utils.data_utils import load_config, get_dataloaders, set_seed
from utils.training_utils import train_one_epoch, validate, get_lr_scheduler
from utils.evaluation_utils import evaluate_predictions

def train_predictor(config_path, encoder_path=None):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Initialize encoder
    encoder = FashionEncoder(
        model_name=config['encoder']['model_name'],
        pretrained=False,  # We'll load weights
        embedding_dim=config['encoder']['embedding_dim']
    ).to(device)
    
    # Load encoder weights
    if encoder_path:
        encoder.load_state_dict(torch.load(encoder_path))
    else:
        # Try to load from default path
        default_path = os.path.join(config['paths']['checkpoints'], 'best_encoder.pt')
        if os.path.exists(default_path):
            encoder.load_state_dict(torch.load(default_path))
        else:
            raise ValueError(f"Encoder weights not found at {default_path}. Please train encoder first.")
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Initialize predictor
    predictor = AestheticPredictor(
        embedding_dim=config['encoder']['embedding_dim'],
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)
    
    # Define optimizer
    optimizer = AdamW(
        predictor.parameters(),
        lr=config['predictor']['lr'],
        weight_decay=0.01
    )
    
    # Define loss function
    criterion = nn.L1Loss()  # Mean Absolute Error
    
    # Training loop
    best_val_mae = float('inf')
    best_model_path = os.path.join(config['paths']['checkpoints'], 'best_predictor.pt')
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    for epoch in range(config['predictor']['epochs']):
        print(f"Epoch {epoch + 1}/{config['predictor']['epochs']}")
        
        # Train
        predictor.train()
        train_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to device
            images = batch['image'].to(device)
            ratings = batch['rating'].to(device)
            
            # Get embeddings from encoder
            with torch.no_grad():
                embeddings, _, _, _ = encoder(images)
            
            # Forward pass through predictor
            predictions = predictor(embeddings)
            
            # Compute loss
            loss = criterion(predictions, ratings)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            all_preds.append(predictions.detach().cpu().numpy())
            all_targets.append(ratings.cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Validation
        predictor.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move data to device
                images = batch['image'].to(device)
                ratings = batch['rating'].to(device)
                
                # Get embeddings from encoder
                embeddings, _, _, _ = encoder(images)
                
                # Forward pass through predictor
                predictions = predictor(embeddings)
                
                # Compute loss
                loss = criterion(predictions, ratings)
                
                # Track metrics
                val_loss += loss.item()
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(ratings.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        train_metrics = evaluate_predictions(
            np.concatenate(all_targets),
            np.concatenate(all_preds)
        )
        
        val_metrics = evaluate_predictions(
            np.concatenate(val_targets),
            np.concatenate(val_preds)
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save(predictor.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    print("Training completed!")
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to encoder weights")
    args = parser.parse_args()
    
    train_predictor(args.config, args.encoder_path)