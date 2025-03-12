import torch
import numpy as np
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        images = batch['image'].to(device)
        ratings = batch['rating'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, ratings)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch['image'].to(device)
            ratings = batch['rating'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, ratings)
            
            # Save predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(ratings.cpu().numpy())
            
            # Update total loss
            total_loss += loss.item()
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return total_loss / len(dataloader), mae

def get_lr_scheduler(optimizer, config, num_training_steps=None):
    """Get learning rate scheduler"""
    scheduler_type = config['encoder']['scheduler']
    
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['encoder']['epochs'], 
            eta_min=1e-6
        )
    elif scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['encoder']['epochs']
        )
    else:
        scheduler = None
        
    return scheduler