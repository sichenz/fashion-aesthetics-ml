# utils/evaluation_utils.py
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions using various metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    return {
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }

def predict_batch(model, dataloader, device):
    """Make predictions for a batch of data"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Save predictions
            all_preds.append(outputs.cpu().numpy())
            
            if 'rating' in batch:
                all_targets.append(batch['rating'].numpy())
    
    # Concatenate predictions
    all_preds = np.concatenate(all_preds)
    
    if all_targets:
        all_targets = np.concatenate(all_targets)
        return all_preds, all_targets
    
    return all_preds