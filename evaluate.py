# evaluate.py
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

from models.encoder import FashionEncoder
from models.predictor import AestheticPredictor
from utils.data_utils import load_config, get_dataloaders, FashionDataset, get_transforms, set_seed
from utils.evaluation_utils import evaluate_predictions, predict_batch

def evaluate_model(config_path):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test dataloader
    _, val_transform = get_transforms(config['data']['image_size'])
    
    test_dataset = FashionDataset(
        config['paths']['processed_data'],
        split='test',
        transform=val_transform,
        labeled=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['predictor']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
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
    
    # Set models to eval mode
    encoder.eval()
    predictor.eval()
    
    # Predict on test set
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = batch['image'].to(device)
            ratings = batch['rating'].numpy()
            
            # Get embeddings from encoder
            embeddings, _, _, _ = encoder(images)
            
            # Forward pass through predictor
            predictions = predictor(embeddings).cpu().numpy()
            
            # Save predictions and targets
            all_preds.append(predictions)
            all_targets.append(ratings)
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    metrics = evaluate_predictions(all_targets, all_preds)
    
    # Print metrics
    print("=== Evaluation Results ===")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Pearson Correlation: {metrics['pearson']:.4f}")
    
    # Plot predicted vs actual ratings
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Predicted vs Actual Aesthetic Ratings")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.grid(True)
    
    # Save figure
    output_dir = os.path.join(config['paths']['processed_data'], 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "predicted_vs_actual.png"))
    
    # Plot error distribution
    errors = all_preds - all_targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'actual': all_targets,
        'predicted': all_preds,
        'error': errors
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Generate report
    report = {
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'pearson': metrics['pearson'],
        'num_samples': len(all_targets)
    }
    
    # Save report
    pd.DataFrame([report]).to_csv(os.path.join(output_dir, "metrics_report.csv"), index=False)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    evaluate_model(args.config)