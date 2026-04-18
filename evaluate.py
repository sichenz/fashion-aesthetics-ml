# evaluate.py
"""
Evaluate the aesthetic predictor on held-out data.

Usage:
    python evaluate.py
    python evaluate.py --config configs/config.yaml
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.device import get_device, print_device_info
from utils.data_utils import load_config, set_seed, get_predictor_dataloaders
from utils.evaluation_utils import evaluate_predictions, plot_predictions
from models.aesthetic_predictor import AestheticPredictor


def evaluate_model(config_path: str = "configs/config.yaml"):
    """Evaluate the trained aesthetic predictor."""
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    print_device_info()

    device = get_device()
    pred_cfg = config["predictor"]

    # Load data
    _, val_loader = get_predictor_dataloaders(config)

    # Load model
    model = AestheticPredictor(
        clip_model_name=pred_cfg["clip_model"],
        hidden_dims=pred_cfg["hidden_dims"],
        dropout=pred_cfg["dropout"],
        freeze_backbone=True,
    ).to(device)

    ckpt_path = Path(config["paths"]["checkpoints_dir"]) / "best_predictor.pt"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Please train the predictor first: python train_predictor.py")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            scores = batch["score"].to(device)
            preds = model(images)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(scores.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = evaluate_predictions(all_targets, all_preds)

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples evaluated:  {len(all_targets)}")
    print(f"  MAE:                {metrics['mae']:.4f}")
    print(f"  RMSE:               {metrics['rmse']:.4f}")
    print(f"  R² Score:           {metrics['r2']:.4f}")
    print(f"  Pearson Corr:       {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  Spearman Corr:      {metrics['spearman']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"{'='*60}")

    out_dir = Path(config["paths"]["outputs_dir"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_predictions(all_targets, all_preds, str(out_dir / "eval_predictions.png"))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Aesthetic Predictor")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
