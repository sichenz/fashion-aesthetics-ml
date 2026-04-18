# train_predictor.py
"""
Train the Aesthetic Score Predictor.

Architecture: CLIP ViT-B/32 (frozen) + MLP head
Data: LAION-Aesthetics or Fashion-MNIST fallback
Objective: L1 regression on aesthetic scores (1-10)

Usage:
    python train_predictor.py
    python train_predictor.py --config configs/config.yaml --epochs 30
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.device import get_device, print_device_info
from utils.data_utils import load_config, set_seed, get_predictor_dataloaders
from utils.evaluation_utils import evaluate_predictions, plot_predictions, plot_training_curves
from models.aesthetic_predictor import AestheticPredictor


def train_predictor(config_path: str, epochs_override: int = None, lr_override: float = None):
    """Train the aesthetic score predictor."""
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    print_device_info()

    device = get_device()
    pred_cfg = config["predictor"]

    epochs = epochs_override or pred_cfg["epochs"]
    lr = lr_override or pred_cfg["lr"]

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_predictor_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────────
    model = AestheticPredictor(
        clip_model_name=pred_cfg["clip_model"],
        hidden_dims=pred_cfg["hidden_dims"],
        dropout=pred_cfg["dropout"],
        freeze_backbone=pred_cfg["freeze_backbone"],
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=pred_cfg["weight_decay"],
    )

    warmup_epochs = pred_cfg.get("warmup_epochs", 3)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.SmoothL1Loss()

    # ── Training Loop ─────────────────────────────────────────────────────
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(config["paths"]["outputs_dir"]) / "predictor"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_path = ckpt_dir / "best_predictor.pt"
    train_losses, val_losses, val_metrics_list = [], [], []

    print(f"\n{'='*60}")
    print(f"  Training Aesthetic Predictor — {epochs} epochs")
    print(f"  LR: {lr}, Batch: {pred_cfg['batch_size']}, Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        # Keep CLIP backbone in eval mode if frozen
        if pred_cfg["freeze_backbone"]:
            model.clip.eval()

        running_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images = batch["image"].to(device)
            scores = batch["score"].to(device)

            preds = model(images)
            loss = criterion(preds, scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images = batch["image"].to(device)
                scores = batch["score"].to(device)

                preds = model(images)
                loss = criterion(preds, scores)

                val_loss_sum += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(scores.cpu().numpy())

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = evaluate_predictions(all_targets, all_preds)
        val_metrics_list.append(metrics)

        scheduler.step()

        elapsed = time.time() - t0
        improved = "★" if metrics["mae"] < best_val_mae else " "
        print(
            f"  Epoch {epoch+1:3d}/{epochs} │ "
            f"Train Loss: {train_loss:.4f} │ "
            f"Val MAE: {metrics['mae']:.4f} │ "
            f"R²: {metrics['r2']:.4f} │ "
            f"Pearson: {metrics['pearson']:.4f} │ "
            f"LR: {scheduler.get_last_lr()[0]:.2e} │ "
            f"{elapsed:.1f}s {improved}"
        )

        # Save best model
        if metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": best_val_mae,
                "metrics": metrics,
                "config": config,
            }, best_path)

    # ── Final Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val MAE: {best_val_mae:.4f}")
    print(f"  Model saved to: {best_path}")
    print(f"{'='*60}")

    # Save plots
    plot_predictions(all_targets, all_preds, str(out_dir / "predictions.png"))
    plot_training_curves(train_losses, val_losses, val_metrics_list, str(out_dir / "training_curves.png"))

    return str(best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aesthetic Score Predictor")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    train_predictor(args.config, args.epochs, args.lr)
