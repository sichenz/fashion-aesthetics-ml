# utils/evaluation_utils.py
"""Evaluation metrics and visualization for aesthetic prediction and generation."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def evaluate_predictions(y_true, y_pred):
    """Compute all evaluation metrics for aesthetic score predictions."""
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    pearson, p_pear = pearsonr(y_true, y_pred)
    spearman, p_spear = spearmanr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "pearson": pearson, "pearson_p": p_pear,
            "spearman": spearman, "spearman_p": p_spear}


def plot_predictions(y_true, y_pred, save_path, title="Predicted vs Actual Aesthetic Scores"):
    """Scatter plot of predicted vs actual scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, c="#6366f1")
    mn, mx = min(y_true.min(), y_pred.min())-0.5, max(y_true.max(), y_pred.max())+0.5
    ax.plot([mn,mx],[mn,mx],"r--",lw=1.5,label="Perfect prediction")
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    # Error histogram
    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=40, alpha=0.7, color="#8b5cf6", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Prediction Error"); ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved prediction plot to {save_path}")


def plot_training_curves(train_losses, val_losses, val_metrics, save_path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses)+1)
    axes[0].plot(epochs, train_losses, "b-", label="Train"); axes[0].plot(epochs, val_losses, "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if val_metrics:
        maes = [m["mae"] for m in val_metrics]; r2s = [m["r2"] for m in val_metrics]
        axes[1].plot(epochs[:len(maes)], maes, "g-"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE"); axes[1].set_title("Validation MAE"); axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs[:len(r2s)], r2s, "m-"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("R²"); axes[2].set_title("Validation R²"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved training curves to {save_path}")


def plot_generated_gallery(images, scores, save_path, title="Generated Designs"):
    """Create a gallery grid of generated images with their predicted scores."""
    n = len(images)
    cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4.5))
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[np.newaxis, :]
    elif cols == 1: axes = axes[:, np.newaxis]
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        if i < n:
            ax.imshow(images[i]); ax.set_title(f"Score: {scores[i]:.2f}", fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved generated gallery to {save_path}")
