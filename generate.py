# generate.py
"""
Generate novel fashion designs and screen them with the aesthetic predictor.

This is the closed-loop system:
1. Generate candidate designs at a target aesthetic score
2. Score each with the trained predictor
3. Rank and display results

Usage:
    python generate.py
    python generate.py --target_score 8.0 --num_samples 8 --prompt "elegant evening dress"
"""
import argparse
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from utils.device import get_device, print_device_info
from utils.data_utils import load_config, set_seed, get_predictor_transforms
from utils.evaluation_utils import plot_generated_gallery
from models.aesthetic_predictor import AestheticPredictor
from models.aesthetic_generator import AestheticGenerator


def load_predictor(config, device):
    """Load the trained aesthetic predictor."""
    pred_cfg = config["predictor"]
    model = AestheticPredictor(
        clip_model_name=pred_cfg["clip_model"],
        hidden_dims=pred_cfg["hidden_dims"],
        dropout=pred_cfg["dropout"],
        freeze_backbone=True,
    ).to(device)

    ckpt_path = Path(config["paths"]["checkpoints_dir"]) / "best_predictor.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded predictor from {ckpt_path} (val MAE: {ckpt['val_mae']:.4f})")
    else:
        print(f"WARNING: No predictor checkpoint found at {ckpt_path}")
        print("  Predictions will use an untrained model.")
    model.eval()
    return model


def score_images(predictor, images, device):
    """Score a list of PIL images with the aesthetic predictor."""
    _, val_tf = get_predictor_transforms(224)
    scores = []
    for img in images:
        tensor = val_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = predictor(tensor).item()
        scores.append(score)
    return scores


def generate_and_screen(
    config_path="configs/config.yaml",
    prompt=None,
    target_score=None,
    num_samples=None,
    guidance_scale=None,
    seed=None,
):
    """Generate designs, score them, and rank by aesthetic quality."""
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    print_device_info()

    device = get_device()
    inf_cfg = config["inference"]

    target = target_score or inf_cfg["target_aesthetic_score"]
    n = num_samples or inf_cfg["num_samples"]
    gs = guidance_scale or inf_cfg["guidance_scale"]
    neg = inf_cfg["negative_prompt"]
    steps = inf_cfg["num_inference_steps"]

    # Build prompt with aesthetic conditioning
    base = prompt or "a high quality fashion product photograph, studio lighting"
    from utils.data_utils import _score_to_label
    label = _score_to_label(target)
    full_prompt = f"{base}, {label} aesthetic quality"

    print(f"\n{'='*60}")
    print(f"  Generating {n} designs")
    print(f"  Prompt: {full_prompt}")
    print(f"  Target score: {target:.1f}")
    print(f"  Guidance scale: {gs}")
    print(f"{'='*60}\n")

    # Load models
    print("Loading aesthetic predictor...")
    predictor = load_predictor(config, device)

    print("Loading design generator...")
    generator = AestheticGenerator(config, device)

    # Try to load fine-tuned LoRA weights
    lora_path = Path(config["paths"]["checkpoints_dir"]) / "generator_lora" / "final"
    if lora_path.exists():
        generator.load_lora(str(lora_path))
    else:
        print("  No fine-tuned LoRA found — using base SD 1.5 model.")

    # Generate
    print(f"\nGenerating {n} images...")
    images = generator.generate(
        prompt=full_prompt,
        negative_prompt=neg,
        num_images=n,
        guidance_scale=gs,
        num_steps=steps,
        seed=seed or config["project"]["seed"],
    )

    # Score
    print("Scoring generated images...")
    scores = score_images(predictor, images, device)

    # Sort by score (highest first)
    ranked = sorted(zip(images, scores), key=lambda x: -x[1])
    images_ranked = [x[0] for x in ranked]
    scores_ranked = [x[1] for x in ranked]

    # Display results
    print(f"\n{'='*60}")
    print(f"  Results (ranked by predicted aesthetic score):")
    for i, (img, sc) in enumerate(ranked):
        print(f"    #{i+1}: Score = {sc:.2f}")
    print(f"{'='*60}")

    # Save outputs
    out_dir = Path(config["paths"]["outputs_dir"]) / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, sc) in enumerate(ranked):
        img.save(out_dir / f"design_{i+1}_score_{sc:.2f}.png")

    plot_generated_gallery(
        images_ranked, scores_ranked,
        str(out_dir / "gallery.png"),
        title=f"Generated Designs (Target: {target:.1f})",
    )

    print(f"\nSaved {n} designs to {out_dir}")
    return images_ranked, scores_ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and screen fashion designs")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--target_score", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    generate_and_screen(
        args.config, args.prompt, args.target_score,
        args.num_samples, args.guidance_scale, args.seed,
    )
