# Fashion Aesthetics ML

A machine learning framework that **(a)** predicts consumer aesthetic ratings for product screening and **(b)** generates novel product designs aligned with consumer preference signals.

Built for **Apple Silicon** (M3 Pro / M-series) — runs entirely on your Mac GPU.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Aesthetic Score Predictor                         │
│  CLIP ViT-B/32 (frozen) → MLP Head → Score (1-10)          │
│  Trained on LAION-Aesthetics V2 (real human ratings)        │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Aesthetic-Conditioned Generator                   │
│  Stable Diffusion 1.5 + LoRA fine-tuning                    │
│  Conditioned on aesthetic quality via text prompts           │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Closed-Loop Design Screening                      │
│  Generate → Score → Rank → Select top designs               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n fashion python=3.11
conda activate fashion
pip install -r requirements.txt
```

### 2. Train the Aesthetic Predictor (~15 min on M3 Pro)

```bash
python train_predictor.py
```

This will:
- Auto-download LAION-Aesthetics data (or fall back to Fashion-MNIST)
- Train a CLIP-based aesthetic score predictor
- Save the best model to `checkpoints/best_predictor.pt`
- Generate evaluation plots in `outputs/predictor/`

### 3. Fine-tune the Generator (~2-4 hrs on M3 Pro)

```bash
python train_generator.py
```

This fine-tunes Stable Diffusion 1.5 with LoRA adapters on fashion images,
conditioned on aesthetic quality descriptors.

### 4. Generate & Screen Designs

```bash
# Generate designs targeting high aesthetic scores
python generate.py --target_score 8.0 --num_samples 4

# With a specific prompt
python generate.py --prompt "elegant minimalist handbag" --target_score 7.5 --num_samples 1 #default num_samples=4

# Custom guidance scale
python generate.py --prompt "casual streetwear sneaker" --guidance_scale 9.0
```

### 5. Evaluate the Predictor

```bash
python evaluate.py
```

## Project Structure

```
fashion-aesthetics-ml/
├── configs/config.yaml              # All hyperparameters
├── models/
│   ├── aesthetic_predictor.py       # CLIP + MLP predictor
│   └── aesthetic_generator.py       # SD 1.5 + LoRA generator
├── utils/
│   ├── device.py                    # MPS/CUDA/CPU auto-detection
│   ├── data_utils.py                # HuggingFace data pipeline
│   └── evaluation_utils.py          # Metrics & visualization
├── train_predictor.py               # Train aesthetic predictor
├── train_generator.py               # Fine-tune SD 1.5 with LoRA
├── generate.py                      # Generate + score + rank designs
├── evaluate.py                      # Evaluate predictor performance
└── requirements.txt                 # Mac-compatible dependencies
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 16 GB   | 18+ GB      |
| GPU       | Apple M1 | Apple M3 Pro+ |
| Disk      | 10 GB   | 20 GB       |

Also works on NVIDIA GPUs (CUDA) — the code auto-detects the best device.

## Key Metrics

The aesthetic predictor is evaluated on:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **Pearson r** — Linear correlation
- **Spearman ρ** — Rank correlation
