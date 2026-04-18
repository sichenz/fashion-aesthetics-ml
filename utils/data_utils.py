# utils/data_utils.py
"""
Data loading and preprocessing utilities.

Data Strategy (v2 — reliable sources):
1. Fashion images: ashraq/fashion-product-images-small (44K products, HuggingFace embedded)
2. Aesthetic scores: Generated using pre-trained LAION Aesthetic Predictor v2
   (CLIP ViT-L/14 linear probe trained on AVA + SAC + LAION-Aesthetics)
3. This gives us real fashion products with calibrated aesthetic scores.
"""
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm


# ============================================================================
# Config & Seed
# ============================================================================

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Transforms
# ============================================================================

def get_predictor_transforms(sz: int = 224):
    """CLIP-compatible transforms for the aesthetic predictor."""
    m = [0.48145466, 0.4578275, 0.40821073]
    s = [0.26862954, 0.26130258, 0.27577711]
    train = transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(sz),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(m, s),
    ])
    val = transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(m, s),
    ])
    return train, val


def get_generator_transforms(sz: int = 512):
    """Stable Diffusion training transforms (output in [-1, 1])."""
    return transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(sz),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def _score_to_label(s: float) -> str:
    """Convert numeric aesthetic score to descriptive text label."""
    if s >= 8.0:
        return "exceptional"
    elif s >= 7.0:
        return "excellent"
    elif s >= 6.0:
        return "very good"
    elif s >= 5.0:
        return "good"
    elif s >= 4.0:
        return "average"
    return "below average"


# ============================================================================
# LAION Aesthetic Predictor v2 — for generating pseudo-labels
# ============================================================================

class _LAIONAestheticPredictor(nn.Module):
    """
    Reproduces the architecture from:
    https://github.com/christophschuhmann/improved-aesthetic-predictor
    Linear probe on CLIP ViT-L/14 embeddings (768-dim).
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _download_aesthetic_weights(data_dir: Path) -> str:
    """Download the pre-trained LAION aesthetic predictor weights."""
    import urllib.request
    weights_path = data_dir / "aesthetic_predictor_weights.pth"
    if not weights_path.exists():
        url = (
            "https://github.com/christophschuhmann/improved-aesthetic-predictor"
            "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
        )
        print("Downloading LAION Aesthetic Predictor v2 weights...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"  Saved to {weights_path}")
    return str(weights_path)


def _score_fashion_images(
    image_paths: List[str],
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """
    Score fashion images using the pre-trained LAION Aesthetic Predictor v2.

    This uses CLIP ViT-L/14 to extract image features, then runs them through
    a linear probe trained on AVA + SAC + LAION-Aesthetics datasets.
    Returns calibrated aesthetic scores on a 1-10 scale.
    """
    from transformers import CLIPModel, CLIPProcessor

    print("Scoring fashion images with LAION Aesthetic Predictor v2...")

    # Load CLIP ViT-L/14
    print("  Loading CLIP ViT-L/14...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    # Load aesthetic predictor head
    weights_path = _download_aesthetic_weights(data_dir)
    aesthetic_head = _LAIONAestheticPredictor()
    aesthetic_head.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    aesthetic_head = aesthetic_head.to(device).eval()

    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Scoring"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # Use a placeholder — will get a neutral score
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            # Extract CLIP image features and manually apply projection
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
            features = clip_model.visual_projection(vision_outputs[1])

            # Normalize features (CLIP standard)
            features = features / features.norm(dim=-1, keepdim=True)

            # Predict aesthetic score
            batch_scores = aesthetic_head(features).squeeze(-1).cpu().tolist()

        scores.extend(batch_scores)

    # Cleanup large models from memory
    del clip_model, aesthetic_head
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Scored {len(scores)} images")
    scores_arr = np.array(scores)
    print(f"  Score distribution: mean={scores_arr.mean():.2f}, "
          f"std={scores_arr.std():.2f}, "
          f"min={scores_arr.min():.2f}, max={scores_arr.max():.2f}")

    return scores


# ============================================================================
# Dataset classes
# ============================================================================

class AestheticScoreDataset(Dataset):
    """Dataset returning images and their aesthetic scores for predictor training."""

    def __init__(self, image_paths, scores, metadata=None, transform=None):
        self.paths = image_paths
        self.scores = scores
        self.metadata = metadata  # optional category/color info
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.tf:
            img = self.tf(img)

        result = {
            "image": img,
            "score": torch.tensor(self.scores[i], dtype=torch.float32),
        }
        if self.metadata:
            result["category"] = self.metadata[i].get("category", "")
        return result


class GeneratorDataset(Dataset):
    """Dataset for Stable Diffusion LoRA fine-tuning with aesthetic captions."""

    def __init__(self, image_paths, scores, captions=None, transform=None):
        self.paths = image_paths
        self.scores = scores
        self.caps = captions
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (512, 512))
        if self.tf:
            img = self.tf(img)
        base = self.caps[i] if self.caps else "a fashion product photo"
        cap = f"{base}, {_score_to_label(self.scores[i])} aesthetic quality"
        return {
            "image": img,
            "caption": cap,
            "score": torch.tensor(self.scores[i], dtype=torch.float32),
        }


# ============================================================================
# Data preparation — download, score, cache
# ============================================================================

def prepare_dataset(
    config: dict,
    force_download: bool = False,
) -> Tuple[List[str], List[float], Optional[List[str]], Optional[List[dict]]]:
    """
    Prepare the fashion aesthetic dataset.

    Pipeline:
    1. Download ashraq/fashion-product-images-small from HuggingFace (44K images)
    2. Save images locally
    3. Score all images with LAION Aesthetic Predictor v2
    4. Cache everything

    Returns:
        (image_paths, scores, captions, metadata_list)
    """
    from utils.device import get_device

    data_dir = Path(config["paths"]["data_dir"])
    images_dir = data_dir / "fashion_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "fashion_dataset.npz"

    # ── Check cache ──────────────────────────────────────────────────────
    if cache_file.exists() and not force_download:
        print(f"Loading cached dataset from {cache_file}")
        cache = np.load(cache_file, allow_pickle=True)
        paths = cache["image_paths"].tolist()
        scores = cache["scores"].tolist()
        caps = cache["captions"].tolist() if "captions" in cache else None
        meta = cache["metadata"].tolist() if "metadata" in cache else None

        # Verify files exist
        valid = sum(1 for p in paths if os.path.exists(p))
        if valid > 100:
            print(f"  {valid}/{len(paths)} images available")
            return paths, scores, caps, meta
        else:
            print(f"  Cache stale ({valid} valid), rebuilding...")

    # ── Step 1: Download fashion images from HuggingFace ────────────────
    print("\n" + "=" * 60)
    print("  Step 1: Downloading Fashion Product Images")
    print("=" * 60)

    paths, caps, meta = [], [], []

    try:
        from datasets import load_dataset

        max_samples = (
            config["data"].get("max_train_samples", 10000)
            + config["data"].get("max_val_samples", 2000)
        )

        print(f"Loading ashraq/fashion-product-images-small (target: {max_samples})...")
        hf_ds = load_dataset("ashraq/fashion-product-images-small", split="train")
        print(f"  Dataset size: {len(hf_ds)} images")

        # Limit to max_samples
        indices = list(range(min(max_samples, len(hf_ds))))
        random.Random(config["project"]["seed"]).shuffle(indices)
        indices = indices[:max_samples]

        print(f"  Saving {len(indices)} images to {images_dir}...")
        for idx in tqdm(indices, desc="  Saving images"):
            row = hf_ds[idx]
            img = row["image"]
            if img is None:
                continue

            img_path = str(images_dir / f"fashion_{idx:06d}.jpg")
            try:
                img.convert("RGB").save(img_path, quality=90)
            except Exception:
                continue

            paths.append(img_path)

            # Build caption from metadata
            parts = []
            name = row.get("productDisplayName", "")
            article = row.get("articleType", "")
            color = row.get("baseColour", "")
            usage = row.get("usage", "")

            if name:
                parts.append(name)
            elif article:
                prefix = f"{color} " if color else ""
                parts.append(f"a {prefix}{article}")

            if usage and usage not in str(parts):
                parts.append(f"{usage} style")

            caption = ", ".join(parts) if parts else "a fashion product"
            caps.append(caption)

            meta.append({
                "category": row.get("masterCategory", ""),
                "subcategory": row.get("subCategory", ""),
                "article_type": article,
                "color": color,
                "gender": row.get("gender", ""),
                "season": row.get("season", ""),
                "usage": usage,
            })

        print(f"  Saved {len(paths)} fashion product images")

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Falling back to Fashion-MNIST...")
        paths, scores_fb, caps = _create_fallback_dataset(config, images_dir)
        meta = None
        # Save cache with fallback scores
        np.savez(
            cache_file,
            image_paths=np.array(paths, dtype=object),
            scores=np.array(scores_fb),
            captions=np.array(caps, dtype=object),
        )
        return paths, scores_fb, caps, meta

    # ── Step 2: Score images with LAION Aesthetic Predictor ─────────────
    print("\n" + "=" * 60)
    print("  Step 2: Scoring with LAION Aesthetic Predictor v2")
    print("=" * 60)

    device = get_device()
    scores = _score_fashion_images(paths, data_dir, device, batch_size=32)

    # ── Step 3: Cache everything ────────────────────────────────────────
    np.savez(
        cache_file,
        image_paths=np.array(paths, dtype=object),
        scores=np.array(scores),
        captions=np.array(caps, dtype=object),
        metadata=np.array(meta, dtype=object) if meta else np.array([]),
    )
    print(f"\nCached dataset to {cache_file}")
    print(f"Total: {len(paths)} fashion images with aesthetic scores")

    return paths, scores, caps, meta


def _create_fallback_dataset(config, images_dir):
    """Fallback using Fashion-MNIST with synthetic scores (for offline testing)."""
    from torchvision.datasets import FashionMNIST

    mx = config["data"].get("max_train_samples", 5000) + config["data"].get("max_val_samples", 1000)
    ds = FashionMNIST(root=str(Path(config["paths"]["data_dir"]) / "cache"), train=True, download=True)
    names = ["t-shirt", "trouser", "pullover", "dress", "coat",
             "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    base = {0: 5.5, 1: 6.0, 2: 5.8, 3: 7.0, 4: 6.5, 5: 6.2, 6: 5.5, 7: 6.8, 8: 7.2, 9: 6.0}
    np.random.seed(config["project"]["seed"])
    paths, scores, caps = [], [], []
    for i in range(min(mx, len(ds))):
        img, lab = ds[i]
        p = str(images_dir / f"fmnist_{i:06d}.jpg")
        img.convert("RGB").resize((224, 224), Image.BICUBIC).save(p)
        sc = float(np.clip(base[lab] + np.random.normal(0, 0.8), 1, 10))
        paths.append(p)
        scores.append(sc)
        caps.append(f"a {names[lab]}, fashion product photo")
    print(f"Created {len(paths)} fallback samples.")
    return paths, scores, caps


# ============================================================================
# DataLoader factories
# ============================================================================

def get_predictor_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for the aesthetic predictor."""
    paths, scores, _, meta = prepare_dataset(config)

    n = len(paths)
    vs = int(n * config["data"]["val_split"])
    ts = n - vs
    idx = list(range(n))
    random.Random(config["project"]["seed"]).shuffle(idx)

    tr_tf, vl_tf = get_predictor_transforms(config["data"]["image_size"])

    tr_ds = AestheticScoreDataset(
        [paths[i] for i in idx[:ts]],
        [scores[i] for i in idx[:ts]],
        [meta[i] for i in idx[:ts]] if meta else None,
        tr_tf,
    )
    vl_ds = AestheticScoreDataset(
        [paths[i] for i in idx[ts:]],
        [scores[i] for i in idx[ts:]],
        [meta[i] for i in idx[ts:]] if meta else None,
        vl_tf,
    )

    nw = config["data"]["num_workers"]
    tr_dl = DataLoader(tr_ds, batch_size=config["predictor"]["batch_size"],
                       shuffle=True, num_workers=nw, drop_last=True)
    vl_dl = DataLoader(vl_ds, batch_size=config["predictor"]["batch_size"],
                       shuffle=False, num_workers=nw)
    print(f"Predictor loaders: {len(tr_ds)} train, {len(vl_ds)} val")
    return tr_dl, vl_dl


def get_generator_dataloader(config: dict) -> DataLoader:
    """Build DataLoader for Stable Diffusion LoRA fine-tuning."""
    paths, scores, caps, _ = prepare_dataset(config)
    tf = get_generator_transforms(config["data"]["gen_image_size"])
    ds = GeneratorDataset(paths, scores, caps, tf)
    dl = DataLoader(ds, batch_size=config["generator"]["batch_size"],
                    shuffle=True, num_workers=config["data"]["num_workers"], drop_last=True)
    print(f"Generator loader: {len(ds)} samples")
    return dl
