# scripts/preprocess_data.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from segment_anything import sam_model_registry, SamPredictor

def preprocess_deepfashion(config):
    """Preprocess DeepFashion dataset"""
    raw_dir = Path(config['paths']['raw_data']) / "deepfashion"
    processed_dir = Path(config['paths']['processed_data'])
    
    # Create directories
    images_dir = processed_dir / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Initialize metadata list
    metadata = []
    
    # Process DeepFashion
    print("Processing DeepFashion dataset...")
    
    # Parse image folder
    img_path = raw_dir / "img"
    if not img_path.exists():
        print(f"Warning: {img_path} does not exist, skipping DeepFashion")
        return []
    
    attributes_path = raw_dir / "anno" / "list_attr_img.txt"
    if attributes_path.exists():
        # Read attributes file
        attr_df = pd.read_csv(attributes_path, sep="\s+", skiprows=2)
        
        # Get attribute names
        attr_names = list(attr_df.columns[1:])
        
        # Process images
        for _, row in tqdm(attr_df.iterrows(), total=len(attr_df)):
            img_file = row['image_name']
            src_img_path = raw_dir / img_file
            
            if src_img_path.exists():
                # Generate unique ID
                image_id = f"df_{len(metadata):06d}"
                
                # Copy and rename image
                dst_img_path = images_dir / f"{image_id}.jpg"
                shutil.copy(src_img_path, dst_img_path)
                
                # Extract attributes
                attrs = {
                    f"attr_{i}": 1 if row[attr] == 1 else 0
                    for i, attr in enumerate(attr_names)
                    if attr in row
                }
                
                # Add category
                category = img_file.split('/')[0]
                
                # Add to metadata
                metadata.append({
                    'image_id': image_id,
                    'source': 'deepfashion',
                    'category': category,
                    **attrs
                })
    else:
        # If attributes file doesn't exist, just process images
        for img_file in tqdm(list(img_path.glob("**/*.jpg"))):
            # Generate unique ID
            image_id = f"df_{len(metadata):06d}"
            
            # Copy and rename image
            dst_img_path = images_dir / f"{image_id}.jpg"
            shutil.copy(img_file, dst_img_path)
            
            # Add to metadata
            metadata.append({
                'image_id': image_id,
                'source': 'deepfashion',
                'category': img_file.parent.name
            })
    
    print(f"Processed {len(metadata)} images from DeepFashion")
    return metadata

def preprocess_fashiongen(config):
    """Preprocess FashionGen dataset"""
    raw_dir = Path(config['paths']['raw_data']) / "fashiongen"
    processed_dir = Path(config['paths']['processed_data'])
    
    # Create directories
    images_dir = processed_dir / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Initialize metadata list
    metadata = []
    
    # Process FashionGen
    print("Processing FashionGen dataset...")
    
    # Parse image folder
    img_path = raw_dir / "images"
    if not img_path.exists():
        print(f"Warning: {img_path} does not exist, skipping FashionGen")
        return []
    
    # Get metadata if available
    meta_path = raw_dir / "fashiongen_metadata.csv"
    meta_df = None
    
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
    
    # Process images
    for img_file in tqdm(list(img_path.glob("*.jpg"))):
        # Generate unique ID
        image_id = f"fg_{len(metadata):06d}"
        
        # Copy and rename image
        dst_img_path = images_dir / f"{image_id}.jpg"
        shutil.copy(img_file, dst_img_path)
        
        # Get metadata if available
        img_metadata = {
            'image_id': image_id,
            'source': 'fashiongen'
        }
        
        if meta_df is not None:
            orig_name = img_file.stem
            meta_row = meta_df[meta_df['image_name'] == orig_name]
            
            if not meta_row.empty:
                for col in meta_row.columns:
                    if col != 'image_name':
                        img_metadata[col] = meta_row.iloc[0][col]
        
        metadata.append(img_metadata)
    
    print(f"Processed {len(metadata)} images from FashionGen")
    return metadata

def prepare_masks(config, metadata):
    """Generate masks for images using SAM"""
    processed_dir = Path(config['paths']['processed_data'])
    images_dir = processed_dir / "images"
    masks_dir = processed_dir / "masks"
    os.makedirs(masks_dir, exist_ok=True)
    
    print("Generating masks using Segment Anything Model...")
    
    # Initialize SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    # Check if SAM checkpoint exists, otherwise download it
    if not os.path.exists(sam_checkpoint):
        print(f"Downloading SAM checkpoint...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            sam_checkpoint
        )
    
    # Initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    # Process images
    for item in tqdm(metadata):
        img_path = images_dir / f"{item['image_id']}.jpg"
        mask_path = masks_dir / f"{item['image_id']}.png"
        
        # Skip if mask already exists
        if mask_path.exists():
            continue
        
        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Generate mask
        predictor.set_image(image)
        
        # Set point prompts at the center of the image
        h, w = image.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])
        
        # Generate mask
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        # Save mask
        mask = masks[0].astype(np.uint8) * 255
        Image.fromarray(mask).save(mask_path)
    
    print("Mask generation completed!")

def create_synthetic_ratings(config, metadata):
    """Create synthetic ratings dataset if needed"""
    raw_dir = Path(config['paths']['raw_data'])
    ratings_path = raw_dir / "ratings" / "synthetic_ratings.csv"
    
    # Load ratings if exists
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        return ratings_df
    
    # Generate synthetic ratings
    print("Creating synthetic ratings dataset...")
    
    # Sample a subset of images for ratings
    rated_images = random.sample(metadata, min(5000, len(metadata)))
    
    # Generate random ratings
    np.random.seed(42)
    
    # Create ratings dataframe
    ratings = []
    for item in rated_images:
        # Generate a rating between 1 and 5
        rating = np.clip(np.random.normal(3.5, 1.0), 1, 5)
        
        ratings.append({
            'image_id': item['image_id'],
            'aesthetic_rating': rating
        })
    
    ratings_df = pd.DataFrame(ratings)
    
    # Save ratings
    os.makedirs(raw_dir / "ratings", exist_ok=True)
    ratings_df.to_csv(ratings_path, index=False)
    
    print(f"Created synthetic ratings for {len(ratings)} images")
    return ratings_df

def split_dataset(metadata, ratings_df, split_ratios=[0.7, 0.15, 0.15]):
    """Split dataset into train, validation, and test sets"""
    # Merge metadata with ratings
    merged_df = pd.DataFrame(metadata)
    
    # Filter to only keep entries with ratings
    rated_df = merged_df[merged_df['image_id'].isin(ratings_df['image_id'])]
    
    # Add ratings
    rated_df = rated_df.merge(ratings_df[['image_id', 'aesthetic_rating']], on='image_id')
    
    # Split into train, val, test
    n = len(rated_df)
    train_size = int(split_ratios[0] * n)
    val_size = int(split_ratios[1] * n)
    
    # Shuffle the data
    rated_df = rated_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    train_df = rated_df.iloc[:train_size]
    val_df = rated_df.iloc[train_size:train_size+val_size]
    test_df = rated_df.iloc[train_size+val_size:]
    
    # Create unlabeled dataframe (all images not in the rated set)
    unlabeled_df = merged_df[~merged_df['image_id'].isin(ratings_df['image_id'])]
    
    return train_df, val_df, test_df, unlabeled_df

def preprocess_datasets(config):
    """Preprocess all datasets"""
    processed_dir = Path(config['paths']['processed_data'])
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process datasets
    deepfashion_metadata = preprocess_deepfashion(config)
    fashiongen_metadata = preprocess_fashiongen(config)
    
    # Combine metadata
    metadata = deepfashion_metadata + fashiongen_metadata
    
    # Check if we have any images
    if not metadata:
        print("No images found. Please check dataset paths.")
        return
    
    # Create synthetic ratings
    ratings_df = create_synthetic_ratings(config, metadata)
    
    # Split dataset
    train_df, val_df, test_df, unlabeled_df = split_dataset(metadata, ratings_df)
    
    # Save splits
    train_df.to_csv(processed_dir / "train_metadata.csv", index=False)
    val_df.to_csv(processed_dir / "val_metadata.csv", index=False)
    test_df.to_csv(processed_dir / "test_metadata.csv", index=False)
    unlabeled_df.to_csv(processed_dir / "unlabeled_metadata.csv", index=False)
    
    # Generate masks
    prepare_masks(config, metadata)
    
    print("Dataset preprocessing completed!")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}, Unlabeled: {len(unlabeled_df)}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_utils import load_config
    
    config = load_config()
    preprocess_datasets(config)