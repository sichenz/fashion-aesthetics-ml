# utils/data_utils.py
import os
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class FashionDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, labeled=False):
        """
        Fashion dataset loader
        Args:
            data_root: Path to the data directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
            labeled: Whether to load labels (for supervised training)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.labeled = labeled
        
        # Load metadata
        meta_path = self.data_root / f"{split}_metadata.csv"
        self.metadata = pd.read_csv(meta_path)
        
        # Set paths
        self.img_paths = [self.data_root / "images" / f"{idx}.jpg" for idx in self.metadata['image_id']]
        
        # If labeled, ensure ratings column exists
        if self.labeled:
            assert 'aesthetic_rating' in self.metadata.columns, "Metadata doesn't contain aesthetic ratings"
            
        # Get attribute columns
        self.attr_columns = [col for col in self.metadata.columns 
                             if col not in ['image_id', 'aesthetic_rating']]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Get attributes as tensor
        attributes = {}
        for attr in self.attr_columns:
            if attr in self.metadata.columns:
                attributes[attr] = self.metadata.iloc[idx][attr]
        
        if self.labeled:
            rating = self.metadata.iloc[idx]['aesthetic_rating']
            return {'image': img, 'attributes': attributes, 'rating': float(rating)}
        else:
            return {'image': img, 'attributes': attributes}

def get_transforms(image_size=512):
    """Get transforms for data augmentation and normalization"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(config):
    """Get dataloaders for training and validation"""
    image_size = config['data']['image_size']
    train_transform, val_transform = get_transforms(image_size)
    
    train_dataset = FashionDataset(
        config['paths']['processed_data'],
        split='train',
        transform=train_transform,
        labeled=True
    )
    
    val_dataset = FashionDataset(
        config['paths']['processed_data'],
        split='val',
        transform=val_transform,
        labeled=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['encoder']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['encoder']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader

def get_unlabeled_dataloader(config):
    """Get dataloader for unlabeled data"""
    image_size = config['data']['image_size']
    _, transform = get_transforms(image_size)
    
    dataset = FashionDataset(
        config['paths']['processed_data'],
        split='unlabeled',
        transform=transform,
        labeled=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['encoder']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    return loader