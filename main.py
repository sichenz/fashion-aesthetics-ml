# main.py
import os
import argparse
import torch
from pathlib import Path

from utils.data_utils import load_config, set_seed
from scripts.download_data import download_datasets
from scripts.preprocess_data import preprocess_datasets
from train_encoder import train_encoder
from train_predictor import train_predictor
from train_generator import train_generator

def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['project']['seed'])
    
    # Create directories
    os.makedirs(config['paths']['raw_data'], exist_ok=True)
    os.makedirs(config['paths']['processed_data'], exist_ok=True)
    os.makedirs(config['paths']['embeddings'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    # Download data if needed
    if args.download_data or not Path(config['paths']['raw_data']).exists():
        print("Downloading datasets...")
        download_datasets(config)
    
    # Preprocess data if needed
    if args.preprocess_data or not Path(config['paths']['processed_data']).exists():
        print("Preprocessing datasets...")
        preprocess_datasets(config)
    
    # Train encoder
    encoder_path = None
    if args.train_encoder:
        print("Training encoder...")
        encoder_path = train_encoder(args.config)
    
    # Train predictor
    predictor_path = None
    if args.train_predictor:
        print("Training predictor...")
        predictor_path = train_predictor(args.config, encoder_path)
    
    # Train generator
    if args.train_generator:
        print("Training generator...")
        generator_path = train_generator(args.config, encoder_path, predictor_path)
    
    print("All tasks completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--download_data", action="store_true", help="Download datasets")
    parser.add_argument("--preprocess_data", action="store_true", help="Preprocess datasets")
    parser.add_argument("--train_encoder", action="store_true", help="Train encoder")
    parser.add_argument("--train_predictor", action="store_true", help="Train predictor")
    parser.add_argument("--train_generator", action="store_true", help="Train generator")
    parser.add_argument("--mode", type=str, default="all", help="Mode: all, train_encoder, train_predictor, train_generator")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()
    
    # Set mode based args
    if args.mode == "all":
        args.download_data = True
        args.preprocess_data = True
        args.train_encoder = True
        args.train_predictor = True
        args.train_generator = True
    elif args.mode == "train_encoder":
        args.train_encoder = True
    elif args.mode == "train_predictor":
        args.train_predictor = True
    elif args.mode == "train_generator":
        args.train_generator = True
    
    main(args)