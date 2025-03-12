# scripts/download_data.py
import os
import zipfile
import requests
import tarfile
import shutil
import gzip
import rarfile
from tqdm import tqdm
from pathlib import Path
import urllib.request
import subprocess
import time

def download_file(url, destination, chunk_size=8192):
    """Download file from URL with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                progress_bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_deepfashion(config):
    """Download DeepFashion dataset (Category and Attribute Prediction)"""
    raw_dir = Path(config['paths']['raw_data'])
    deep_fashion_dir = raw_dir / "deepfashion"
    os.makedirs(deep_fashion_dir, exist_ok=True)
    
    # Google Drive URL
    gdrive_url = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ?resourcekey=0-2s7M82p8Bn7riqxWVlgctw"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the DeepFashion dataset:")
    print(f"1. Go to: {gdrive_url}")
    print("2. Navigate to the 'img' folder")
    print("3. Download 'img_highres.zip' (high-resolution images)")
    print("4. Navigate back to the main folder and download:")
    print("   - 'Anno_coarse' folder (coarse annotations)")
    print("   - 'Anno_fine' folder (fine-grained annotations)")
    print("   - 'Eval' folder (evaluation protocols)")
    print("   - 'README.txt'")
    print(f"5. Create the following structure in {deep_fashion_dir}:")
    print("   deepfashion/")
    print("   ├── Anno_coarse/")
    print("   ├── Anno_fine/")
    print("   ├── Eval/")
    print("   ├── Img/ (extract img_highres.zip here)")
    print("   └── README.txt")
    print("6. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file to mark instructions were shown
    with open(deep_fashion_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download DeepFashion dataset from: {gdrive_url}\n")
        f.write("Items needed:\n")
        f.write("- img/img_highres.zip (for high-resolution images)\n")
        f.write("- Anno_coarse/ folder\n")
        f.write("- Anno_fine/ folder\n")
        f.write("- Eval/ folder\n")
        f.write("- README.txt\n")

def download_deepfashion2(config):
    """Download DeepFashion2 dataset sample"""
    raw_dir = Path(config['paths']['raw_data'])
    df2_dir = raw_dir / "deepfashion2"
    os.makedirs(df2_dir, exist_ok=True)
    
    # Google Drive URL
    gdrive_url = "https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the DeepFashion2 dataset:")
    print(f"1. Go to: {gdrive_url}")
    print("2. Download the following files as shown in the screenshots:")
    print("   - train.zip (9.9 GB)")
    print("   - validation.zip (1.69 GB)")
    print("   - test.zip (3.11 GB)")
    print("   - json_for_validation.zip (14.2 MB)")
    print(f"3. Place the downloaded files in: {df2_dir}")
    print("4. Extract the files to create the following structure:")
    print("   deepfashion2/")
    print("   ├── train/")
    print("   ├── validation/")
    print("   ├── test/")
    print("   └── json_for_validation/")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file to mark instructions were shown
    with open(df2_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download DeepFashion2 dataset from: {gdrive_url}\n")
        f.write("Files needed:\n")
        f.write("- train.zip (9.9 GB)\n")
        f.write("- validation.zip (1.69 GB)\n")
        f.write("- test.zip (3.11 GB)\n")
        f.write("- json_for_validation.zip (14.2 MB)\n")

def download_fashion200k(config):
    """Download Fashion200k dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    fashion200k_dir = raw_dir / "fashion200k"
    os.makedirs(fashion200k_dir, exist_ok=True)
    
    # Google Drive URL
    gdrive_url = "https://drive.google.com/drive/folders/0B4Eo9mft9jwoamlYWFZBSHFzV3c?resourcekey=0-2s7M82p8Bn7riqxWVlgctw"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the Fashion200k dataset:")
    print(f"1. Go to: {gdrive_url}")
    print("2. Based on the screenshot, you should download:")
    print("   - image_urls.txt (47.9 MB)")
    print("   - 'labels' folder")
    print("   - 'detection' folder")
    print(f"3. Place the downloaded files in: {fashion200k_dir}")
    print("4. For the image_urls.txt file, you can use a script to download images during preprocessing")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file to mark instructions were shown
    with open(fashion200k_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Fashion200k dataset from: {gdrive_url}\n")
        f.write("Items needed:\n")
        f.write("- image_urls.txt (47.9 MB)\n")
        f.write("- 'labels' folder\n")
        f.write("- 'detection' folder\n")

def download_polyvore_dataset(config):
    """Download Polyvore Outfits dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    polyvore_dir = raw_dir / "polyvore"
    os.makedirs(polyvore_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://hkaidlab-my.sharepoint.com/personal/xingxingzou_aidlab_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxingzou%5Faidlab%5Fhk%2FDocuments%2FAiDLab%20%2D%20fAshIon%20TEAM%20%2D%20DATA%2FRe%2DPolyVore%2Ezip"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the Re-PolyVore dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download Re-PolyVore.zip")
    print(f"3. Extract the file to: {polyvore_dir}")
    print("4. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(polyvore_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Re-PolyVore dataset from: {sharepoint_url}\n")
        f.write("File needed: Re-PolyVore.zip\n")

def download_a100_dataset(config):
    """Download A100 dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    a100_dir = raw_dir / "a100"
    os.makedirs(a100_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://polyuit-my.sharepoint.com/personal/xingxzou_polyu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxzou%5Fpolyu%5Fedu%5Fhk%2FDocuments%2FfAshIon%2DDATA%2F%E3%80%90fAshIon%E3%80%91A100%2Ezip"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the A100 dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download 【fAshIon】A100.zip")
    print(f"3. Extract the file to: {a100_dir}")
    print("4. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(a100_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download A100 dataset from: {sharepoint_url}\n")
        f.write("File needed: 【fAshIon】A100.zip\n")

def download_evaluation3_dataset(config):
    """Download Evaluation3 dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    eval3_dir = raw_dir / "evaluation3"
    os.makedirs(eval3_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://polyuit-my.sharepoint.com/personal/xingxzou_polyu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxzou%5Fpolyu%5Fedu%5Fhk%2FDocuments%2FfAshIon%2DDATA%2F%E3%80%90fAshIon%E3%80%91EVALUATION3%2FEvaluation3%2Ezip"
    
    # From the screenshot, this dataset contains outfit evaluations with Good/Normal/Bad ratings
    print("\nIMPORTANT: You need to manually download the Evaluation3 dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download Evaluation3.zip")
    print(f"3. Extract the file to: {eval3_dir}")
    print("4. This dataset contains outfit aesthetic evaluations labeled as Good/Normal/Bad")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(eval3_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Evaluation3 dataset from: {sharepoint_url}\n")
        f.write("File needed: Evaluation3.zip\n")
        f.write("This dataset contains outfits with aesthetic ratings (Good/Normal/Bad).\n")
        f.write("It will be valuable for your aesthetic evaluation model.\n")

def download_outfit4you_dataset(config):
    """Download Outfit4You dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    o4u_dir = raw_dir / "outfit4you"
    os.makedirs(o4u_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://polyuit-my.sharepoint.com/personal/xingxzou_polyu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxzou%5Fpolyu%5Fedu%5Fhk%2FDocuments%2FfAshIon%2DDATA%2F%E3%80%90fAshIon%E3%80%91O4U%2FOutfit4You%2Ezip"
    
    # Provide instructions for manual download
    print("\nIMPORTANT: You need to manually download the Outfit4You dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download Outfit4You.zip")
    print(f"3. Extract the file to: {o4u_dir}")
    print("4. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(o4u_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Outfit4You dataset from: {sharepoint_url}\n")
        f.write("File needed: Outfit4You.zip\n")

def download_print14_dataset(config):
    """Download Print14 dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    print14_dir = raw_dir / "print14"
    os.makedirs(print14_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://polyuit-my.sharepoint.com/personal/xingxzou_polyu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxzou%5Fpolyu%5Fedu%5Fhk%2FDocuments%2FfAshIon%2DDATA%2F%E3%80%90fAshIon%E3%80%91Print14%2Erar"
    
    # From the screenshot, this dataset contains 14 types of fashion prints
    print("\nIMPORTANT: You need to manually download the Print14 dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download 【fAshIon】Print14.rar")
    print(f"3. Extract the file to: {print14_dir}")
    print("4. This dataset contains 14 types of fashion prints: stripe, dotted, allover, camouflage, checks, abstract, floral, etc.")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(print14_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Print14 dataset from: {sharepoint_url}\n")
        f.write("File needed: 【fAshIon】Print14.rar\n")
        f.write("Note: You need RAR extraction software to unpack this file.\n")
        f.write("This dataset contains 14 types of fashion prints categorized into separate folders.\n")

def download_typeaware_dataset(config):
    """Download Type-aware dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    typeaware_dir = raw_dir / "typeaware"
    os.makedirs(typeaware_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://hkaidlab-my.sharepoint.com/personal/xingxingzou_aidlab_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxingzou%5Faidlab%5Fhk%2FDocuments%2FAiDLab%20%2D%20fAshIon%20TEAM%20%2D%20DATA%2FRe%2DTypeAware%2Ezip"
    
    # From the screenshot, this contains fashion items organized into 20 categories
    print("\nIMPORTANT: You need to manually download the Type-aware dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download Re-TypeAware.zip")
    print(f"3. Extract the file to: {typeaware_dir}")
    print("4. This dataset contains fashion items organized into 20 categories including:")
    print("   Tops, Skirts, Pants, Outerwear, Dresses, Jumpsuits, Shoes, Bags, etc.")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(typeaware_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download Type-aware dataset from: {sharepoint_url}\n")
        f.write("File needed: Re-TypeAware.zip\n")
        f.write("This dataset contains fashion items organized into 20 categories.\n")
        f.write("It will be valuable for your type-based categorization model.\n")

def download_fashionda_dataset(config):
    """Download fashionDA dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    fashionda_dir = raw_dir / "fashionda"
    os.makedirs(fashionda_dir, exist_ok=True)
    
    # SharePoint URL
    sharepoint_url = "https://hkaidlab-my.sharepoint.com/personal/xingxingzou_aidlab_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxingxingzou%5Faidlab%5Fhk%2FDocuments%2FAiDLab%20%2D%20fAshIon%20TEAM%20%2D%20DATA%2F%E3%80%90AiDLab%E3%80%91fAshIon%2DDA%2Erar"
    
    # From the screenshot, this is a cross-domain dataset with products, sketches, and drawings
    print("\nIMPORTANT: You need to manually download the fashionDA dataset:")
    print(f"1. Go to: {sharepoint_url}")
    print("2. Download 【AiDLab】fAshIon-DA.rar")
    print(f"3. Extract the file to: {fashionda_dir}")
    print("4. This is a cross-domain dataset with three formats of fashion items:")
    print("   - Product images")
    print("   - Sketches")
    print("   - Drawings")
    print("5. After downloading, run the preprocessing script\n")
    
    # Create a placeholder file
    with open(fashionda_dir / "_DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Please download fashionDA dataset from: {sharepoint_url}\n")
        f.write("File needed: 【AiDLab】fAshIon-DA.rar\n")
        f.write("Note: You need RAR extraction software to unpack this file.\n")
        f.write("This is a cross-domain dataset with product images, sketches, and drawings.\n")

def download_aesthetic_ratings(config):
    """Download aesthetic ratings dataset"""
    raw_dir = Path(config['paths']['raw_data'])
    ratings_dir = raw_dir / "ratings"
    os.makedirs(ratings_dir, exist_ok=True)
    
    # URL for aesthetic ratings
    ratings_url = "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/train.csv"
    ratings_file = ratings_dir / "aesthetic_ratings.csv"
    
    if ratings_file.exists():
        print("Aesthetic ratings already downloaded.")
        return
    
    # Try to download the ratings
    print("Downloading aesthetic ratings dataset...")
    try:
        success = download_file(ratings_url, ratings_file)
        if success:
            print("Aesthetic ratings downloaded successfully!")
        else:
            print("Failed to download aesthetic ratings.")
    except Exception as e:
        print(f"Error downloading aesthetic ratings: {e}")
        print("Please download ratings manually:")
        print("1. Go to https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/train.csv")
        print("2. Download the raw file")
        print(f"3. Save it to: {ratings_file}")

def download_datasets(config):
    """Download all datasets"""
    os.makedirs(config['paths']['raw_data'], exist_ok=True)
    
    print("Starting dataset setup process...")
    
    # Download rating dataset (small, reliable)
    download_aesthetic_ratings(config)
    
    # Setup instructions for all fashion datasets
    download_deepfashion(config)
    download_deepfashion2(config)
    download_fashion200k(config)
    download_polyvore_dataset(config) 
    download_a100_dataset(config)
    download_evaluation3_dataset(config)
    download_outfit4you_dataset(config)
    download_print14_dataset(config)
    download_typeaware_dataset(config)
    download_fashionda_dataset(config)
    
    print("\n" + "="*50)
    print("FASHION DATASET DOWNLOAD INSTRUCTIONS SUMMARY")
    print("="*50)
    print("This script has created instruction files for all required datasets.")
    print("Due to access restrictions and large file sizes, you need to manually download most datasets.")
    print("\nRecommended datasets to prioritize for your research:")
    print("1. DeepFashion or DeepFashion2 - Large, well-annotated datasets with attributes")
    print("2. Evaluation3 - Contains outfit aesthetic ratings (good/normal/bad)")
    print("3. Print14 - For texture and pattern analysis")
    print("4. TypeAware - Fashion items organized into categories")
    print("\nAfter downloading the datasets, run the preprocessing script to prepare them for training.")
    print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_utils import load_config
    
    config = load_config()
    download_datasets(config)