#!/usr/bin/env python3
"""
Quick script to download images for a sample of the dataset
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils import download_images

def main():
    # Configuration
    DATASET_FOLDER = Path('dataset')
    IMAGE_FOLDER = Path('images')
    
    # Read data
    print("Loading datasets...")
    train = pd.read_csv(DATASET_FOLDER / 'train.csv')
    
    # Use sample_test if available
    sample_test_path = DATASET_FOLDER / 'sample_test.csv'
    if sample_test_path.exists():
        print("Using sample_test.csv...")
        test = pd.read_csv(sample_test_path)
    else:
        test = pd.read_csv(DATASET_FOLDER / 'test.csv')
    
    # Sample the data
    print(f"Original sizes - Train: {len(train)}, Test: {len(test)}")
    train_sample = train.sample(n=min(5000, len(train)), random_state=42)
    test_sample = test.sample(n=min(100, len(test)), random_state=42)
    
    print(f"Sampled sizes - Train: {len(train_sample)}, Test: {len(test_sample)}")
    
    # Combine and get unique image links
    all_links = pd.concat([
        train_sample['image_link'],
        test_sample['image_link']
    ]).dropna().unique()
    
    print(f"Total unique image URLs to download: {len(all_links)}")
    print(f"Saving to: {IMAGE_FOLDER}")
    
    # Download
    if not IMAGE_FOLDER.exists():
        IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading images (this may take a few minutes)...")
    download_images(all_links.tolist(), str(IMAGE_FOLDER))
    
    # Check how many were downloaded
    downloaded = len([f for f in IMAGE_FOLDER.iterdir() if f.is_file()])
    print(f"\nDownload complete! {downloaded} images downloaded to {IMAGE_FOLDER}/")

if __name__ == "__main__":
    main()

