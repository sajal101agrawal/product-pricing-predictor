#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate predictions for real test data using a saved model

This script:
1. Loads a saved model
2. Reads test.csv (or subset of rows)
3. Generates text embeddings
4. Looks for images in images directory
5. Downloads missing images if < 10% found (unless --skip_download or --force_download)
6. Generates image embeddings
7. Makes predictions
8. Saves results to result.csv

Usage:
    # Generate predictions for all rows (auto-downloads if < 10% images found)
    python generate_predictions.py --model_path models/test_model
    
    # Generate predictions for first 100 rows
    python generate_predictions.py --model_path models/test_model --max_rows 100
    
    # Skip image downloading entirely
    python generate_predictions.py --model_path models/test_model --skip_download
    
    # Force download all images regardless of how many exist
    python generate_predictions.py --model_path models/test_model --force_download
"""

import os
import sys
import gc
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import json

# Add src directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import required functions
from temp import (
    normalize_text, extract_brand, parse_units, parse_pack,
    safe_image_open, infer_image_paths, device, ViTEncoder,
    knn_predict
)
from utils import download_images

import torch
from sentence_transformers import SentenceTransformer

SEED = 42

def load_model_artifacts(load_dir):
    """Load all model artifacts from directory"""
    load_path = Path(load_dir)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Model directory not found: {load_path}")
    
    print(f"\nLoading model from {load_path}...")
    
    with open(load_path / "model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"  Configuration:")
    print(f"    - Per-unit normalization: {artifacts['config']['per_unit']}")
    print(f"    - PCA dimensions: {artifacts['config']['pca_dim']}")
    print(f"    - Images enabled: {artifacts['config']['enable_images']}")
    
    return artifacts

def download_missing_images(test_df, images_dir):
    """Download missing images from URLs in test data"""
    # Get all image links from the dataframe
    # The download_images function already checks if files exist, so we don't need to
    if "image_link" not in test_df.columns:
        print("\n⚠ No 'image_link' column found in test data")
        return
    
    image_links = test_df["image_link"].fillna("").tolist()
    
    # Filter out empty strings - but keep the count for display
    valid_links = [link for link in image_links if link and isinstance(link, str)]
    
    if len(valid_links) == 0:
        print("\n⚠ No valid image links found in test data")
        return
    
    print(f"  Downloading from {len(valid_links)} URLs...")
    images_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        download_images(valid_links, str(images_dir))
        print(f"  ✓ Download completed")
    except Exception as e:
        print(f"  Warning: Some images could not be downloaded: {e}")

def unit_factor(df):
    """Calculate unit factor for per-unit normalization"""
    f = (df["pack"].clip(lower=1).astype(float) * df["unit_base"].clip(lower=1e-3).astype(float)).values
    f = np.where(df["unit_base"].values <= 0, df["pack"].clip(lower=1).astype(float), f)
    return f

def encode_text_embeddings(texts, text_model_name, use_cuda, batch_size=512):
    """Generate text embeddings"""
    txt_encoder = SentenceTransformer(
        text_model_name,
        device=str(device(allow_cuda=use_cuda))
    )
    
    embs = []
    for i in range(0, len(texts), batch_size):
        if i % 2048 == 0 and i > 0:
            print(f"  Encoding text batch {i}/{len(texts)}...")
        embs.append(txt_encoder.encode(texts[i:i+batch_size], normalize_embeddings=True))
    
    return np.vstack(embs).astype(np.float32)

def encode_image_embeddings(image_paths, use_cuda, chunk_size=1280, batch_size=128):
    """Generate image embeddings"""
    vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
    
    all_embeddings = []
    for i in range(0, len(image_paths), chunk_size):
        if i % (chunk_size * 5) == 0 and i > 0:
            print(f"  Encoding images {i}/{len(image_paths)}...")
        chunk_paths = image_paths[i:i+chunk_size]
        chunk_images = [safe_image_open(p) for p in chunk_paths]
        chunk_emb = vit.encode(chunk_images, batch=batch_size)
        all_embeddings.append(chunk_emb)
        del chunk_images, chunk_emb
        gc.collect()
    
    return np.vstack(all_embeddings).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for real test data using saved model"
    )
    parser.add_argument(
        "--model_path", 
        required=True,
        help="Path to saved model directory (e.g., models/test_model)"
    )
    parser.add_argument(
        "--test_csv",
        default="dataset/test.csv",
        help="Path to test CSV file (default: dataset/test.csv)"
    )
    parser.add_argument(
        "--images_dir",
        default="images",
        help="Directory containing images (default: images)"
    )
    parser.add_argument(
        "--output",
        default="result.csv",
        help="Output CSV file path (default: result.csv)"
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: None = all rows)"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Use CPU only (no CUDA)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip image downloading step (default: False, will download if < 10%% images found)"
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download images even if many already exist (default: False)"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    model_path = Path(args.model_path)
    test_csv = Path(args.test_csv)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    
    print("=" * 80)
    print(" " * 25 + "Generate Predictions")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Test CSV: {test_csv}")
    print(f"  Images directory: {images_dir}")
    print(f"  Output: {output_path}")
    print(f"  Max rows: {args.max_rows if args.max_rows else 'All'}")
    print(f"  CPU only: {args.cpu_only}")
    if args.skip_download:
        print(f"  Image download: Disabled")
    elif args.force_download:
        print(f"  Image download: Forced (always attempt)")
    else:
        print(f"  Image download: Auto (only if < 10% found)")
    print()
    print("=" * 80)
    print()
    
    # Check if test file exists
    if not test_csv.exists():
        print(f"ERROR: Test file not found: {test_csv}")
        sys.exit(1)
    
    # Load model
    artifacts = load_model_artifacts(model_path)
    
    # Load test data
    print(f"\nLoading test data from {test_csv}...")
    test = pd.read_csv(test_csv)
    print(f"  Loaded {len(test):,} samples")
    
    # Limit rows if specified
    if args.max_rows and len(test) > args.max_rows:
        print(f"  Limiting to first {args.max_rows:,} rows")
        test = test.head(args.max_rows).reset_index(drop=True)
    
    # Feature engineering
    print("\nFeature engineering...")
    test["catalog_content"] = test["catalog_content"].fillna("").map(normalize_text)
    test["title"] = test["catalog_content"].str.split(".").str[0]
    test["brand"] = test["title"].map(extract_brand)
    test["pack"] = test["catalog_content"].map(parse_pack).astype(int)
    test["unit_base"] = test["catalog_content"].map(parse_units).astype(float)
    test["len_chars"] = test["catalog_content"].str.len().astype(int)
    test["len_words"] = test["catalog_content"].str.split().map(len).astype(int)
    test["has_digits"] = test["catalog_content"].str.contains(r'\d').astype(int)
    
    # Unit factor for denormalization
    factor_te = unit_factor(test)
    
    # TF-IDF transform
    print("\nTransforming TF-IDF features...")
    T_tfidf = artifacts["tfidf"].transform(test["catalog_content"])
    print(f"  TF-IDF shape: {T_tfidf.shape}")
    
    # Text embeddings
    print("\nGenerating text embeddings...")
    use_cuda = (not args.cpu_only)
    te_txt = encode_text_embeddings(
        test["catalog_content"].tolist(),
        artifacts["config"]["text_model_name"],
        use_cuda,
        batch_size=512
    )
    print(f"  Text embeddings shape: {te_txt.shape}")
    
    # Image embeddings
    if artifacts["config"]["enable_images"]:
        print("\nProcessing images...")
        
        # Find image paths first (to check what's already available)
        img_te = infer_image_paths(test, images_dir)
        n_found = sum(1 for p in img_te if p and os.path.exists(p))
        found_ratio = n_found / max(1, len(img_te))
        print(f"  Found {n_found:,}/{len(img_te):,} images ({found_ratio*100:.1f}%)")
        
        # Download missing images based on conditions (matching temp.py logic)
        should_download = False
        if not args.skip_download and "image_link" in test.columns:
            if args.force_download:
                print(f"  Force download enabled. Attempting to download all images...")
                should_download = True
            elif found_ratio < 0.1:
                print(f"  Only {found_ratio*100:.1f}% of images found. Attempting to download...")
                should_download = True
        
        if should_download:
            download_missing_images(test, images_dir)
            
            # Re-check after download
            img_te = infer_image_paths(test, images_dir)
            n_found = sum(1 for p in img_te if p and os.path.exists(p))
            found_ratio = n_found / max(1, len(img_te))
            print(f"  After download: {found_ratio*100:.1f}% available")
        
        # Generate image embeddings
        print("  Generating image embeddings...")
        te_img = encode_image_embeddings(img_te, use_cuda, chunk_size=1280, batch_size=128)
        print(f"  Image embeddings shape: {te_img.shape}")
    else:
        print("\nImages disabled in model, using zero embeddings...")
        te_img = np.zeros((len(test), 384), dtype=np.float32)
    
    # PCA transform
    if artifacts["config"]["pca_dim"] > 0:
        print(f"\nApplying PCA ({artifacts['config']['pca_dim']} dimensions)...")
        te_txt = artifacts["pca_text"].transform(te_txt)
        te_img = artifacts["pca_img"].transform(te_img)
    
    # Tabular features
    tab_cols = ["pack", "unit_base", "len_chars", "len_words", "has_digits"]
    T_tab = test[tab_cols].values.astype(np.float32)
    
    # Target encodings
    print("\nApplying target encodings...")
    global_mean = artifacts["config"]["global_mean"]
    T_brand = test["brand"].map(artifacts["brand_mapping"]).fillna(global_mean).values.astype(float)
    T_bxpk = (test["brand"].astype(str) + "||" + test["pack"].astype(str)).map(
        artifacts["bxpk_mapping"]
    ).fillna(global_mean).values.astype(float)
    
    # Combined features
    T_emb = np.hstack([
        te_txt, te_img, T_tab, 
        T_brand.reshape(-1, 1), 
        T_bxpk.reshape(-1, 1)
    ]).astype(np.float32)
    print(f"  Combined features shape: {T_emb.shape}")
    
    # Generate predictions from base models
    print("\nGenerating predictions from base models...")
    
    print("  [1/5] Ridge model...")
    ridge_test = artifacts["ridge_models"][0].predict(T_tfidf)
    
    print("  [2/5] LightGBM TF-IDF model...")
    lgbm_tfidf_test = artifacts["lgbm_tfidf_models"][0].predict(T_tfidf)
    
    print("  [3/5] CatBoost model...")
    cat_test = artifacts["cat_models"][0].predict(T_emb)
    
    print("  [4/5] Anchor LightGBM model...")
    # Prepare anchor features
    tst = test.copy()
    grp = artifacts["brand_pack_stats"]
    tst = tst.merge(grp, left_on=["brand", "pack"], right_index=True, how="left")
    gmed = artifacts["config"]["global_median"]
    tst["brand_pack_median"].fillna(gmed, inplace=True)
    tst["brand_pack_count"].fillna(0, inplace=True)
    tst["brand_pack_median_log"] = np.log1p(tst["brand_pack_median"])
    
    brand_enc = artifacts["brand_encoding"]
    Xte = tst[["pack", "unit_base", "len_chars", "len_words", "has_digits"]].copy()
    Xte["brand_id"] = tst["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
    for nm in ("brand_pack_median_log", "brand_pack_count"):
        Xte[nm] = tst[nm].values
    
    anchor_test = artifacts["anchor_models"][0].predict(Xte)
    
    print("  [5/5] KNN model...")
    Z_te = artifacts["scaler"].transform(T_emb)
    knn_test = knn_predict(
        artifacts["Z_tr"], 
        artifacts["target"],
        Z_te,
        n_neighbors=artifacts["config"]["knn_neighbors"],
        tau=artifacts["config"]["knn_tau"],
        batch=5000
    )
    
    # Meta model
    print("\nApplying meta model...")
    base_test = np.vstack([
        ridge_test, 
        lgbm_tfidf_test, 
        cat_test, 
        anchor_test, 
        knn_test
    ]).T
    meta_test_logits = artifacts["meta_model"].predict(base_test)
    
    # Denormalize
    print("Denormalizing predictions...")
    if artifacts["config"]["per_unit"]:
        test_price = np.expm1(meta_test_logits) * np.maximum(factor_te, 1e-3)
    else:
        test_price = np.expm1(meta_test_logits)
    
    # Isotonic calibration
    print("Applying isotonic calibration...")
    test_price_cal = artifacts["isotonic"].transform(test_price)
    test_price_cal = np.maximum(test_price_cal, 0.01)
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}...")
    result = pd.DataFrame({
        "sample_id": test["sample_id"],
        "price": test_price_cal.astype(float)
    })
    result.to_csv(output_path, index=False)
    
    print()
    print("=" * 80)
    print("✓ Predictions generated successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Processed: {len(result):,} samples")
    print(f"  - Output file: {output_path}")
    print(f"  - Price range: ${result['price'].min():.2f} - ${result['price'].max():.2f}")
    print(f"  - Mean price: ${result['price'].mean():.2f}")
    print(f"  - Median price: ${result['price'].median():.2f}")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

