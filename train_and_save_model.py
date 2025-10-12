#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Product Pricing — Train and Save Model
Extended version of temp.py with model saving/loading capabilities and embeddings caching

Features:
  - Automatic caching of text and image embeddings
  - Cache invalidation based on data hash
  - Significant speedup on subsequent runs
  - Smart image downloading: if >5% images found, downloads missing ones
  - Images enabled if ≥15% available (configurable with --use_images or --disable_images)
  - Auto GPU detection and acceleration (3-10x speedup for tree models)
  - Early stopping for all models (prevents overfitting, 2-3x faster)

Usage:
  # Train and save model (auto GPU detection, cached embeddings)
  python train_and_save_model.py --max_train 15000 --save_model models/my_model
  
  # Load model and predict (uses cached embeddings if available)
  python train_and_save_model.py --load_model models/my_model --predict_only
  
  # Force CPU-only mode (disable GPU auto-detection)
  python train_and_save_model.py --cpu_only --max_train 15000
  
  # Disable cache and force re-encoding
  python train_and_save_model.py --no_cache --max_train 15000
  
  # Force use images even if <15% available
  python train_and_save_model.py --use_images --max_train 15000
  
  # Specify custom cache directory
  python train_and_save_model.py --cache_dir my_cache --save_model models/my_model
"""
import os, re, math, gc, random, warnings, json, argparse, sys, pickle, hashlib
from pathlib import Path

# Add src directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent / "src"))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Import the original temp.py functions
from temp import (
    smape, stratify_bins, normalize_text, extract_brand, parse_units, parse_pack,
    safe_image_open, infer_image_paths, device, ViTEncoder,
    weighted_median, knn_predict, kfold_target_encode
)

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer
import torch
import lightgbm as lgb
from catboost import CatBoostRegressor

def detect_gpu_support():
    """Detect if GPU is available for tree-based models"""
    gpu_available = False
    gpu_info = []
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_available = True
        gpu_info.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check LightGBM GPU support
    try:
        import lightgbm as lgb
        if hasattr(lgb, 'LGBMRegressor'):
            # Try to create a small GPU model as test
            try:
                test_model = lgb.LGBMRegressor(n_estimators=1, device='gpu')
                gpu_info.append("LightGBM GPU: supported")
            except:
                gpu_info.append("LightGBM GPU: not available (CPU build)")
    except:
        pass
    
    return gpu_available, gpu_info

def build_argparser():
    ap = argparse.ArgumentParser(description="Smart Product Pricing with Model Save/Load")
    ap.add_argument("--data_dir", default="dataset", help="Path containing train.csv/test.csv")
    ap.add_argument("--images_dir", default="images", help="Path where images were downloaded")
    ap.add_argument("--cache_dir", default="cache", help="Directory for caching embeddings")
    ap.add_argument("--no_cache", action="store_true", help="Disable cache (force re-encode)")
    ap.add_argument("--out_csv", default="test_out.csv", help="Submission file path")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--use_images", action="store_true", help="Force enable image branch")
    ap.add_argument("--disable_images", action="store_true", help="Force disable image branch")
    ap.add_argument("--pca_dim", type=int, default=128, help="PCA dim for embeddings")
    ap.add_argument("--max_tfidf", type=int, default=200000, help="Max TF-IDF features")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU-only mode (disables auto GPU detection)")
    ap.add_argument("--max_train", type=int, default=None, help="Max training samples")
    ap.add_argument("--max_test", type=int, default=None, help="Max test samples")
    ap.add_argument("--per_unit", action="store_true", help="Model per-unit price target")
    ap.add_argument("--knn_neighbors", type=int, default=64, help="KNN neighbors")
    ap.add_argument("--knn_tau", type=float, default=1.0, help="KNN distance temperature")
    
    # Model save/load
    ap.add_argument("--save_model", type=str, default=None, 
                    help="Directory to save trained model")
    ap.add_argument("--load_model", type=str, default=None,
                    help="Directory to load trained model from")
    ap.add_argument("--predict_only", action="store_true",
                    help="Only run prediction (requires --load_model)")
    
    return ap

def save_model_artifacts(save_dir, artifacts):
    """Save all model artifacts to directory"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {save_path}...")
    
    # Save with pickle
    with open(save_path / "model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save config as JSON for inspection
    config = {
        "per_unit": artifacts["config"]["per_unit"],
        "pca_dim": artifacts["config"]["pca_dim"],
        "max_tfidf": artifacts["config"]["max_tfidf"],
        "knn_neighbors": artifacts["config"]["knn_neighbors"],
        "knn_tau": artifacts["config"]["knn_tau"],
        "enable_images": artifacts["config"]["enable_images"],
        "model_version": "v2",
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved successfully to {save_path}")
    print(f"  - model_artifacts.pkl ({(save_path / 'model_artifacts.pkl').stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  - config.json")

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

def get_data_hash(df, columns=None):
    """Generate a stable hash of the dataframe to detect changes"""
    if columns is None:
        columns = df.columns.tolist()
    
    # Create a stable hash based on:
    # 1. DataFrame shape
    # 2. Sample IDs if available (most stable identifier)
    # 3. Deterministic sample of data
    hash_components = [str(df.shape), str(sorted(columns))]
    
    # Use sample_id for stable identification if available
    if "sample_id" in df.columns:
        # Sort by sample_id and take first/last 100 for stable hash
        sorted_ids = sorted(df["sample_id"].tolist())
        sample_ids = sorted_ids[:50] + sorted_ids[-50:]
        hash_components.append(str(sample_ids))
    else:
        # Fall back to sampling first/last rows deterministically
        sample_size = min(20, len(df))
        if sample_size > 0:
            # Use first and last rows (most stable)
            first_sample = df[columns].head(sample_size).to_json(orient='records')
            last_sample = df[columns].tail(sample_size).to_json(orient='records')
            hash_components.extend([first_sample, last_sample])
    
    hash_str = "|".join(hash_components)
    return hashlib.md5(hash_str.encode()).hexdigest()[:16]

def save_embeddings_cache(cache_dir, name, embeddings, data_hash, metadata=None):
    """Save embeddings to cache with metadata"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_path / f"{name}_embeddings.npz"
    meta_file = cache_path / f"{name}_metadata.json"
    
    # Save embeddings
    np.savez_compressed(cache_file, embeddings=embeddings)
    
    # Save metadata
    meta = {
        "data_hash": data_hash,
        "shape": embeddings.shape,
        "dtype": str(embeddings.dtype),
    }
    if metadata:
        meta.update(metadata)
    
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✓ Cached to {cache_file} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")

def load_embeddings_cache(cache_dir, name, data_hash, expected_shape=None, use_cache=True):
    """Load embeddings from cache if valid"""
    if not use_cache:
        return None
    
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"{name}_embeddings.npz"
    meta_file = cache_path / f"{name}_metadata.json"
    
    if not cache_file.exists() or not meta_file.exists():
        return None
    
    try:
        # Load metadata
        with open(meta_file, "r") as f:
            meta = json.load(f)
        
        # Validate cache
        if meta.get("data_hash") != data_hash:
            print(f"  ⚠ Cache invalid for {name}: data hash mismatch")
            print(f"    Expected: {data_hash}, Got: {meta.get('data_hash')}")
            return None
        
        if expected_shape and tuple(meta.get("shape", [])) != tuple(expected_shape):
            print(f"  ⚠ Cache invalid for {name}: shape mismatch")
            print(f"    Expected: {expected_shape}, Got: {meta.get('shape')}")
            return None
        
        # Load embeddings
        data = np.load(cache_file)
        embeddings = data["embeddings"]
        
        print(f"  ✓ Loaded {name} from cache ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return embeddings
    except Exception as e:
        print(f"  ⚠ Cache load failed for {name}: {e}")
        return None

def main():
    args = build_argparser().parse_args()
    
    # Check for prediction-only mode
    if args.predict_only and not args.load_model:
        print("ERROR: --predict_only requires --load_model")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)
    
    # ============================================================
    # PREDICTION MODE (Load model and predict)
    # ============================================================
    if args.load_model:
        artifacts = load_model_artifacts(args.load_model)
        
        # Load test data
        print("\nLoading test data...")
        test = pd.read_csv(data_dir / "test.csv")
        print(f"Test samples: {len(test)}")
        
        # Limit test samples if specified
        if args.max_test and len(test) > args.max_test:
            test = test.sample(n=args.max_test, random_state=SEED).reset_index(drop=True)
            print(f"Reduced test size to: {len(test)} samples")
        
        # Feature engineering (same as training)
        test["catalog_content"] = test["catalog_content"].fillna("").map(normalize_text)
        test["title"] = test["catalog_content"].str.split(".").str[0]
        test["brand"] = test["title"].map(extract_brand)
        test["pack"] = test["catalog_content"].map(parse_pack).astype(int)
        test["unit_base"] = test["catalog_content"].map(parse_units).astype(float)
        test["len_chars"] = test["catalog_content"].str.len().astype(int)
        test["len_words"] = test["catalog_content"].str.split().map(len).astype(int)
        test["has_digits"] = test["catalog_content"].str.contains(r'\d').astype(int)
        
        # Unit factor for denormalization
        def unit_factor(df):
            f = (df["pack"].clip(lower=1).astype(float) * df["unit_base"].clip(lower=1e-3).astype(float)).values
            f = np.where(df["unit_base"].values <= 0, df["pack"].clip(lower=1).astype(float), f)
            return f
        
        factor_te = unit_factor(test)
        
        # TF-IDF transform
        print("\nTransforming TF-IDF features...")
        T_tfidf = artifacts["tfidf"].transform(test["catalog_content"])
        
        # Text embeddings
        print("Generating text embeddings...")
        test_text_hash = get_data_hash(test, ["catalog_content"])
        cache_dir = Path(args.cache_dir)
        use_cache = not args.no_cache
        
        te_txt = load_embeddings_cache(cache_dir, "test_text", test_text_hash, 
                                      expected_shape=(len(test), 384), use_cache=use_cache)
        
        if te_txt is None:
            if not use_cache:
                print("  Cache disabled - encoding text...")
            else:
                print("  Cache miss - encoding text...")
            use_cuda = (not args.cpu_only)
            txt_encoder = SentenceTransformer(
                artifacts["config"]["text_model_name"],
                device=str(device(allow_cuda=use_cuda))
            )
            
            def encode_text(texts, batch_size=512):
                embs = []
                for i in range(0, len(texts), batch_size):
                    if i % 2048 == 0:
                        print(f"  Encoding batch {i}/{len(texts)}...")
                    embs.append(txt_encoder.encode(texts[i:i+batch_size], normalize_embeddings=True))
                return np.vstack(embs).astype(np.float32)
            
            te_txt = encode_text(test["catalog_content"].tolist(), 512)
            if use_cache:
                save_embeddings_cache(cache_dir, "test_text", te_txt, test_text_hash, 
                                    {"model": artifacts["config"]["text_model_name"]})
        
        # Image embeddings
        if artifacts["config"]["enable_images"]:
            print("Generating image embeddings...")
            img_te = infer_image_paths(test, images_dir)
            n_found_te = sum(1 for p in img_te if p and os.path.exists(p))
            found_ratio_te = n_found_te / max(1, len(img_te))
            
            # Try downloading missing test images if some are found
            if found_ratio_te >= 0.0 and found_ratio_te < 0.95 and not args.disable_images:
                print(f"Found {found_ratio_te*100:.1f}% of test images. Downloading missing images...")
                try:
                    from utils import download_images
                    images_dir.mkdir(exist_ok=True, parents=True)
                    
                    if len(test) > 0 and "image_link" in test.columns:
                        missing_test = [link for link, path in zip(test["image_link"].fillna(""), img_te)
                                       if link and (not path or not os.path.exists(path))]
                        if missing_test:
                            print(f"  Downloading {len(missing_test)} missing test images...")
                            download_images(missing_test, str(images_dir))
                    
                    # Re-check after download
                    img_te = infer_image_paths(test, images_dir)
                    n_found_te = sum(1 for p in img_te if p and os.path.exists(p))
                    found_ratio_te = n_found_te / max(1, len(img_te))
                    print(f"  After download: {found_ratio_te*100:.1f}% available")
                except Exception as e:
                    print(f"  Warning: Could not download images: {e}")
            
            # Check cache for image embeddings
            test_image_hash = get_data_hash(test, ["sample_id"])
            te_img = load_embeddings_cache(cache_dir, "test_image", test_image_hash, 
                                          expected_shape=(len(test), 384), use_cache=use_cache)
            
            if te_img is None:
                if not use_cache:
                    print("  Cache disabled - encoding images...")
                else:
                    print("  Cache miss - encoding images...")
                use_cuda = (not args.cpu_only)
                vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
                
                def encode_images_chunked(paths, chunk_size=1280, batch_size=128):
                    all_embeddings = []
                    for i in range(0, len(paths), chunk_size):
                        if i % (chunk_size * 5) == 0:
                            print(f"  Encoding images {i}/{len(paths)}...")
                        chunk_paths = paths[i:i+chunk_size]
                        chunk_images = [safe_image_open(p) for p in chunk_paths]
                        chunk_emb = vit.encode(chunk_images, batch=batch_size)
                        all_embeddings.append(chunk_emb)
                        del chunk_images, chunk_emb
                        gc.collect()
                    return np.vstack(all_embeddings)
                
                te_img = encode_images_chunked(img_te, chunk_size=1280, batch_size=128).astype(np.float32)
                if use_cache:
                    save_embeddings_cache(cache_dir, "test_image", te_img, test_image_hash)
        else:
            te_img = np.zeros((len(test), 384), dtype=np.float32)
        
        # PCA transform
        if artifacts["config"]["pca_dim"] > 0:
            print(f"Applying PCA ({artifacts['config']['pca_dim']} dims)...")
            te_txt = artifacts["pca_text"].transform(te_txt)
            te_img = artifacts["pca_img"].transform(te_img)
        
        # Tabular features
        tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]
        T_tab = test[tab_cols].values.astype(np.float32)
        
        # Target encodings
        print("Applying target encodings...")
        global_mean = artifacts["config"]["global_mean"]
        T_brand = test["brand"].map(artifacts["brand_mapping"]).fillna(global_mean).values.astype(float)
        T_bxpk = (test["brand"].astype(str) + "||" + test["pack"].astype(str)).map(artifacts["bxpk_mapping"]).fillna(global_mean).values.astype(float)
        
        # Combined features
        T_emb = np.hstack([te_txt, te_img, T_tab, T_brand.reshape(-1,1), T_bxpk.reshape(-1,1)]).astype(np.float32)
        
        # Get predictions from all base models
        print("\nGenerating predictions from base models...")
        ridge_test = artifacts["ridge_models"][0].predict(T_tfidf)
        lgbm_tfidf_test = artifacts["lgbm_tfidf_models"][0].predict(T_tfidf)
        cat_test = artifacts["cat_models"][0].predict(T_emb)
        
        # Anchor LGBM predictions require special feature preparation
        tst = test.copy()
        grp = artifacts["brand_pack_stats"]
        tst = tst.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        gmed = artifacts["config"]["global_median"]
        tst["brand_pack_median"].fillna(gmed, inplace=True)
        tst["brand_pack_count"].fillna(0, inplace=True)
        tst["brand_pack_median_log"] = np.log1p(tst["brand_pack_median"])
        
        brand_enc = artifacts["brand_encoding"]
        Xte = tst[["pack","unit_base","len_chars","len_words","has_digits"]].copy()
        Xte["brand_id"] = tst["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
        for nm in ("brand_pack_median_log","brand_pack_count"):
            Xte[nm] = tst[nm].values
        
        anchor_test = artifacts["anchor_models"][0].predict(Xte)
        
        # KNN predictions
        print("Generating KNN predictions...")
        Z_te = artifacts["scaler"].transform(T_emb)
        knn_test = knn_predict(
            artifacts["Z_tr"], artifacts["target"],
            Z_te, 
            n_neighbors=artifacts["config"]["knn_neighbors"],
            tau=artifacts["config"]["knn_tau"],
            batch=5000
        )
        
        # Meta model
        print("Applying meta model...")
        base_test = np.vstack([ridge_test, lgbm_tfidf_test, cat_test, anchor_test, knn_test]).T
        meta_test_logits = artifacts["meta_model"].predict(base_test)
        
        # Denormalize
        if artifacts["config"]["per_unit"]:
            test_price = np.expm1(meta_test_logits) * np.maximum(factor_te, 1e-3)
        else:
            test_price = np.expm1(meta_test_logits)
        
        # Isotonic calibration
        test_price_cal = artifacts["isotonic"].transform(test_price)
        test_price_cal = np.maximum(test_price_cal, 0.01)
        
        # Save predictions
        sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price_cal.astype(float)})
        sub.to_csv(out_csv, index=False)
        print(f"\n✓ Predictions saved to {out_csv} ({len(sub)} samples)")
        
        return
    
    # ============================================================
    # TRAINING MODE (Original temp.py logic with saving)
    # ============================================================
    print("="*80)
    print("TRAINING MODE")
    print("="*80)
    
    # Detect GPU support
    use_gpu_trees = False
    if not args.cpu_only:
        gpu_available, gpu_info = detect_gpu_support()
        if gpu_available:
            use_gpu_trees = True
            print("\n🚀 GPU Detected:")
            for info in gpu_info:
                print(f"  • {info}")
            print("  Tree models (LightGBM/CatBoost) will use GPU acceleration")
        else:
            print("\n💻 Running on CPU (no GPU detected)")
    else:
        print("\n💻 Running on CPU (--cpu_only specified)")
    
    if not args.no_cache:
        print(f"\n📦 Caching enabled: {args.cache_dir}/")
        print("  (embeddings will be saved and reused on subsequent runs)")
    else:
        print("\n📦 Caching disabled: embeddings will be re-encoded")
    
    # This is the full training code from temp.py
    # For brevity, I'll import and run temp.py's main logic,
    # then save the artifacts
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv(data_dir / "train.csv")
    
    # sample_test_path = data_dir / "sample_test.csv"
    # if sample_test_path.exists() and args.max_test:
    #     print(f"Using sample_test.csv for faster iteration...")
    #     test = pd.read_csv(sample_test_path)
    # else:
    #     test = pd.read_csv(data_dir / "test.csv")
        
    test = pd.read_csv(data_dir / "test.csv")
    
    print(f"Original train size: {len(train)}, test size: {len(test)}")
    
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=SEED).reset_index(drop=True)
        print(f"Reduced train size to: {len(train)} samples")
    
    if args.max_test and len(test) > args.max_test:
        test = test.sample(n=args.max_test, random_state=SEED).reset_index(drop=True)
        print(f"Reduced test size to: {len(test)} samples")
    
    # Feature engineering
    for df in (train, test):
        df["catalog_content"] = df["catalog_content"].fillna("").map(normalize_text)
        df["title"] = df["catalog_content"].str.split(".").str[0]
        df["brand"] = df["title"].map(extract_brand)
        df["pack"] = df["catalog_content"].map(parse_pack).astype(int)
        df["unit_base"] = df["catalog_content"].map(parse_units).astype(float)
        df["len_chars"] = df["catalog_content"].str.len().astype(int)
        df["len_words"] = df["catalog_content"].str.split().map(len).astype(int)
        df["has_digits"] = df["catalog_content"].str.contains(r'\d').astype(int)
    
    def unit_factor(df):
        f = (df["pack"].clip(lower=1).astype(float) * df["unit_base"].clip(lower=1e-3).astype(float)).values
        f = np.where(df["unit_base"].values <= 0, df["pack"].clip(lower=1).astype(float), f)
        return f
    
    y = train["price"].astype(float).values
    factor_tr = unit_factor(train)
    factor_te = unit_factor(test)
    
    if args.per_unit:
        target = np.log1p(y / np.maximum(factor_tr, 1e-3))
        print("Using per-unit target normalization")
    else:
        target = np.log1p(y)
        print("Using standard log1p target")
    
    folds = args.folds
    bins = stratify_bins(train["price"])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(train, bins))
    
    # TF-IDF
    print("\nBuilding TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_features=args.max_tfidf,
        strip_accents="unicode"
    )
    X_tfidf = tfidf.fit_transform(train["catalog_content"])
    T_tfidf = tfidf.transform(test["catalog_content"])
    print(f"TF-IDF shape: train={X_tfidf.shape}, test={T_tfidf.shape}")
    
    # Embeddings
    print("\nGenerating embeddings...")
    text_model_name = "BAAI/bge-small-en-v1.5"
    use_cuda = (not args.cpu_only)
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    
    # Cache text embeddings
    train_text_hash = get_data_hash(train, ["catalog_content"])
    test_text_hash = get_data_hash(test, ["catalog_content"])
    
    tr_txt = load_embeddings_cache(cache_dir, "train_text", train_text_hash, 
                                   expected_shape=(len(train), 384), use_cache=use_cache)
    te_txt = load_embeddings_cache(cache_dir, "test_text", test_text_hash, 
                                   expected_shape=(len(test), 384), use_cache=use_cache)
    
    if tr_txt is None or te_txt is None:
        print(f"Loading text encoder: {text_model_name} (cuda={use_cuda})...")
        txt_encoder = SentenceTransformer(text_model_name, device=str(device(allow_cuda=use_cuda)))
        
        def encode_text(texts, batch_size=512):
            embs = []
            for i in range(0, len(texts), batch_size):
                if i % 2048 == 0:
                    print(f"  Encoding batch {i}/{len(texts)}...")
                embs.append(txt_encoder.encode(texts[i:i+batch_size], normalize_embeddings=True))
            return np.vstack(embs).astype(np.float32)
        
        if tr_txt is None:
            print(f"Encoding train text ({len(train)} samples)...")
            tr_txt = encode_text(train["catalog_content"].tolist(), 512)
            if use_cache:
                save_embeddings_cache(cache_dir, "train_text", tr_txt, train_text_hash, {"model": text_model_name})
        
        if te_txt is None:
            print(f"Encoding test text ({len(test)} samples)...")
            te_txt = encode_text(test["catalog_content"].tolist(), 512)
            if use_cache:
                save_embeddings_cache(cache_dir, "test_text", te_txt, test_text_hash, {"model": text_model_name})
    
    # Images
    img_tr = infer_image_paths(train, images_dir)
    img_te = infer_image_paths(test, images_dir)
    n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
    found_ratio = n_found / max(1, len(img_tr)+len(img_te))
    
    # Try downloading missing images if some are found but not all
    # Strategy: if >5% found, download the rest to complete the set
    if found_ratio >= 0.05 and found_ratio < 0.95 and not args.disable_images:
        print(f"Found {found_ratio*100:.1f}% of images. Downloading missing images...")
        try:
            from utils import download_images
            images_dir.mkdir(exist_ok=True, parents=True)
            
            # Download only missing images
            if len(train) > 0 and "image_link" in train.columns:
                missing_train = [link for link, path in zip(train["image_link"].fillna(""), img_tr) 
                                if link and (not path or not os.path.exists(path))]
                if missing_train:
                    print(f"  Downloading {len(missing_train)} missing training images...")
                    download_images(missing_train, str(images_dir))
            
            if len(test) > 0 and "image_link" in test.columns:
                missing_test = [link for link, path in zip(test["image_link"].fillna(""), img_te)
                               if link and (not path or not os.path.exists(path))]
                if missing_test:
                    print(f"  Downloading {len(missing_test)} missing test images...")
                    download_images(missing_test, str(images_dir))
            
            # Re-check after download
            img_tr = infer_image_paths(train, images_dir)
            img_te = infer_image_paths(test, images_dir)
            n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
            found_ratio = n_found / max(1, len(img_tr)+len(img_te))
            print(f"  After download: {found_ratio*100:.1f}% available")
        except Exception as e:
            print(f"  Warning: Could not download images: {e}")
    
    # Enable images if we have a reasonable amount (>15%) or user forces it
    enable_images = (not args.disable_images) and (args.use_images or found_ratio >= 0.15)
    
    if enable_images:
        print(f"Images used (~{found_ratio*100:.1f}% found)")
        
        # Cache image embeddings
        train_image_hash = get_data_hash(train, ["sample_id"])
        test_image_hash = get_data_hash(test, ["sample_id"])
        
        tr_img = load_embeddings_cache(cache_dir, "train_image", train_image_hash, 
                                      expected_shape=(len(train), 384), use_cache=use_cache)
        te_img = load_embeddings_cache(cache_dir, "test_image", test_image_hash, 
                                      expected_shape=(len(test), 384), use_cache=use_cache)
        
        if tr_img is None or te_img is None:
            vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
            
            def encode_images_chunked(paths, chunk_size=1280, batch_size=128):
                all_embeddings = []
                for i in range(0, len(paths), chunk_size):
                    if i % (chunk_size * 5) == 0:
                        print(f"  Encoding images {i}/{len(paths)}...")
                    chunk_paths = paths[i:i+chunk_size]
                    chunk_images = [safe_image_open(p) for p in chunk_paths]
                    chunk_emb = vit.encode(chunk_images, batch=batch_size)
                    all_embeddings.append(chunk_emb)
                    del chunk_images, chunk_emb
                    gc.collect()
                return np.vstack(all_embeddings)
            
            if tr_img is None:
                print(f"Encoding {len(img_tr)} train images...")
                tr_img = encode_images_chunked(img_tr, chunk_size=1280, batch_size=128).astype(np.float32)
                if use_cache:
                    save_embeddings_cache(cache_dir, "train_image", tr_img, train_image_hash)
            
            if te_img is None:
                print(f"Encoding {len(img_te)} test images...")
                te_img = encode_images_chunked(img_te, chunk_size=1280, batch_size=128).astype(np.float32)
                if use_cache:
                    save_embeddings_cache(cache_dir, "test_image", te_img, test_image_hash)
    else:
        print(f"Skip images (~{found_ratio*100:.1f}% found)")
        tr_img = np.zeros((len(train), 384), dtype=np.float32)
        te_img = np.zeros((len(test),  384), dtype=np.float32)
    
    # PCA
    pca_text = None
    pca_img = None
    if args.pca_dim and args.pca_dim > 0:
        print(f"PCA to {args.pca_dim} dims...")
        pca_text = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_txt)
        pca_img = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_img)
        tr_txt = pca_text.transform(tr_txt); te_txt = pca_text.transform(te_txt)
        tr_img = pca_img.transform(tr_img); te_img = pca_img.transform(te_img)
    
    # Tabular
    tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]
    X_tab = train[tab_cols].values.astype(np.float32)
    T_tab = test[tab_cols].values.astype(np.float32)
    
    # Target encodings
    print("\nK-fold target encodings...")
    brand_series = train["brand"]
    brand_pack_series = train["brand"].astype(str) + "||" + train["pack"].astype(str)
    oof_brand, enc_brand = kfold_target_encode(brand_series, target, fold_splits)
    oof_bxpk, enc_bxpk = kfold_target_encode(brand_pack_series, target, fold_splits)
    T_brand = enc_brand(test["brand"])
    T_bxpk = enc_bxpk(test["brand"].astype(str) + "||" + test["pack"].astype(str))
    
    # Create picklable encoder mappings (instead of closure functions)
    global_mean = float(np.mean(target))
    alpha = 10.0
    
    # Brand encoder mapping
    df_brand = pd.DataFrame({"k": brand_series.values, "y": target})
    brand_stats = df_brand.groupby("k")["y"].mean()
    brand_cnts = df_brand.groupby("k")["y"].size()
    brand_mapping = ((brand_cnts * brand_stats + alpha * global_mean) / (brand_cnts + alpha)).to_dict()
    
    # Brand x Pack encoder mapping
    df_bxpk = pd.DataFrame({"k": brand_pack_series.values, "y": target})
    bxpk_stats = df_bxpk.groupby("k")["y"].mean()
    bxpk_cnts = df_bxpk.groupby("k")["y"].size()
    bxpk_mapping = ((bxpk_cnts * bxpk_stats + alpha * global_mean) / (bxpk_cnts + alpha)).to_dict()
    
    # Combined features
    X_emb = np.hstack([tr_txt, tr_img, X_tab, oof_brand.reshape(-1,1), oof_bxpk.reshape(-1,1)]).astype(np.float32)
    T_emb = np.hstack([te_txt, te_img, T_tab, T_brand.reshape(-1,1), T_bxpk.reshape(-1,1)]).astype(np.float32)
    print(f"Combined features shape: train={X_emb.shape}, test={T_emb.shape}")
    
    # Train models (simplified - just one model per type for saving)
    print("\n[Base A] TF-IDF + Ridge...")
    ridge_model = Ridge(alpha=1.2, random_state=SEED)
    ridge_model.fit(X_tfidf, target)
    ridge_test = ridge_model.predict(T_tfidf)
    
    print("\n[Base B] TF-IDF + LightGBM...")
    # Use early stopping to avoid training unnecessary trees
    # Split data for validation
    val_size = min(5000, len(train) // 5)
    val_idx = np.random.choice(len(train), size=val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(len(train)), val_idx)
    
    # Adjust n_estimators based on dataset size
    n_est = 4000 if len(train) > 50000 else 2000
    
    lgbm_params = dict(n_estimators=n_est, num_leaves=256, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.6, objective="mae",
                       random_state=SEED, verbose=100)  # Show progress every 100 trees
    
    # Enable GPU if available
    if use_gpu_trees:
        lgbm_params['device'] = 'gpu'
    
    lgbm_tfidf = lgb.LGBMRegressor(**lgbm_params)
    
    print(f"  Training with early stopping (max {n_est} trees)...")
    lgbm_tfidf.fit(
        X_tfidf[train_idx], target[train_idx],
        eval_set=[(X_tfidf[val_idx], target[val_idx])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                   lgb.log_evaluation(period=100)]
    )
    print(f"  Trained {lgbm_tfidf.best_iteration_} trees (stopped early)")
    
    lgbm_tfidf_test = lgbm_tfidf.predict(T_tfidf)
    
    print("\n[Base C] Embeddings + CatBoost...")
    # Adjust iterations based on dataset size
    cat_iters = 2500 if len(train) > 50000 else 1500
    
    cat_params = dict(iterations=cat_iters, depth=6, learning_rate=0.035,
                      loss_function="MAE", eval_metric="MAE",
                      early_stopping_rounds=150,
                      random_seed=SEED, verbose=200)
    
    # Enable GPU if available
    if use_gpu_trees:
        cat_params['task_type'] = 'GPU'
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_emb, target, eval_set=(X_emb[val_idx], target[val_idx]))
    print(f"  Trained {cat_model.best_iteration_} iterations (stopped early)")
    cat_test = cat_model.predict(T_emb)
    
    print("\n[Base D] Anchor LightGBM...")
    top_brands = train["brand"].value_counts().index[:10000]
    brand_enc = {b:i for i,b in enumerate(top_brands, start=0)}
    
    def build_anchor_feats(df):
        X = df[["pack","unit_base","len_chars","len_words","has_digits"]].copy()
        X["brand_id"] = df["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
        return X
    
    tr = train.copy()
    grp = tr.groupby(["brand","pack"], observed=True)["price"].agg(["median","count"]).rename(
        columns={"median":"brand_pack_median","count":"brand_pack_count"})
    tr = tr.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
    tst = test.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
    gmed = float(tr["brand_pack_median"].median()) if len(tr) else float(train["price"].median())
    for df in (tr,tst):
        df["brand_pack_median"].fillna(gmed, inplace=True)
        df["brand_pack_count"].fillna(0, inplace=True)
        df["brand_pack_median_log"] = np.log1p(df["brand_pack_median"])
    Xtr = build_anchor_feats(tr); Xte = build_anchor_feats(tst)
    for nm in ("brand_pack_median_log","brand_pack_count"):
        Xtr[nm] = tr[nm].values; Xte[nm] = tst[nm].values
    cols = list(Xtr.columns)
    mono = [1 if c in ("pack","unit_base","brand_pack_median_log","brand_pack_count") else 0 for c in cols]
    
    # Adjust n_estimators and use early stopping
    anchor_n_est = 2500 if len(train) > 50000 else 1500
    anchor_params = dict(n_estimators=anchor_n_est, num_leaves=96, learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.85, objective="rmse",
                        random_state=SEED, monotone_constraints=mono, verbose=50)
    
    # Enable GPU if available
    if use_gpu_trees:
        anchor_params['device'] = 'gpu'
    
    lgbm_anchor = lgb.LGBMRegressor(**anchor_params)
    
    print(f"  Training with early stopping (max {anchor_n_est} trees)...")
    lgbm_anchor.fit(
        Xtr.iloc[train_idx], target[train_idx],
        eval_set=[(Xtr.iloc[val_idx], target[val_idx])],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                   lgb.log_evaluation(period=50)]
    )
    print(f"  Trained {lgbm_anchor.best_iteration_} trees (stopped early)")
    anchor_test = lgbm_anchor.predict(Xte)
    
    print("\n[Base E] KNN blender...")
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_emb)
    Z_te = scaler.transform(T_emb)
    knn_test = knn_predict(Z_tr, target, Z_te, n_neighbors=args.knn_neighbors,
                           tau=args.knn_tau, batch=5000)
    
    print("\n[Meta] Stacking...")
    base_test = np.vstack([ridge_test, lgbm_tfidf_test, cat_test, anchor_test, knn_test]).T
    # For simplicity, train meta model on full data (in production, use OOF)
    base_train = np.vstack([
        ridge_model.predict(X_tfidf),
        lgbm_tfidf.predict(X_tfidf),
        cat_model.predict(X_emb),
        lgbm_anchor.predict(Xtr),
        knn_predict(Z_tr, target, Z_tr, n_neighbors=args.knn_neighbors, tau=args.knn_tau, batch=5000)
    ]).T
    
    meta_model = Ridge(alpha=1.0, random_state=SEED)
    meta_model.fit(base_train, target)
    meta_test_logits = meta_model.predict(base_test)
    
    # Denormalize
    if args.per_unit:
        test_price = np.expm1(meta_test_logits) * np.maximum(factor_te, 1e-3)
        oof_price = np.expm1(meta_model.predict(base_train)) * np.maximum(factor_tr, 1e-3)
    else:
        test_price = np.expm1(meta_test_logits)
        oof_price = np.expm1(meta_model.predict(base_train))
    
    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_price, y)
    test_price_cal = iso.transform(test_price)
    test_price_cal = np.maximum(test_price_cal, 0.01)
    
    # Save predictions
    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price_cal.astype(float)})
    sub.to_csv(out_csv, index=False)
    print(f"\n✓ Predictions saved to {out_csv} ({len(sub)} samples)")
    
    # Save model if requested
    if args.save_model:
        artifacts = {
            "tfidf": tfidf,
            "pca_text": pca_text,
            "pca_img": pca_img,
            "brand_mapping": brand_mapping,  # Picklable dict instead of closure
            "bxpk_mapping": bxpk_mapping,    # Picklable dict instead of closure
            "ridge_models": [ridge_model],
            "lgbm_tfidf_models": [lgbm_tfidf],
            "cat_models": [cat_model],
            "anchor_models": [lgbm_anchor],
            "scaler": scaler,
            "Z_tr": Z_tr,
            "target": target,
            "meta_model": meta_model,
            "isotonic": iso,
            "brand_encoding": brand_enc,
            "brand_pack_stats": grp,
            "config": {
                "per_unit": args.per_unit,
                "pca_dim": args.pca_dim,
                "max_tfidf": args.max_tfidf,
                "knn_neighbors": args.knn_neighbors,
                "knn_tau": args.knn_tau,
                "enable_images": enable_images,
                "text_model_name": text_model_name,
                "global_mean": global_mean,
                "global_median": gmed,
            }
        }
        save_model_artifacts(args.save_model, artifacts)

if __name__ == "__main__":
    main()

