#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Product Pricing — Train and Save Model V3 (Research-Backed Improvements)

MAJOR IMPROVEMENTS for SMAPE < 35:
===========================================
1. OOF (Out-Of-Fold) stacking - leak-free meta training
2. Char TF-IDF - captures brands/SKUs/units better
3. StratifiedGroupKFold - prevents near-duplicate leakage (group by brand||pack)
4. Robust per-unit scaling - with floor to prevent division by zero
5. Adversarial validation - reweights samples to match test distribution
6. Quantile LGBM (q20/q50/q80) - robust to tails and SMAPE asymmetry
7. Advanced calibration - ratio debias + brand×pack priors + isotonic
8. OpenCLIP ViT-B/32 - better vision encoder
9. Full GPU support - all models use GPU where possible

Research backing:
- OOF stacking: prevents in-fold leakage, improves generalization
- Char TF-IDF: captures morphological patterns (brands, SKUs)
- StratifiedGroupKFold: prevents near-duplicate train/val split
- Adversarial validation: corrects covariate shift (Kaggle standard)
- Quantile boosting: median is robust to SMAPE asymmetry
- Monotone constraints: preserve physical relationships

Usage:
  # Full training with all improvements (GPU auto-detected)
  python train_and_save_model_v3.py --max_train 0 --save_model models/v3_model --max_test 75000
  
  # With explicit flags
  python train_and_save_model_v3.py --stack_cv --char_tfidf_max 200000 --ratio_calib --prior_blend --per_unit --max_train 15000 --save_model models/v3_model
  
  # CPU only mode
  python train_and_save_model_v3.py --cpu_only --max_train 10000 --save_model models/v3_cpu
"""
import os, re, math, gc, random, warnings, json, argparse, sys, pickle, hashlib
from pathlib import Path

# Add src directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent / "src"))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Import the original temp.py functions
from temp import (
    smape, stratify_bins, normalize_text, extract_brand, parse_units, parse_pack,
    safe_image_open, infer_image_paths, device, ViTEncoder,
    weighted_median, knn_predict
)

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
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
    lgbm_gpu_works = False
    gpu_info = []
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_available = True
        gpu_info.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check LightGBM GPU support with actual test
    try:
        import lightgbm as lgb
        import numpy as np
        
        # Create a tiny dataset and try to train on GPU
        X_test = np.random.rand(100, 10)
        y_test = np.random.rand(100)
        
        test_model = lgb.LGBMRegressor(n_estimators=1, device='gpu', verbose=-1)
        test_model.fit(X_test, y_test)
        
        lgbm_gpu_works = True
        gpu_info.append("LightGBM GPU: ✓ working")
    except Exception as e:
        error_msg = str(e).lower()
        if 'opencl' in error_msg:
            gpu_info.append("LightGBM GPU: ✗ OpenCL not available")
        elif 'cuda' in error_msg:
            gpu_info.append("LightGBM GPU: ✗ CUDA not configured")
        else:
            gpu_info.append("LightGBM GPU: ✗ not available")
    
    return gpu_available and lgbm_gpu_works, gpu_info

def build_argparser():
    ap = argparse.ArgumentParser(description="Smart Product Pricing V3 - Research-backed improvements")
    ap.add_argument("--data_dir", default="dataset", help="Path containing train.csv/test.csv")
    ap.add_argument("--images_dir", default="images", help="Path where images were downloaded")
    ap.add_argument("--cache_dir", default="cache", help="Directory for caching embeddings")
    ap.add_argument("--no_cache", action="store_true", help="Disable cache (force re-encode)")
    ap.add_argument("--out_csv", default="test_out.csv", help="Submission file path")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--use_images", action="store_true", help="Force enable image branch")
    ap.add_argument("--disable_images", action="store_true", help="Force disable image branch")
    ap.add_argument("--pca_dim", type=int, default=128, help="PCA dim for embeddings")
    ap.add_argument("--max_tfidf", type=int, default=300000, help="Max word TF-IDF features")
    ap.add_argument("--char_tfidf_max", type=int, default=200000, help="Max char TF-IDF features (0 to disable)")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU-only mode (disables auto GPU detection)")
    ap.add_argument("--max_train", type=int, default=None, help="Max training samples")
    ap.add_argument("--max_test", type=int, default=None, help="Max test samples")
    ap.add_argument("--per_unit", action="store_true", help="Model per-unit price target")
    ap.add_argument("--knn_neighbors", type=int, default=64, help="KNN neighbors")
    ap.add_argument("--knn_tau", type=float, default=1.0, help="KNN distance temperature")
    
    # V3 improvements
    ap.add_argument("--stack_cv", action="store_true", help="Use OOF stacking (leak-free meta)")
    ap.add_argument("--ratio_calib", action="store_true", help="Apply ratio calibration")
    ap.add_argument("--prior_blend", action="store_true", help="Blend with brand×pack priors")
    ap.add_argument("--adversarial_reweight", action="store_true", help="Use adversarial validation reweighting")
    ap.add_argument("--quantile_lgbm", action="store_true", help="Add quantile LGBM models (q20/q50/q80)")
    ap.add_argument("--use_openclip", action="store_true", help="Use OpenCLIP ViT-B/32 instead of timm ViT-Small")
    ap.add_argument("--small_thresh", type=float, default=10.0, help="Threshold for small-price gating")
    
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
        "char_tfidf_max": artifacts["config"].get("char_tfidf_max", 0),
        "knn_neighbors": artifacts["config"]["knn_neighbors"],
        "knn_tau": artifacts["config"]["knn_tau"],
        "enable_images": artifacts["config"]["enable_images"],
        "model_version": "v3",
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

def robust_unit_factor(df):
    """Calculate unit factor with floor to prevent division issues (V3 improvement)"""
    q = df["unit_base"].astype(float).values
    p = df["pack"].clip(lower=1).astype(float).values
    # Calculate factor
    f = np.where(q > 0, p * q, p)
    # Apply floor to prevent extreme divisions (key for SMAPE)
    return np.maximum(f, 0.2)

def kfold_target_encode_oof(series, y, fold_splits, prior=None):
    """Leak-safe target encoding with OOF (V3 improvement)"""
    y = np.asarray(y, dtype=float)
    n = len(series)
    oof = np.zeros(n, dtype=float)
    global_mean = float(np.mean(y)) if prior is None else float(prior)
    
    # Compute OOF encodings
    for tr_idx, va_idx in fold_splits:
        s_tr, s_va = series.iloc[tr_idx], series.iloc[va_idx]
        y_tr = y[tr_idx]
        df_tr = pd.DataFrame({"k": s_tr.values, "y": y_tr})
        stats = df_tr.groupby("k")["y"].mean()
        cnts = df_tr.groupby("k")["y"].size()
        alpha = 10.0  # smoothing parameter
        smooth = (cnts * stats + alpha * global_mean) / (cnts + alpha)
        enc = s_va.map(smooth).fillna(global_mean).values
        oof[va_idx] = enc
    
    # Full encoder for test set
    df_all = pd.DataFrame({"k": series.values, "y": y})
    stats = df_all.groupby("k")["y"].mean()
    cnts = df_all.groupby("k")["y"].size()
    alpha = 10.0
    smooth_mapping = ((cnts * stats + alpha * global_mean) / (cnts + alpha)).to_dict()
    
    return oof, smooth_mapping, global_mean

def compute_adversarial_weights(X_train, X_test, y_train):
    """Compute adversarial validation sample weights (V3 improvement)
    
    Trains a classifier to distinguish train from test. Training samples
    that look more like test get higher weights, correcting covariate shift.
    """
    print("\n[Adversarial Validation] Computing sample weights...")
    
    # Create combined dataset
    y_adv = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    X_all = scipy.sparse.vstack([X_train, X_test])
    
    # Train classifier
    clf = LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED)
    clf.fit(X_all, y_adv)
    
    # Get probability that each train sample looks like test
    p_test_like = clf.predict_proba(X_train)[:, 1]
    
    # Compute weights: higher weight for samples that look like test
    sample_w = 1.0 / np.clip(p_test_like, 1e-3, 0.999)
    
    # Normalize weights to have mean 1.0
    sample_w = sample_w / np.mean(sample_w)
    
    print(f"  Sample weights: min={sample_w.min():.3f}, max={sample_w.max():.3f}, mean={sample_w.mean():.3f}")
    print(f"  Adversarial AUC: {clf.score(X_all, y_adv):.3f} (0.5=no shift, 1.0=perfect shift)")
    
    return sample_w

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
        print("PREDICTION MODE - Feature not fully implemented for V3 yet")
        print("Please use generate_predictions.py for now")
        sys.exit(1)
    
    # ============================================================
    # TRAINING MODE (V3 with research-backed improvements)
    # ============================================================
    print("="*80)
    print("TRAINING MODE - V3 (Research-Backed Improvements)")
    print("="*80)
    
    # Detect GPU support
    use_gpu_trees = False
    if not args.cpu_only:
        gpu_available, gpu_info = detect_gpu_support()
        if gpu_available:
            use_gpu_trees = True
            print("\n🚀 GPU Available:")
            for info in gpu_info:
                print(f"  • {info}")
            print("  ✓ Tree models (LightGBM/CatBoost) will use GPU acceleration")
        else:
            print("\n💻 GPU Status:")
            for info in gpu_info:
                print(f"  • {info}")
            print("  → Training will use CPU (slower but reliable)")
    else:
        print("\n💻 Running on CPU (--cpu_only specified)")
    
    if not args.no_cache:
        print(f"\n📦 Caching enabled: {args.cache_dir}/")
        print("  (embeddings will be saved and reused on subsequent runs)")
    else:
        print("\n📦 Caching disabled: embeddings will be re-encoded")
    
    # V3 improvements summary
    print("\n🔬 V3 Improvements Enabled:")
    print(f"  • OOF Stacking: {'✓' if args.stack_cv else '✗'}")
    print(f"  • Char TF-IDF: {'✓' if args.char_tfidf_max > 0 else '✗'}")
    print(f"  • StratifiedGroupKFold: ✓ (always enabled)")
    print(f"  • Robust unit scaling: ✓ (always enabled)")
    print(f"  • Adversarial reweighting: {'✓' if args.adversarial_reweight else '✗'}")
    print(f"  • Quantile LGBM: {'✓' if args.quantile_lgbm else '✗'}")
    print(f"  • Ratio calibration: {'✓' if args.ratio_calib else '✗'}")
    print(f"  • Brand×Pack priors: {'✓' if args.prior_blend else '✗'}")
    print(f"  • OpenCLIP vision: {'✓' if args.use_openclip else '✗'}")
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv(data_dir / "train.csv")
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
    
    y = train["price"].astype(float).values
    factor_tr = robust_unit_factor(train)
    factor_te = robust_unit_factor(test)
    
    if args.per_unit:
        target = np.log1p(y / factor_tr)
        print("Using per-unit target normalization with robust scaling")
    else:
        target = np.log1p(y)
        print("Using standard log1p target")
    
    folds = args.folds
    bins = stratify_bins(train["price"])
    
    # V3 IMPROVEMENT: StratifiedGroupKFold to prevent near-duplicate leakage
    groups = train["brand"].fillna("unk").astype(str) + "||" + train["pack"].astype(str)
    print(f"\n[V3] Using StratifiedGroupKFold (groups=brand||pack, n={folds})")
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_splits = list(sgkf.split(train, bins, groups))
    
    # TF-IDF - Word level
    print("\nBuilding word TF-IDF features...")
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        min_df=3,
        max_features=args.max_tfidf,
        strip_accents="unicode"
    )
    X_tfidf_word = tfidf_word.fit_transform(train["catalog_content"])
    T_tfidf_word = tfidf_word.transform(test["catalog_content"])
    print(f"Word TF-IDF shape: train={X_tfidf_word.shape}, test={T_tfidf_word.shape}")
    
    # V3 IMPROVEMENT: Char TF-IDF
    if args.char_tfidf_max > 0:
        print("\n[V3] Building char TF-IDF features...")
        tfidf_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3,5),
            min_df=5,
            max_features=args.char_tfidf_max,
            strip_accents="unicode"
        )
        X_tfidf_char = tfidf_char.fit_transform(train["catalog_content"])
        T_tfidf_char = tfidf_char.transform(test["catalog_content"])
        print(f"Char TF-IDF shape: train={X_tfidf_char.shape}, test={T_tfidf_char.shape}")
        
        # Combine word + char TF-IDF
        X_tfidf = scipy.sparse.hstack([X_tfidf_word, X_tfidf_char])
        T_tfidf = scipy.sparse.hstack([T_tfidf_word, T_tfidf_char])
        print(f"Combined TF-IDF shape: train={X_tfidf.shape}, test={T_tfidf.shape}")
    else:
        X_tfidf = X_tfidf_word
        T_tfidf = T_tfidf_word
    
    # V3 IMPROVEMENT: Adversarial validation weights
    sample_weights = None
    if args.adversarial_reweight:
        sample_weights = compute_adversarial_weights(X_tfidf, T_tfidf, target)
    
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
        
        # V3 IMPROVEMENT: OpenCLIP option
        if args.use_openclip:
            try:
                import open_clip
                print("[V3] Using OpenCLIP ViT-B/32 (better vision encoder)...")
                clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", 
                    pretrained="laion2b_s34b_b79k",
                    device="cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
                )
                
                def encode_images_openclip(paths, batch_size=128):
                    """Encode images with OpenCLIP"""
                    all_embs = []
                    dev = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
                    
                    with torch.no_grad():
                        for i in range(0, len(paths), batch_size):
                            if i % (batch_size * 10) == 0 and i > 0:
                                print(f"    Encoding images {i}/{len(paths)}...")
                            
                            batch_paths = paths[i:i+batch_size]
                            images = []
                            for p in batch_paths:
                                img = safe_image_open(p)
                                if img:
                                    images.append(clip_preprocess(img).unsqueeze(0))
                                else:
                                    # Create zero tensor for missing images
                                    images.append(torch.zeros(1, 3, 224, 224))
                            
                            if images:
                                imgs_tensor = torch.cat(images).to(dev)
                                embs = clip_model.encode_image(imgs_tensor).float().cpu().numpy()
                                # Normalize
                                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                                embs = embs / norms
                                all_embs.append(embs)
                                
                                del imgs_tensor, images
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                    
                    return np.vstack(all_embs).astype(np.float32)
                
                # Cache image embeddings
                train_image_hash = get_data_hash(train, ["sample_id"]) + "_openclip"
                test_image_hash = get_data_hash(test, ["sample_id"]) + "_openclip"
                
                tr_img = load_embeddings_cache(cache_dir, "train_image_openclip", train_image_hash, 
                                              expected_shape=(len(train), 512), use_cache=use_cache)
                te_img = load_embeddings_cache(cache_dir, "test_image_openclip", test_image_hash, 
                                              expected_shape=(len(test), 512), use_cache=use_cache)
                
                if tr_img is None:
                    print(f"Encoding {len(img_tr)} train images with OpenCLIP...")
                    tr_img = encode_images_openclip(img_tr, batch_size=128)
                    if use_cache:
                        save_embeddings_cache(cache_dir, "train_image_openclip", tr_img, train_image_hash)
                
                if te_img is None:
                    print(f"Encoding {len(img_te)} test images with OpenCLIP...")
                    te_img = encode_images_openclip(img_te, batch_size=128)
                    if use_cache:
                        save_embeddings_cache(cache_dir, "test_image_openclip", te_img, test_image_hash)
                
            except ImportError:
                print("  ⚠ OpenCLIP not installed, falling back to ViT-Small")
                print("  Install with: pip install open-clip-torch")
                args.use_openclip = False
            except Exception as e:
                print(f"  ⚠ OpenCLIP failed: {e}")
                print("  Falling back to ViT-Small")
                args.use_openclip = False
        
        # Fall back to standard ViT if OpenCLIP not used
        if not args.use_openclip:
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
        # Determine image embedding size based on model
        img_dim = 512 if args.use_openclip else 384
        tr_img = np.zeros((len(train), img_dim), dtype=np.float32)
        te_img = np.zeros((len(test), img_dim), dtype=np.float32)
    
    # PCA
    pca_text = None
    pca_img = None
    if args.pca_dim and args.pca_dim > 0:
        print(f"PCA to {args.pca_dim} dims...")
        pca_text = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_txt)
        pca_img = PCA(n_components=min(args.pca_dim, tr_img.shape[1]), random_state=SEED).fit(tr_img)
        tr_txt = pca_text.transform(tr_txt); te_txt = pca_text.transform(te_txt)
        tr_img = pca_img.transform(tr_img); te_img = pca_img.transform(te_img)
    
    # Tabular
    tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]
    X_tab = train[tab_cols].values.astype(np.float32)
    T_tab = test[tab_cols].values.astype(np.float32)
    
    # Target encodings - OOF version
    print("\nK-fold target encodings (OOF)...")
    brand_series = train["brand"]
    brand_pack_series = train["brand"].astype(str) + "||" + train["pack"].astype(str)
    oof_brand, brand_mapping, global_mean_brand = kfold_target_encode_oof(brand_series, target, fold_splits)
    oof_bxpk, bxpk_mapping, global_mean_bxpk = kfold_target_encode_oof(brand_pack_series, target, fold_splits)
    
    # Test encodings
    T_brand = test["brand"].map(brand_mapping).fillna(global_mean_brand).values.astype(float)
    T_bxpk = (test["brand"].astype(str) + "||" + test["pack"].astype(str)).map(bxpk_mapping).fillna(global_mean_bxpk).values.astype(float)
    
    # Combined features
    X_emb = np.hstack([tr_txt, tr_img, X_tab, oof_brand.reshape(-1,1), oof_bxpk.reshape(-1,1)]).astype(np.float32)
    T_emb = np.hstack([te_txt, te_img, T_tab, T_brand.reshape(-1,1), T_bxpk.reshape(-1,1)]).astype(np.float32)
    print(f"Combined features shape: train={X_emb.shape}, test={T_emb.shape}")
    
    # Continuing in next section...
    print("\n" + "="*80)
    print("TRAINING BASE MODELS WITH OOF STACKING")
    print("="*80)
    
    # Store OOF predictions and test predictions for each base model
    base_model_names = []
    oof_predictions = []
    test_predictions = []
    
    # ========================================
    # BASE MODEL A: Ridge (Word TF-IDF)
    # ========================================
    print("\n[Base A] TF-IDF Word + Ridge...")
    ridge_word_oof = np.zeros(len(train))
    ridge_word_test = np.zeros(len(test))
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        model = Ridge(alpha=1.2, random_state=SEED)
        
        # Apply sample weights if adversarial reweighting is enabled
        if sample_weights is not None:
            # Ridge doesn't support sample_weight directly, use WLS
            w = np.sqrt(sample_weights[tr_idx])
            X_w = X_tfidf_word[tr_idx].multiply(w[:, np.newaxis])
            y_w = target[tr_idx] * w
            model.fit(X_w, y_w)
        else:
            model.fit(X_tfidf_word[tr_idx], target[tr_idx])
        
        ridge_word_oof[va_idx] = model.predict(X_tfidf_word[va_idx])
        ridge_word_test += model.predict(T_tfidf_word) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], ridge_word_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    
    base_model_names.append("ridge_word")
    oof_predictions.append(ridge_word_oof)
    test_predictions.append(ridge_word_test)
    gc.collect()
    
    # ========================================
    # BASE MODEL B: Ridge (Char TF-IDF) - if enabled
    # ========================================
    if args.char_tfidf_max > 0:
        print("\n[Base B] TF-IDF Char + Ridge...")
        ridge_char_oof = np.zeros(len(train))
        ridge_char_test = np.zeros(len(test))
        
        for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
            model = Ridge(alpha=1.0, random_state=SEED)
            
            if sample_weights is not None:
                w = np.sqrt(sample_weights[tr_idx])
                X_w = X_tfidf_char[tr_idx].multiply(w[:, np.newaxis])
                y_w = target[tr_idx] * w
                model.fit(X_w, y_w)
            else:
                model.fit(X_tfidf_char[tr_idx], target[tr_idx])
            
            ridge_char_oof[va_idx] = model.predict(X_tfidf_char[va_idx])
            ridge_char_test += model.predict(T_tfidf_char) / len(fold_splits)
            mae = mean_absolute_error(target[va_idx], ridge_char_oof[va_idx])
            print(f"  Fold {fold} MAE: {mae:.4f}")
        
        base_model_names.append("ridge_char")
        oof_predictions.append(ridge_char_oof)
        test_predictions.append(ridge_char_test)
        gc.collect()
    
    # ========================================
    # BASE MODEL C: LightGBM (Combined TF-IDF)
    # ========================================
    print("\n[Base C] TF-IDF Combined + LightGBM...")
    lgbm_tfidf_oof = np.zeros(len(train))
    lgbm_tfidf_test = np.zeros(len(test))
    
    # Adjust n_estimators based on dataset size
    n_est = 4000 if len(train) > 50000 else 2000
    
    lgbm_params = dict(
        n_estimators=n_est,
        num_leaves=256,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        objective="mae",
        random_state=SEED,
        verbose=-1
    )
    
    # Enable GPU if available
    if use_gpu_trees:
        lgbm_params['device'] = 'gpu'
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        
        # Prepare sample weights
        sw = sample_weights[tr_idx] if sample_weights is not None else None
        
        try:
            lgbm_model.fit(
                X_tfidf[tr_idx], target[tr_idx],
                sample_weight=sw,
                eval_set=[(X_tfidf[va_idx], target[va_idx])],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        except Exception as e:
            if 'gpu' in str(e).lower() or 'opencl' in str(e).lower():
                print(f"  ⚠ GPU failed, falling back to CPU...")
                lgbm_params['device'] = 'cpu'
                lgbm_model = lgb.LGBMRegressor(**lgbm_params)
                lgbm_model.fit(
                    X_tfidf[tr_idx], target[tr_idx],
                    sample_weight=sw,
                    eval_set=[(X_tfidf[va_idx], target[va_idx])],
                    eval_metric='mae',
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
            else:
                raise
        
        lgbm_tfidf_oof[va_idx] = lgbm_model.predict(X_tfidf[va_idx])
        lgbm_tfidf_test += lgbm_model.predict(T_tfidf) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], lgbm_tfidf_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}, Trees: {lgbm_model.best_iteration_}")
    
    base_model_names.append("lgbm_tfidf")
    oof_predictions.append(lgbm_tfidf_oof)
    test_predictions.append(lgbm_tfidf_test)
    gc.collect()
    
    # ========================================
    # BASE MODEL D: CatBoost (Embeddings)
    # ========================================
    print("\n[Base D] Embeddings + CatBoost...")
    cat_oof = np.zeros(len(train))
    cat_test = np.zeros(len(test))
    
    # Adjust iterations based on dataset size
    cat_iters = 2500 if len(train) > 50000 else 1500
    
    cat_params = dict(
        iterations=cat_iters,
        depth=6,
        learning_rate=0.035,
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=150,
        random_seed=SEED,
        verbose=False
    )
    
    # Enable GPU if available
    if use_gpu_trees:
        cat_params['task_type'] = 'GPU'
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        # SMAPE-aware weights: higher weight for smaller prices
        scale_tr = np.expm1(np.maximum(0.0, oof_bxpk[tr_idx]))
        weights = 1.0 / (1e-3 + scale_tr)
        
        # Combine with adversarial weights if enabled
        if sample_weights is not None:
            weights = weights * sample_weights[tr_idx]
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        cat_model = CatBoostRegressor(**cat_params)
        
        try:
            cat_model.fit(
                X_emb[tr_idx], target[tr_idx],
                sample_weight=weights,
                eval_set=(X_emb[va_idx], target[va_idx]),
                use_best_model=True
            )
        except Exception as e:
            if 'gpu' in str(e).lower():
                print(f"  ⚠ GPU failed, falling back to CPU...")
                cat_params['task_type'] = 'CPU'
                cat_model = CatBoostRegressor(**cat_params)
                cat_model.fit(
                    X_emb[tr_idx], target[tr_idx],
                    sample_weight=weights,
                    eval_set=(X_emb[va_idx], target[va_idx]),
                    use_best_model=True
                )
            else:
                raise
        
        cat_oof[va_idx] = cat_model.predict(X_emb[va_idx])
        cat_test += cat_model.predict(T_emb) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], cat_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}, Iterations: {cat_model.best_iteration_}")
    
    base_model_names.append("catboost")
    oof_predictions.append(cat_oof)
    test_predictions.append(cat_test)
    gc.collect()
    
    # ========================================
    # BASE MODEL E: Anchor LightGBM (Monotone)
    # ========================================
    print("\n[Base E] Anchor LightGBM (monotone constraints)...")
    top_brands = train["brand"].value_counts().index[:10000]
    brand_enc = {b:i for i,b in enumerate(top_brands, start=0)}
    
    def build_anchor_feats(df):
        X = df[["pack","unit_base","len_chars","len_words","has_digits"]].copy()
        X["brand_id"] = df["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
        return X
    
    anchor_oof = np.zeros(len(train))
    anchor_test = np.zeros(len(test))
    
    # Adjust n_estimators
    anchor_n_est = 2500 if len(train) > 50000 else 1500
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        tr, va = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
        grp = tr.groupby(["brand","pack"], observed=True)["price"].agg(["median","count"]).rename(
            columns={"median":"brand_pack_median","count":"brand_pack_count"})
        tr = tr.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        va = va.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        tst = test.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        gmed = float(tr["brand_pack_median"].median()) if len(tr) else float(train["price"].median())
        for df in (tr,va,tst):
            df["brand_pack_median"].fillna(gmed, inplace=True)
            df["brand_pack_count"].fillna(0, inplace=True)
            df["brand_pack_median_log"] = np.log1p(df["brand_pack_median"])
        
        Xtr = build_anchor_feats(tr); Xva = build_anchor_feats(va); Xte = build_anchor_feats(tst)
        for nm in ("brand_pack_median_log","brand_pack_count"):
            Xtr[nm] = tr[nm].values; Xva[nm] = va[nm].values; Xte[nm] = tst[nm].values
        
        cols = list(Xtr.columns)
        mono = [1 if c in ("pack","unit_base","brand_pack_median_log","brand_pack_count") else 0 for c in cols]
        
        # Weights
        approx_scale = np.expm1(tr["brand_pack_median_log"].values)
        weights = 1.0 / (1e-3 + approx_scale)
        if sample_weights is not None:
            weights = weights * sample_weights[tr_idx]
        weights = weights / np.mean(weights)
        
        anchor_params = dict(
            n_estimators=anchor_n_est,
            num_leaves=96,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="rmse",  # required for monotone_constraints
            random_state=SEED,
            monotone_constraints=mono,
            verbose=-1
        )
        
        if use_gpu_trees:
            anchor_params['device'] = 'gpu'
        
        lgbm_anchor = lgb.LGBMRegressor(**anchor_params)
        
        try:
            lgbm_anchor.fit(
                Xtr, target[tr_idx],
                sample_weight=weights,
                eval_set=[(Xva, target[va_idx])],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        except Exception as e:
            if 'gpu' in str(e).lower() or 'opencl' in str(e).lower():
                print(f"  ⚠ GPU failed, falling back to CPU...")
                anchor_params['device'] = 'cpu'
                lgbm_anchor = lgb.LGBMRegressor(**anchor_params)
                lgbm_anchor.fit(
                    Xtr, target[tr_idx],
                    sample_weight=weights,
                    eval_set=[(Xva, target[va_idx])],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
            else:
                raise
        
        anchor_oof[va_idx] = lgbm_anchor.predict(Xva)
        anchor_test += lgbm_anchor.predict(Xte) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], anchor_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}, Trees: {lgbm_anchor.best_iteration_}")
        
        del Xtr, Xva, Xte, tr, va, tst
        gc.collect()
    
    base_model_names.append("anchor")
    oof_predictions.append(anchor_oof)
    test_predictions.append(anchor_test)
    gc.collect()
    
    # ========================================
    # BASE MODEL F: KNN Blender
    # ========================================
    print("\n[Base F] KNN blender...")
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_emb)
    Z_te = scaler.transform(T_emb)
    
    knn_oof = np.zeros(len(train))
    knn_test = np.zeros(len(test))
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        ztr, zva = Z_tr[tr_idx], Z_tr[va_idx]
        ytr = target[tr_idx]
        knn_oof[va_idx] = knn_predict(ztr, ytr, zva, n_neighbors=args.knn_neighbors, 
                                       tau=args.knn_tau, batch=5000)
        mae = mean_absolute_error(target[va_idx], knn_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    
    knn_test = knn_predict(Z_tr, target, Z_te, n_neighbors=args.knn_neighbors, 
                           tau=args.knn_tau, batch=5000)
    
    base_model_names.append("knn")
    oof_predictions.append(knn_oof)
    test_predictions.append(knn_test)
    gc.collect()
    
    # ========================================
    # V3 IMPROVEMENT: Quantile LGBM Models
    # ========================================
    if args.quantile_lgbm:
        print("\n[V3] Quantile LightGBM models (q20/q50/q80)...")
        
        for alpha in [0.2, 0.5, 0.8]:
            print(f"\n  Training quantile α={alpha}...")
            q_oof = np.zeros(len(train))
            q_test = np.zeros(len(test))
            
            q_params = dict(
                n_estimators=1200,
                num_leaves=192,
                learning_rate=0.035,
                subsample=0.9,
                colsample_bytree=0.7,
                objective="quantile",
                alpha=alpha,
                random_state=SEED,
                verbose=-1
            )
            
            if use_gpu_trees:
                q_params['device'] = 'gpu'
            
            for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
                q_model = lgb.LGBMRegressor(**q_params)
                
                sw = sample_weights[tr_idx] if sample_weights is not None else None
                
                try:
                    q_model.fit(
                        X_tfidf[tr_idx], target[tr_idx],
                        sample_weight=sw,
                        eval_set=[(X_tfidf[va_idx], target[va_idx])],
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                except Exception as e:
                    if 'gpu' in str(e).lower() or 'opencl' in str(e).lower():
                        q_params['device'] = 'cpu'
                        q_model = lgb.LGBMRegressor(**q_params)
                        q_model.fit(
                            X_tfidf[tr_idx], target[tr_idx],
                            sample_weight=sw,
                            eval_set=[(X_tfidf[va_idx], target[va_idx])],
                            callbacks=[lgb.early_stopping(100, verbose=False)]
                        )
                    else:
                        raise
                
                q_oof[va_idx] = q_model.predict(X_tfidf[va_idx])
                q_test += q_model.predict(T_tfidf) / len(fold_splits)
            
            mae = mean_absolute_error(target, q_oof)
            print(f"    Overall MAE: {mae:.4f}")
            
            base_model_names.append(f"lgbm_q{int(alpha*100)}")
            oof_predictions.append(q_oof)
            test_predictions.append(q_test)
            gc.collect()
    
    # ========================================
    # META MODEL (ElasticNet with positive constraint)
    # ========================================
    print("\n" + "="*80)
    print("META MODEL (ElasticNet with OOF stacking)")
    print("="*80)
    
    # Stack OOF predictions
    base_oof = np.vstack(oof_predictions).T
    base_test = np.vstack(test_predictions).T
    
    print(f"\nBase model predictions shape: OOF={base_oof.shape}, Test={base_test.shape}")
    print(f"Base models: {base_model_names}")
    
    # Train meta model with CV
    meta_oof = np.zeros(len(train))
    meta_test_logits = np.zeros(len(test))
    
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        # Use ElasticNet with positive constraint (all weights >= 0)
        meta_model = ElasticNet(alpha=0.001, l1_ratio=0.5, positive=True, random_state=SEED, max_iter=10000)
        meta_model.fit(base_oof[tr_idx], target[tr_idx])
        
        meta_oof[va_idx] = meta_model.predict(base_oof[va_idx])
        meta_test_logits += meta_model.predict(base_test) / len(fold_splits)
        
        mae = mean_absolute_error(target[va_idx], meta_oof[va_idx])
        print(f"  Fold {fold} Meta MAE: {mae:.4f}")
    
    print(f"\nMeta model weights: {meta_model.coef_}")
    
    # ========================================
    # DENORMALIZATION
    # ========================================
    print("\n" + "="*80)
    print("DENORMALIZATION & CALIBRATION")
    print("="*80)
    
    # Convert from log space
    if args.per_unit:
        oof_price = np.expm1(meta_oof) * factor_tr
        test_price = np.expm1(meta_test_logits) * factor_te
    else:
        oof_price = np.expm1(meta_oof)
        test_price = np.expm1(meta_test_logits)
    
    # V3 IMPROVEMENT: Ratio calibration
    if args.ratio_calib:
        print("\n[V3] Applying ratio calibration...")
        ratio = (y.sum() + 1e-6) / (oof_price.sum() + 1e-6)
        test_price *= ratio
        oof_price *= ratio
        print(f"  Calibration ratio: {ratio:.4f}")
    
    # V3 IMPROVEMENT: Brand×Pack priors
    if args.prior_blend:
        print("\n[V3] Blending with brand×pack priors...")
        pri = train.groupby(["brand","pack"], observed=True)["price"].agg(["median","count"])
        pri.columns = ["prior_median", "prior_count"]
        
        # Merge with test
        test_with_prior = test.merge(pri, left_on=["brand","pack"], right_index=True, how="left")
        med = test_with_prior["prior_median"].fillna(train["price"].median())
        cnt = test_with_prior["prior_count"].fillna(0)
        
        # Adaptive blending: more trust in prior when count is high
        blend_weight = cnt.clip(0, 200) / (cnt.clip(0, 200) + 30.0)
        test_price = blend_weight.values * med.values + (1 - blend_weight.values) * test_price
        
        # Clip to reasonable bounds
        test_price = np.clip(test_price, 0.35 * med.values, 2.5 * med.values)
        print(f"  Prior blend weight range: {blend_weight.min():.3f} - {blend_weight.max():.3f}")
    
    # Isotonic calibration (always applied)
    print("\nApplying isotonic calibration...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_price, y)
    test_price_cal = iso.transform(test_price)
    test_price_cal = np.maximum(test_price_cal, 0.01)
    
    # Final OOF SMAPE
    oof_price_cal = iso.transform(oof_price)
    oof_smape = smape(y, oof_price_cal)
    print(f"\n✓ Final OOF SMAPE: {oof_smape:.2f}%")
    
    # Save predictions
    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price_cal.astype(float)})
    sub.to_csv(out_csv, index=False)
    print(f"\n✓ Predictions saved to {out_csv} ({len(sub)} samples)")
    
    # Save model if requested
    if args.save_model:
        print("\nModel saving not fully implemented for V3 yet")
        print("Will be added in future update")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal OOF SMAPE: {oof_smape:.2f}%")
    print(f"Target: < 35%")
    print(f"Improvement needed: {max(0, oof_smape - 35):.2f}%")

if __name__ == "__main__":
    main()

