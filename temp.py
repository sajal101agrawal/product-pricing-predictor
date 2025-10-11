#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Product Pricing — v2 (accuracy-focused)

High-impact upgrades:
- Per-unit target normalization (normalizes pack/quantity effects)
- Leak-safe K-fold target encoding (brand & brand×pack)
- KNN blender in embedding space (robust for long-tail)
- More diverse base learners (5 instead of 3)
- SMAPE-aware sample weights
- Meta CV + Isotonic calibration

Expected dataset layout:
  dataset/train.csv  (with columns: sample_id, catalog_content, image_link, price)
  dataset/test.csv   (with columns: sample_id, catalog_content, image_link)
Optional images directory:
  images/  (filenames can be {sample_id}.jpg or URL basename; code auto-detects)

Outputs:
  test_out.csv       (columns: sample_id, price)
"""
import os, re, math, gc, random, warnings, json, argparse, sys
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

# ------------------------
# CLI
# ------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Smart Product Pricing v2 - Accuracy-focused")
    ap.add_argument("--data_dir", default="dataset", help="Path containing train.csv/test.csv")
    ap.add_argument("--images_dir", default="images", help="Path where images were downloaded")
    ap.add_argument("--out_csv", default="test_out.csv", help="Submission file path")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--use_images", action="store_true", help="Force enable image branch even if few images exist")
    ap.add_argument("--disable_images", action="store_true", help="Force disable image branch")
    ap.add_argument("--pca_dim", type=int, default=128, help="PCA dim for text/img embeddings (0 to disable)")
    ap.add_argument("--max_tfidf", type=int, default=200000, help="Max TF-IDF features")
    ap.add_argument("--cpu_only", action="store_true", help="Do not use CUDA even if available")
    ap.add_argument("--max_train", type=int, default=None, help="Max training samples to use (None=all)")
    ap.add_argument("--max_test", type=int, default=None, help="Max test samples to use (None=all, for fast iteration)")
    # New v2 arguments
    ap.add_argument("--per_unit", action="store_true", help="Model per-unit price target (normalizes pack/quantity)")
    ap.add_argument("--knn_neighbors", type=int, default=64, help="Number of neighbors for KNN blender")
    ap.add_argument("--knn_tau", type=float, default=1.0, help="Distance temperature for KNN weights")
    return ap

# ------------------------
# Basic utils & metric
# ------------------------
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0

def stratify_bins(y, q=20):
    ylog = np.log1p(y.astype(float))
    try:
        return pd.qcut(ylog, q=q, labels=False, duplicates="drop")
    except Exception:
        # fallback if too few unique values
        return pd.cut(ylog, bins=q, labels=False, include_lowest=True, duplicates="drop")

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_brand(title: str) -> str:
    if not title: return "__unknown__"
    parts = re.split(r'[\-\|\:\–]', title, maxsplit=1)
    head = parts[0].strip()
    toks = head.split()[:3]
    if not toks: return "__unknown__"
    if all(t.isdigit() for t in toks): return "__unknown__"
    return " ".join(toks)

UNIT_MAP = {"ml":1e-3, "l":1.0, "g":1e-3, "kg":1.0}  # normalize to L or kg
UNIT_PATTERN = r'(\d+(?:\.\d+)?)\s*(ml|l|g|kg)\b'
PACK_PATTERN = r'(?:pack\s*of\s*|x\s*|\u00D7\s*|\(\s*)(\d{1,3})(?:\s*(?:pcs|pieces|units)?\s*\))?'

def parse_units(text: str) -> float:
    if not isinstance(text, str): return 0.0
    total = 0.0
    for m in re.finditer(UNIT_PATTERN, text):
        val, unit = m.group(1), m.group(2)
        try:
            val = float(val)
            factor = UNIT_MAP.get(unit, None)
            if factor is not None: total += val * factor
        except: pass
    return float(total)

def parse_pack(text: str) -> int:
    if not isinstance(text, str): return 1
    m = re.search(PACK_PATTERN, text)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 200: return n
        except: pass
    return 1

# ------------------------
# Image helpers
# ------------------------
from PIL import Image

def safe_image_open(fp, size=(224,224)):
    try:
        im = Image.open(fp).convert("RGB").resize(size)
        return im
    except:
        return None

def infer_image_paths(df, images_dir: Path):
    fps = []
    for sid, url in zip(df["sample_id"], df.get("image_link", [""]*len(df))):
        # priority 1: sample_id.jpg/png
        tried = []
        sid_base = re.sub(r'[^a-zA-Z0-9]', '_', str(sid))
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            fp = images_dir / f"{sid_base}{ext}"
            tried.append(str(fp))
            if fp.exists():
                fps.append(str(fp))
                break
        else:
            # priority 2: URL basename
            base = os.path.basename(str(url).split("?")[0])
            fp2 = images_dir / base
            tried.append(str(fp2))
            fps.append(str(fp2) if fp2.exists() else None)
    return fps

# ------------------------
# Embedders (small & fast; permissive licenses)
# ------------------------
import torch

def device(allow_cuda=True):
    if allow_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

# Text: BGE-small or MiniLM (both Apache-2.0)
from sentence_transformers import SentenceTransformer

# Image: ViT Small via timm (Apache-2.0)
import timm
import torchvision.transforms as T

class ViTEncoder(torch.nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", use_cuda=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.dev = device(allow_cuda=use_cuda)
        self.model.to(self.dev)
        self.tfm = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    @torch.no_grad()
    def encode(self, pil_list, batch=128):
        # pil_list: list of PIL Images (already resized in safe_image_open or here)
        bufs, out = [], []
        none_indices = []
        
        for idx, im in enumerate(pil_list):
            if im is None:
                none_indices.append(idx)
                out.append(None)
            else:
                bufs.append(self.tfm(im))
                if len(bufs) == batch:
                    b = torch.stack(bufs).float().to(self.dev)
                    feats = self.model(b).detach().cpu().numpy()
                    out.extend([f for f in feats])
                    bufs = []
                    # Clear GPU cache periodically
                    if torch.cuda.is_available() and len(out) % (batch * 10) == 0:
                        torch.cuda.empty_cache()
        
        # Process remaining images
        if bufs:
            b = torch.stack(bufs).float().to(self.dev)
            feats = self.model(b).detach().cpu().numpy()
            out.extend([f for f in feats])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Normalize length with zeros where missing
        dim = len(out[0]) if (len(out)>0 and out[0] is not None) else 384
        Z = []
        for e in out:
            if e is None:
                Z.append(np.zeros(dim, dtype=float))
            else:
                Z.append(e.astype(float))
        return np.vstack(Z)

# ------------------------
# Main training pipeline
# ------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostRegressor

# ------------------------
# v2 Helper Functions
# ------------------------
def weighted_median(values, weights):
    """Compute weighted median (robust to outliers)"""
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    c = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return v[np.searchsorted(c, cutoff)]

def knn_predict(train_feats, train_target, test_feats, n_neighbors=64, tau=1.0, batch=5000):
    """KNN prediction with distance-based weighting and weighted median"""
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(train_feats)
    preds = np.zeros(len(test_feats), dtype=float)
    for i in range(0, len(test_feats), batch):
        X = test_feats[i:i+batch]
        dist, idx = nn.kneighbors(X, return_distance=True)
        d0 = np.median(dist, axis=1, keepdims=True) + 1e-6
        w = np.exp(-dist / (tau * d0))
        for r in range(X.shape[0]):
            vals = train_target[idx[r]]
            preds[i+r] = weighted_median(vals, w[r])
    return preds

def kfold_target_encode(series, y, fold_splits, prior=None):
    """Leak-safe target encoding with smoothing"""
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
    smooth = (cnts * stats + alpha * global_mean) / (cnts + alpha)
    
    def encode_new(s):
        return s.map(smooth).fillna(global_mean).values.astype(float)
    
    return oof, encode_new

def main():
    args = build_argparser().parse_args()

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)

    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    
    # Try to use sample_test.csv if available, otherwise use test.csv
    sample_test_path = data_dir / "sample_test.csv"
    if sample_test_path.exists() and args.max_test:
        print(f"Using sample_test.csv for faster iteration...")
        test = pd.read_csv(sample_test_path)
    else:
        test = pd.read_csv(data_dir / "test.csv")

    print(f"Original train size: {len(train)}, test size: {len(test)}")
    
    # Limit train samples
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=SEED).reset_index(drop=True)
        print(f"Reduced train size to: {len(train)} samples")
    
    # Limit test samples for fast iteration
    if args.max_test and len(test) > args.max_test:
        test = test.sample(n=args.max_test, random_state=SEED).reset_index(drop=True)
        print(f"Reduced test size to: {len(test)} samples for fast iteration")

    assert "price" in train.columns, "train.csv must contain 'price'"
    
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

    # Per-unit target normalization (v2 improvement)
    def unit_factor(df):
        """Calculate unit factor for per-unit normalization"""
        f = (df["pack"].clip(lower=1).astype(float) * df["unit_base"].clip(lower=1e-3).astype(float)).values
        f = np.where(df["unit_base"].values <= 0, df["pack"].clip(lower=1).astype(float), f)
        return f

    y = train["price"].astype(float).values
    factor_tr = unit_factor(train)
    factor_te = unit_factor(test)

    # Choose target based on --per_unit flag
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

    # ---------------- TF-IDF features ----------------
    print("Building TF-IDF features...")
    print(f"Max features: {args.max_tfidf}")
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_features=args.max_tfidf,
        strip_accents="unicode"
    )
    X_tfidf = tfidf.fit_transform(train["catalog_content"])
    T_tfidf = tfidf.transform(test["catalog_content"])
    print(f"TF-IDF shape: train={X_tfidf.shape}, test={T_tfidf.shape}")

    # ---------------- Text & Image embeddings ----------------
    print("\nGenerating embeddings...")
    text_model_name = "BAAI/bge-small-en-v1.5"
    use_cuda = (not args.cpu_only)
    print(f"Loading text encoder: {text_model_name} (cuda={use_cuda})...")
    txt_encoder = SentenceTransformer(text_model_name, device=str(device(allow_cuda=use_cuda)))

    def encode_text(texts, batch_size=512):
        embs = []
        for i in range(0, len(texts), batch_size):
            if i % 2048 == 0:
                print(f"  Encoding batch {i}/{len(texts)}...")
            embs.append(txt_encoder.encode(texts[i:i+batch_size], normalize_embeddings=True))
        return np.vstack(embs).astype(np.float32)

    print(f"Encoding train text ({len(train)} samples)...")
    tr_txt = encode_text(train["catalog_content"].tolist(), 512)
    print(f"Encoding test text ({len(test)} samples)...")
    te_txt = encode_text(test["catalog_content"].tolist(), 512)

    # Image embeddings
    img_tr = infer_image_paths(train, images_dir)
    img_te = infer_image_paths(test, images_dir)
    n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
    found_ratio = n_found / max(1, len(img_tr)+len(img_te))
    
    # Try downloading if too few images
    if found_ratio < 0.1 and not args.disable_images:
        print(f"Only {found_ratio*100:.1f}% of images found. Attempting to download...")
        try:
            from utils import download_images
            images_dir.mkdir(exist_ok=True, parents=True)
            
            if len(train) > 0 and "image_link" in train.columns:
                print(f"Downloading {len(train)} training images...")
                download_images(train["image_link"].fillna("").tolist(), str(images_dir))
            
            if len(test) > 0 and "image_link" in test.columns:
                print(f"Downloading {len(test)} test images...")
                download_images(test["image_link"].fillna("").tolist(), str(images_dir))
            
            # Re-check
            img_tr = infer_image_paths(train, images_dir)
            img_te = infer_image_paths(test, images_dir)
            n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
            found_ratio = n_found / max(1, len(img_tr)+len(img_te))
            print(f"After download: {found_ratio*100:.1f}% available")
        except Exception as e:
            print(f"Warning: Could not download images: {e}")
    
    enable_images = (not args.disable_images) and (args.use_images or found_ratio >= 0.2)

    if enable_images:
        print(f"Images used (~{found_ratio*100:.1f}% found)")
        vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
        
        # Process images in chunks to avoid loading all into memory at once
        def encode_images_chunked(paths, chunk_size=1280, batch_size=128):
            """Encode images in chunks with GPU batching"""
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
        
        print(f"Encoding {len(img_tr)} train images with GPU (batch=128)...")
        tr_img = encode_images_chunked(img_tr, chunk_size=1280, batch_size=128).astype(np.float32)
        print(f"Encoding {len(img_te)} test images with GPU (batch=128)...")
        te_img = encode_images_chunked(img_te, chunk_size=1280, batch_size=128).astype(np.float32)
    else:
        print(f"Skip images (~{found_ratio*100:.1f}% found)")
        tr_img = np.zeros((len(train), 384), dtype=np.float32)
        te_img = np.zeros((len(test),  384), dtype=np.float32)

    # Optional PCA compression
    if args.pca_dim and args.pca_dim > 0:
        print(f"PCA to {args.pca_dim} dims...")
        p_txt = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_txt)
        p_img = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_img)
        tr_txt = p_txt.transform(tr_txt); te_txt = p_txt.transform(te_txt)
        tr_img = p_img.transform(tr_img); te_img = p_img.transform(te_img)

    # Tabular features
    tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]
    X_tab = train[tab_cols].values.astype(np.float32)
    T_tab = test[tab_cols].values.astype(np.float32)

    # K-fold target encodings (v2 improvement - leak-safe)
    print("\nK-fold target encodings...")
    brand_series = train["brand"]
    brand_pack_series = train["brand"].astype(str) + "||" + train["pack"].astype(str)
    oof_brand, enc_brand = kfold_target_encode(brand_series, target, fold_splits)
    oof_bxpk, enc_bxpk = kfold_target_encode(brand_pack_series, target, fold_splits)
    T_brand = enc_brand(test["brand"])
    T_bxpk = enc_bxpk(test["brand"].astype(str) + "||" + test["pack"].astype(str))

    # Combine all features for embedding models
    X_emb = np.hstack([tr_txt, tr_img, X_tab, oof_brand.reshape(-1,1), oof_bxpk.reshape(-1,1)]).astype(np.float32)
    T_emb = np.hstack([te_txt, te_img, T_tab, T_brand.reshape(-1,1), T_bxpk.reshape(-1,1)]).astype(np.float32)
    print(f"Combined features shape: train={X_emb.shape}, test={T_emb.shape}")

    # ============================================================
    # Base Learners (5 models in v2)
    # ============================================================
    
    # Base A: TF-IDF + Ridge
    print("\n[Base A] TF-IDF + Ridge...")
    ridge_oof = np.zeros(len(train)); ridge_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        model = Ridge(alpha=1.2, random_state=SEED)
        model.fit(X_tfidf[tr_idx], target[tr_idx])
        ridge_oof[va_idx] = model.predict(X_tfidf[va_idx])
        ridge_test += model.predict(T_tfidf) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], ridge_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base B: TF-IDF + LightGBM
    print("\n[Base B] TF-IDF + LightGBM...")
    lgbm_tfidf_oof = np.zeros(len(train)); lgbm_tfidf_test = np.zeros(len(test))
    lgbm_params = dict(n_estimators=4000, num_leaves=256, learning_rate=0.03, 
                       subsample=0.8, colsample_bytree=0.6, objective="mae", 
                       random_state=SEED, verbose=-1)
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        lgbm_tfidf = lgb.LGBMRegressor(**lgbm_params)
        lgbm_tfidf.fit(X_tfidf[tr_idx], target[tr_idx], 
                       eval_set=[(X_tfidf[va_idx], target[va_idx])], 
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        lgbm_tfidf_oof[va_idx] = lgbm_tfidf.predict(X_tfidf[va_idx])
        lgbm_tfidf_test += lgbm_tfidf.predict(T_tfidf) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], lgbm_tfidf_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base C: Embeddings + CatBoost (SMAPE-like weights)
    print("\n[Base C] Embeddings + CatBoost...")
    cat_oof = np.zeros(len(train)); cat_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        scale_tr = np.expm1(np.maximum(0.0, oof_bxpk[tr_idx]))
        weights = 1.0 / (1e-3 + scale_tr)
        model = CatBoostRegressor(iterations=2500, depth=6, learning_rate=0.035, 
                                   loss_function="MAE", eval_metric="MAE", 
                                   random_seed=SEED, verbose=False)
        model.fit(X_emb[tr_idx], target[tr_idx], sample_weight=weights, 
                  eval_set=(X_emb[va_idx], target[va_idx]), use_best_model=True)
        cat_oof[va_idx] = model.predict(X_emb[va_idx])
        cat_test += model.predict(T_emb) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], cat_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base D: Anchor LGBM (monotone constraints)
    print("\n[Base D] Anchor LightGBM...")
    top_brands = train["brand"].value_counts().index[:10000]
    brand_enc = {b:i for i,b in enumerate(top_brands, start=0)}
    def build_anchor_feats(df):
        X = df[["pack","unit_base","len_chars","len_words","has_digits"]].copy()
        X["brand_id"] = df["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
        return X

    anchor_oof = np.zeros(len(train)); anchor_test = np.zeros(len(test))
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
        approx_scale = np.expm1(tr["brand_pack_median_log"].values)
        weights = 1.0 / (1e-3 + approx_scale)
        # Note: monotone_constraints requires objective="rmse" (not compatible with "mae")
        lgbm = lgb.LGBMRegressor(n_estimators=2500, num_leaves=96, learning_rate=0.03, 
                                 subsample=0.85, colsample_bytree=0.85, objective="rmse", 
                                 random_state=SEED, monotone_constraints=mono, verbose=-1)
        lgbm.fit(Xtr, target[tr_idx], sample_weight=weights, 
                 eval_set=[(Xva, target[va_idx])], 
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        anchor_oof[va_idx] = lgbm.predict(Xva)
        anchor_test += lgbm.predict(Xte) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], anchor_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
        del Xtr, Xva, Xte, tr, va, tst; gc.collect()

    # Base E: KNN blender
    print("\n[Base E] KNN blender...")
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_emb); Z_te = scaler.transform(T_emb)
    knn_oof = np.zeros(len(train)); knn_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        ztr, zva = Z_tr[tr_idx], Z_tr[va_idx]
        ytr = target[tr_idx]
        knn_oof[va_idx] = knn_predict(ztr, ytr, zva, n_neighbors=args.knn_neighbors, 
                                       tau=args.knn_tau, batch=5000)
        mae = mean_absolute_error(target[va_idx], knn_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    knn_test = knn_predict(Z_tr, target, Z_te, n_neighbors=args.knn_neighbors, 
                           tau=args.knn_tau, batch=5000)
    gc.collect()

    # ============================================================
    # Meta + Isotonic Calibration (v2 improvement)
    # ============================================================
    print("\n[Meta] Stacking with isotonic calibration...")
    base_oof = np.vstack([ridge_oof, lgbm_tfidf_oof, cat_oof, anchor_oof, knn_oof]).T
    base_test = np.vstack([ridge_test, lgbm_tfidf_test, cat_test, anchor_test, knn_test]).T

    meta_oof = np.zeros(len(train)); meta_test_logits = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(base_oof[tr_idx], target[tr_idx])
        meta_oof[va_idx] = m.predict(base_oof[va_idx])
        meta_test_logits += m.predict(base_test) / len(fold_splits)

    # Convert back from log space and apply per-unit denormalization
    if args.per_unit:
        oof_price = np.expm1(meta_oof) * np.maximum(factor_tr, 1e-3)
        test_price = np.expm1(meta_test_logits) * np.maximum(factor_te, 1e-3)
    else:
        oof_price = np.expm1(meta_oof)
        test_price = np.expm1(meta_test_logits)

    # Isotonic calibration (reduces bias)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_price, train["price"].values.astype(float))
    test_price_cal = iso.transform(test_price)
    test_price_cal = np.maximum(test_price_cal, 0.01)

    # Final SMAPE on OOF (for monitoring)
    oof_smape = smape(train["price"].values, oof_price)
    print(f"\nOOF SMAPE: {oof_smape:.2f}%")

    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price_cal.astype(float)})
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(sub)} rows.")

if __name__ == "__main__":
    main()
