#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Price Prediction (SMAPE target), license-clean and competition-compliant.

- Uses ONLY the provided dataset files (no external price lookup).
- Works CPU-only by default; will use GPU if available (for embedding extraction).
- Small, fast, open-source encoders: BGE-small (text) + ViT-Small (image) via timm.
- Three base learners -> meta-stacker. Positive price constraint. Exact CSV format.

Expected dataset layout:
  dataset/train.csv  (with columns: sample_id, catalog_content, image_link, price)
  dataset/test.csv   (with columns: sample_id, catalog_content, image_link)
Optional images directory (if you've downloaded images using your src/utils.py):
  images/  (filenames can be {sample_id}.jpg or URL basename; code auto-detects)

Outputs:
  test_out.csv       (columns: sample_id, price)

Author: MIT/Apache-2.0 compatible stack only.
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
    ap = argparse.ArgumentParser(description="Smart Product Pricing - Multimodal Stacking")
    ap.add_argument("--data_dir", default="dataset", help="Path containing train.csv/test.csv")
    ap.add_argument("--images_dir", default="images", help="Path where images were downloaded")
    ap.add_argument("--out_csv", default="test_out.csv", help="Submission file path")
    ap.add_argument("--folds", type=int, default=5, help="CV folds")
    ap.add_argument("--use_images", action="store_true", help="Force enable image branch even if few images exist")
    ap.add_argument("--disable_images", action="store_true", help="Force disable image branch")
    ap.add_argument("--pca_dim", type=int, default=128, help="PCA dim for text/img embeddings (0 to disable)")
    ap.add_argument("--max_tfidf", type=int, default=150000, help="Max TF-IDF features")
    ap.add_argument("--cpu_only", action="store_true", help="Do not use CUDA even if available")
    ap.add_argument("--max_train", type=int, default=5000, help="Max training samples to use")
    ap.add_argument("--max_test", type=int, default=100, help="Max test samples to use (for fast iteration)")
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
    def encode(self, pil_list, batch=64):
        # pil_list: list of PIL Images (already resized in safe_image_open or here)
        bufs, out = [], []
        for im in pil_list:
            if im is None:
                out.append(None)
            else:
                bufs.append(self.tfm(im))
                if len(bufs) == batch:
                    b = torch.stack(bufs).float().to(self.dev)
                    feats = self.model(b).detach().cpu().numpy()
                    out.extend([f for f in feats])
                    bufs = []
        if bufs:
            b = torch.stack(bufs).float().to(self.dev)
            feats = self.model(b).detach().cpu().numpy()
            out.extend([f for f in feats])
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

import lightgbm as lgb
from catboost import CatBoostRegressor

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
    # normalize/catalog prep
    for df in (train, test):
        df["catalog_content"] = df["catalog_content"].fillna("").map(normalize_text)
        df["title"] = df["catalog_content"].str.split(".").str[0]
        df["brand"] = df["title"].map(extract_brand)
        df["pack"] = df["catalog_content"].map(parse_pack).astype(int)
        df["unit_base"] = df["catalog_content"].map(parse_units).astype(float)
        df["len_chars"] = df["catalog_content"].str.len().astype(int)
        df["len_words"] = df["catalog_content"].str.split().map(len).astype(int)
        df["has_digits"] = df["catalog_content"].str.contains(r'\d').astype(int)

    y = train["price"].astype(float)
    y_log = np.log1p(y)
    folds = args.folds
    bins = stratify_bins(y)

    # ---------------- TF-IDF + Ridge (text branch) ----------------
    print("TF-IDF + Ridge...")
    print(f"Building TF-IDF with max_features={args.max_tfidf}...")
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_features=args.max_tfidf,
        strip_accents="unicode"
    )
    ridge_oof = np.zeros(len(train))
    ridge_test = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
        print(f"  Fold {f}/{folds} - Fitting TF-IDF...", flush=True)
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        Xtr = tfidf.fit_transform(tr["catalog_content"])
        Xva = tfidf.transform(va["catalog_content"])
        Xte = tfidf.transform(test["catalog_content"])
        print(f"  Fold {f}/{folds} - Training Ridge on {Xtr.shape} features...", flush=True)

        model = Ridge(alpha=1.5, random_state=SEED)
        model.fit(Xtr, ytr)
        ridge_oof[va_idx] = model.predict(Xva)
        ridge_test += model.predict(Xte) / folds

        mae = mean_absolute_error(yva, ridge_oof[va_idx])
        print(f"[Fold {f}] Ridge MAE(log-space): {mae:.4f}")
        del Xtr, Xva, Xte; gc.collect()

    # ---------------- Text & Image embeddings + CatBoost -----------
    print("\nEmbeddings (text/image) + CatBoost...")
    # Text encoder: bge-small (fast, Apache-2.0)
    text_model_name = "BAAI/bge-small-en-v1.5"
    # Device selection
    use_cuda = (not args.cpu_only)
    print(f"Loading text encoder: {text_model_name} (cuda={use_cuda})...", flush=True)
    txt_encoder = SentenceTransformer(text_model_name, device=str(device(allow_cuda=use_cuda)))

    # Encode text (normalize)
    def encode_text(texts, batch_size=512):
        embs = []
        for i in range(0, len(texts), batch_size):
            if i % 2048 == 0:
                print(f"  Encoding batch {i}/{len(texts)}...", flush=True)
            embs.append(txt_encoder.encode(texts[i:i+batch_size], normalize_embeddings=True))
        return np.vstack(embs).astype(np.float32)

    print(f"Encoding train text embeddings ({len(train)} samples)...", flush=True)
    train_text_emb = encode_text(train["catalog_content"].tolist(), 512)
    print(f"Encoding test text embeddings ({len(test)} samples)...", flush=True)
    test_text_emb  = encode_text(test["catalog_content"].tolist(), 512)

    # Image embeddings (optional if images available)
    # First, check if images need to be downloaded
    img_paths_train = infer_image_paths(train, images_dir)
    img_paths_test  = infer_image_paths(test, images_dir)

    n_found_train = sum(1 for p in img_paths_train if p and os.path.exists(p))
    n_found_test  = sum(1 for p in img_paths_test  if p and os.path.exists(p))
    found_ratio = (n_found_train + n_found_test) / max(1, len(img_paths_train) + len(img_paths_test))
    
    # If very few images are found and not explicitly disabled, try downloading
    if found_ratio < 0.1 and not args.disable_images:
        print(f"Only {found_ratio*100:.1f}% of images found. Attempting to download images...")
        try:
            from utils import download_images
            
            # Create images directory if it doesn't exist
            images_dir.mkdir(exist_ok=True, parents=True)
            
            # Download train images
            if len(train) > 0 and "image_link" in train.columns:
                print(f"Downloading {len(train)} training images...")
                train_links = train["image_link"].fillna("").tolist()
                download_images(train_links, str(images_dir))
            
            # Download test images
            if len(test) > 0 and "image_link" in test.columns:
                print(f"Downloading {len(test)} test images...")
                test_links = test["image_link"].fillna("").tolist()
                download_images(test_links, str(images_dir))
            
            # Re-check after downloading
            img_paths_train = infer_image_paths(train, images_dir)
            img_paths_test  = infer_image_paths(test, images_dir)
            n_found_train = sum(1 for p in img_paths_train if p and os.path.exists(p))
            n_found_test  = sum(1 for p in img_paths_test  if p and os.path.exists(p))
            found_ratio = (n_found_train + n_found_test) / max(1, len(img_paths_train) + len(img_paths_test))
            print(f"After download: {found_ratio*100:.1f}% of images available ({n_found_train} train, {n_found_test} test)")
        except Exception as e:
            print(f"Warning: Could not download images: {e}")
            print("Continuing without images...")
    
    enable_images = (not args.disable_images) and (args.use_images or (found_ratio >= 0.2))  # auto-disable if too few

    if enable_images:
        print(f"Images found for ~{found_ratio*100:.1f}% samples. Using image branch.")
        vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
        def batch_open(paths):
            ims = []
            for p in paths:
                if p and os.path.exists(p):
                    ims.append(safe_image_open(p))
                else:
                    ims.append(None)
            return ims
        train_img_emb = vit.encode(batch_open(img_paths_train), batch=64).astype(np.float32)
        test_img_emb  = vit.encode(batch_open(img_paths_test),  batch=64).astype(np.float32)
    else:
        print(f"Insufficient images (~{found_ratio*100:.1f}% found). Skipping image branch; using zeros.")
        # Fill zeros with fixed dim 384 (ViT-Small) to keep shapes consistent
        train_img_emb = np.zeros((len(train), 384), dtype=np.float32)
        test_img_emb  = np.zeros((len(test),  384), dtype=np.float32)

    # Optional PCA compression
    if args.pca_dim and args.pca_dim > 0:
        print(f"PCA to {args.pca_dim} dims for text and image embeddings...")
        p_txt = PCA(n_components=args.pca_dim, random_state=SEED).fit(train_text_emb)
        p_img = PCA(n_components=args.pca_dim, random_state=SEED).fit(train_img_emb)
        train_text_emb = p_txt.transform(train_text_emb)
        test_text_emb  = p_txt.transform(test_text_emb)
        train_img_emb  = p_img.transform(train_img_emb)
        test_img_emb   = p_img.transform(test_img_emb)

    # Tabular numerics
    tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]
    X_tab = train[tab_cols].values.astype(np.float32)
    T_tab = test[tab_cols].values.astype(np.float32)

    X_emb = np.hstack([train_text_emb, train_img_emb, X_tab]).astype(np.float32)
    T_emb = np.hstack([test_text_emb,  test_img_emb,  T_tab]).astype(np.float32)

    cat_oof = np.zeros(len(train))
    cat_test = np.zeros(len(test))

    for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
        trX, vaX = X_emb[tr_idx], X_emb[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        print(f"  Fold {f}/{folds} - Training CatBoost...", flush=True)
        model = CatBoostRegressor(
            iterations=2000,
            depth=6,
            learning_rate=0.035,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=SEED,
            verbose=False
        )
        model.fit(trX, ytr, eval_set=(vaX, yva), use_best_model=True)
        cat_oof[va_idx] = model.predict(vaX)
        cat_test += model.predict(T_emb) / folds

        mae = mean_absolute_error(yva, cat_oof[va_idx])
        print(f"[Fold {f}] CatBoost MAE(log-space): {mae:.4f}")
        del trX, vaX; gc.collect()

    # ---------------- Brand-Pack Anchor (LightGBM) -----------------
    print("Brand-Pack Anchor GBM...")
    top_brands = train["brand"].value_counts().index[:10000]
    brand_enc = {b:i for i,b in enumerate(top_brands, start=0)}

    def build_anchor_feats(df):
        X = df[tab_cols].copy()
        X["brand_id"] = df["brand"].map(lambda b: brand_enc.get(b, -1)).astype(int)
        return X

    anchor_oof = np.zeros(len(train))
    anchor_test = np.zeros(len(test))

    for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
        tr, va = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        grp = tr.groupby(["brand","pack"], observed=True)["price"].agg(["median","count"]).rename(columns={"median":"brand_pack_median","count":"brand_pack_count"})
        tr = tr.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        va = va.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
        tst = test.merge(grp, left_on=["brand","pack"], right_index=True, how="left")

        global_med = float(tr["price"].median()) if len(tr) else float(train["price"].median())
        for df in (tr, va, tst):
            df["brand_pack_median"].fillna(global_med, inplace=True)
            df["brand_pack_count"].fillna(0, inplace=True)
            df["brand_pack_median_log"] = np.log1p(df["brand_pack_median"])

        Xtr = build_anchor_feats(tr)
        Xva = build_anchor_feats(va)
        Xte = build_anchor_feats(tst)

        for nm in ("brand_pack_median_log","brand_pack_count"):
            Xtr[nm] = tr[nm].values
            Xva[nm] = va[nm].values
            Xte[nm] = tst[nm].values

        lgbm = lgb.LGBMRegressor(
            n_estimators=2000,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="mae",
            random_state=SEED,
            verbose=-1
        )
        print(f"  Fold {f}/{folds} - Training LightGBM...", flush=True)
        lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        anchor_oof[va_idx] = lgbm.predict(Xva)
        anchor_test += lgbm.predict(Xte) / folds

        mae = mean_absolute_error(yva, anchor_oof[va_idx])
        print(f"[Fold {f}] Anchor LGBM MAE(log-space): {mae:.4f}")
        del Xtr, Xva, Xte, tr, va, tst; gc.collect()

    # ---------------- Stacker -----------------
    print("Stacking meta-learner (Ridge)...")
    base_oof = np.vstack([ridge_oof, cat_oof, anchor_oof]).T
    base_test = np.vstack([ridge_test, cat_test, anchor_test]).T

    meta = Ridge(alpha=1.0, random_state=SEED)
    meta.fit(base_oof, y_log.values)
    pred_log_test = meta.predict(base_test)

    # Final positivity constraint
    pred_price = np.expm1(pred_log_test)
    pred_price = np.maximum(pred_price, 0.01)

    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": pred_price.astype(float)})
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(sub)} rows.")

if __name__ == "__main__":
    main()
