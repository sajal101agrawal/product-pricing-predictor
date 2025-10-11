#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Product Pricing — v2.5 (Hybrid: v2 improvements + v3 fixes)

Key improvements over v2:
- Added char-level TF-IDF (captures brands/SKUs/patterns)
- Gated meta-learner (separate models for small vs large prices)
- Isotonic calibration OFF by default (prevented value collapse)
- Kept per-unit normalization
- Kept target encoding
- Dropped weak branches (KNN, Anchor) - can re-enable if needed
- Text-heavy approach (what works best)
"""
import os, re, gc, math, random, warnings, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def build_argparser():
    ap = argparse.ArgumentParser(description="Smart Product Pricing v2.5 - Hybrid")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--images_dir", default="images")
    ap.add_argument("--out_csv", default="test_out.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--use_images", action="store_true")
    ap.add_argument("--disable_images", action="store_true")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--max_tfidf", type=int, default=200000)
    ap.add_argument("--max_char_tfidf", type=int, default=150000)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--per_unit", action="store_true", help="Use per-unit normalization (RECOMMENDED)")
    ap.add_argument("--small_thresh", type=float, default=5.0, help="Threshold for small vs large prices")
    ap.add_argument("--enable_isotonic", action="store_true", help="Enable isotonic calibration (not recommended)")
    return ap

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0

def stratify_bins(y, q=20):
    ylog = np.log1p(y.astype(float))
    try:
        return pd.qcut(ylog, q=q, labels=False, duplicates="drop")
    except:
        return pd.cut(ylog, bins=q, labels=False, include_lowest=True, duplicates="drop")

def normalize_text(s):
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_brand(title):
    if not title: return "__unknown__"
    parts = re.split(r'[\-\|\:\–]', title, maxsplit=1)
    head = parts[0].strip()
    toks = head.split()[:3]
    if not toks: return "__unknown__"
    if all(t.isdigit() for t in toks): return "__unknown__"
    return " ".join(toks)

UNIT_MAP = {"ml":1e-3, "l":1.0, "g":1e-3, "kg":1.0}
UNIT_PATTERN = r'(\d+(?:\.\d+)?)\s*(ml|l|g|kg)\b'
PACK_PATTERN = r'(?:pack\s*of\s*|x\s*|\u00D7\s*|\(\s*)(\d{1,3})(?:\s*(?:pcs|pieces|units)?\s*\))?'

def parse_units(text):
    if not isinstance(text, str): return 0.0
    total = 0.0
    for m in re.finditer(UNIT_PATTERN, text):
        try:
            val = float(m.group(1)); unit = m.group(2)
            factor = UNIT_MAP.get(unit, None)
            if factor is not None: total += val * factor
        except: pass
    return float(total)

def parse_pack(text):
    if not isinstance(text, str): return 1
    m = re.search(PACK_PATTERN, text)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 200: return n
        except: pass
    return 1

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
        sid_base = re.sub(r'[^a-zA-Z0-9]', '_', str(sid))
        done = False
        for ext in (".jpg",".jpeg",".png",".webp"):
            fp = images_dir / f"{sid_base}{ext}"
            if fp.exists():
                fps.append(str(fp)); done=True; break
        if not done:
            base = os.path.basename(str(url).split("?")[0])
            fp2 = images_dir / base
            fps.append(str(fp2) if fp2.exists() else None)
    return fps

import torch
from sentence_transformers import SentenceTransformer
import timm, torchvision.transforms as T

def device(allow_cuda=True):
    if allow_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

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
        bufs, out = [], []
        for im in pil_list:
            if im is None: out.append(None)
            else:
                bufs.append(self.tfm(im))
                if len(bufs)==batch:
                    b = torch.stack(bufs).float().to(self.dev)
                    feats = self.model(b).detach().cpu().numpy()
                    out.extend([f for f in feats]); bufs=[]
        if bufs:
            b = torch.stack(bufs).float().to(self.dev)
            feats = self.model(b).detach().cpu().numpy()
            out.extend([f for f in feats])
        dim = len(out[0]) if (len(out)>0 and out[0] is not None) else 384
        Z = []
        for e in out:
            Z.append(np.zeros(dim, dtype=float) if e is None else e.astype(float))
        return np.vstack(Z)

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostRegressor

def kfold_target_encode(series, y, fold_splits, prior=None):
    """Leak-safe target encoding with smoothing"""
    y = np.asarray(y, dtype=float)
    n = len(series)
    oof = np.zeros(n, dtype=float)
    global_mean = float(np.mean(y)) if prior is None else float(prior)
    
    for tr_idx, va_idx in fold_splits:
        s_tr, s_va = series.iloc[tr_idx], series.iloc[va_idx]
        y_tr = y[tr_idx]
        df_tr = pd.DataFrame({"k": s_tr.values, "y": y_tr})
        stats = df_tr.groupby("k")["y"].mean()
        cnts = df_tr.groupby("k")["y"].size()
        alpha = 10.0
        smooth = (cnts * stats + alpha * global_mean) / (cnts + alpha)
        enc = s_va.map(smooth).fillna(global_mean).values
        oof[va_idx] = enc
    
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
    
    # Suppress tokenizer warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)

    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    
    sample_test_path = data_dir / "sample_test.csv"
    if sample_test_path.exists() and args.max_test:
        print(f"Using sample_test.csv for faster iteration...")
        test = pd.read_csv(sample_test_path)
    else:
        test = pd.read_csv(data_dir / "test.csv")

    print(f"Original train size: {len(train)}, test size: {len(test)}")
    
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=SEED).reset_index(drop=True)
        print(f"Reduced train size to: {len(train)} samples")
    
    if args.max_test and len(test) > args.max_test:
        test = test.sample(n=args.max_test, random_state=SEED).reset_index(drop=True)
        print(f"Reduced test size to: {len(test)} samples for fast iteration")

    assert "price" in train.columns

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

    # Per-unit target normalization
    def unit_factor(df):
        f = (df["pack"].clip(lower=1).astype(float) * df["unit_base"].clip(lower=1e-3).astype(float)).values
        f = np.where(df["unit_base"].values <= 0, df["pack"].clip(lower=1).astype(float), f)
        return f

    y = train["price"].astype(float).values
    factor_tr = unit_factor(train)
    factor_te = unit_factor(test)

    if args.per_unit:
        target = np.log1p(y / np.maximum(factor_tr, 1e-3))
        print("Using per-unit target normalization ✅")
    else:
        target = np.log1p(y)
        print("Using standard log1p target")
    
    folds = args.folds
    bins = stratify_bins(train["price"])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(train, bins))

    # ============================================================
    # TEXT FEATURES (Word + Char TF-IDF)
    # ============================================================
    print("\nBuilding TF-IDF features...")
    
    # Word-level TF-IDF
    print(f"Word TF-IDF (max_features={args.max_tfidf})...")
    word_tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_features=args.max_tfidf,
        strip_accents="unicode"
    )
    X_word = word_tfidf.fit_transform(train["catalog_content"])
    T_word = word_tfidf.transform(test["catalog_content"])
    
    # Char-level TF-IDF (NEW in v2.5!)
    print(f"Char TF-IDF (max_features={args.max_char_tfidf})...")
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        min_df=5,
        max_features=args.max_char_tfidf
    )
    X_char = char_tfidf.fit_transform(train["catalog_content"])
    T_char = char_tfidf.transform(test["catalog_content"])
    
    print(f"  Word TF-IDF shape: {X_word.shape}")
    print(f"  Char TF-IDF shape: {X_char.shape}")

    # ============================================================
    # EMBEDDINGS (Text + optional Images + Tabular)
    # ============================================================
    print("\nGenerating embeddings...")
    use_cuda = (not args.cpu_only)
    txt_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=str(device(allow_cuda=use_cuda)))

    def encode_text(texts, bs=512):
        embs = []
        for i in range(0, len(texts), bs):
            embs.append(txt_encoder.encode(texts[i:i+bs], normalize_embeddings=True))
        return np.vstack(embs).astype(np.float32)

    tr_txt = encode_text(train["catalog_content"].tolist())
    te_txt = encode_text(test["catalog_content"].tolist())

    # Images (optional)
    img_tr = infer_image_paths(train, images_dir)
    img_te = infer_image_paths(test, images_dir)
    n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
    found_ratio = n_found / max(1, len(img_tr)+len(img_te))
    
    enable_images = (not args.disable_images) and (args.use_images or found_ratio >= 0.2)

    if enable_images:
        print(f"Images used (~{found_ratio*100:.1f}% found)")
        vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
        def open_batch(paths):
            return [safe_image_open(p) for p in paths]
        tr_img = vit.encode(open_batch(img_tr), 64).astype(np.float32)
        te_img = vit.encode(open_batch(img_te), 64).astype(np.float32)
    else:
        print(f"Skip images (~{found_ratio*100:.1f}% found)")
        tr_img = np.zeros((len(train), 384), dtype=np.float32)
        te_img = np.zeros((len(test),  384), dtype=np.float32)

    if args.pca_dim and args.pca_dim > 0:
        print(f"PCA to {args.pca_dim} dims...")
        p_txt = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_txt)
        p_img = PCA(n_components=args.pca_dim, random_state=SEED).fit(tr_img)
        tr_txt = p_txt.transform(tr_txt); te_txt = p_txt.transform(te_txt)
        tr_img = p_img.transform(tr_img); te_img = p_img.transform(te_img)

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

    X_emb = np.hstack([tr_txt, tr_img, X_tab, oof_brand.reshape(-1,1), oof_bxpk.reshape(-1,1)]).astype(np.float32)
    T_emb = np.hstack([te_txt, te_img, T_tab, T_brand.reshape(-1,1), T_bxpk.reshape(-1,1)]).astype(np.float32)

    # ============================================================
    # BASE LEARNERS (Text-heavy approach)
    # ============================================================
    
    # Base A: Word TF-IDF + Ridge
    print("\n[Base A] Word TF-IDF + Ridge...")
    ridge_word_oof = np.zeros(len(train)); ridge_word_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        model = Ridge(alpha=1.2, random_state=SEED)
        model.fit(X_word[tr_idx], target[tr_idx])
        ridge_word_oof[va_idx] = model.predict(X_word[va_idx])
        ridge_word_test += model.predict(T_word) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], ridge_word_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base B: Char TF-IDF + Ridge (NEW!)
    print("\n[Base B] Char TF-IDF + Ridge...")
    ridge_char_oof = np.zeros(len(train)); ridge_char_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(X_char[tr_idx], target[tr_idx])
        ridge_char_oof[va_idx] = model.predict(X_char[va_idx])
        ridge_char_test += model.predict(T_char) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], ridge_char_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base C: Word TF-IDF + LightGBM
    print("\n[Base C] Word TF-IDF + LightGBM...")
    lgbm_word_oof = np.zeros(len(train)); lgbm_word_test = np.zeros(len(test))
    lgbm_params = dict(n_estimators=3000, num_leaves=256, learning_rate=0.03, 
                       subsample=0.8, colsample_bytree=0.6, objective="mae", 
                       random_state=SEED, verbose=-1)
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        lgbm = lgb.LGBMRegressor(**lgbm_params)
        lgbm.fit(X_word[tr_idx], target[tr_idx], 
                 eval_set=[(X_word[va_idx], target[va_idx])], 
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        lgbm_word_oof[va_idx] = lgbm.predict(X_word[va_idx])
        lgbm_word_test += lgbm.predict(T_word) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], lgbm_word_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # Base D: Embeddings + CatBoost (lighter version)
    print("\n[Base D] Embeddings + CatBoost...")
    cat_oof = np.zeros(len(train)); cat_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        model = CatBoostRegressor(iterations=1800, depth=6, learning_rate=0.04, 
                                   loss_function="MAE", eval_metric="MAE", 
                                   random_seed=SEED, verbose=False)
        model.fit(X_emb[tr_idx], target[tr_idx], 
                  eval_set=(X_emb[va_idx], target[va_idx]), use_best_model=True)
        cat_oof[va_idx] = model.predict(X_emb[va_idx])
        cat_test += model.predict(T_emb) / len(fold_splits)
        mae = mean_absolute_error(target[va_idx], cat_oof[va_idx])
        print(f"  Fold {fold} MAE: {mae:.4f}")
    gc.collect()

    # ============================================================
    # GATED META-LEARNER (Small vs Large prices)
    # ============================================================
    print("\n[Meta] Gated meta-learner...")
    
    # Identify small vs large prices
    small = (train["price"].values <= args.small_thresh).astype(int)
    print(f"  Small prices (≤${args.small_thresh}): {small.sum()} samples ({100*small.mean():.1f}%)")
    print(f"  Large prices (>${args.small_thresh}): {(1-small).sum()} samples ({100*(1-small).mean():.1f}%)")
    
    # Build meta features
    X_meta_oof = np.vstack([ridge_word_oof, ridge_char_oof, lgbm_word_oof, cat_oof]).T
    X_meta_test = np.vstack([ridge_word_test, ridge_char_test, lgbm_word_test, cat_test]).T
    
    # Train classifier to predict small vs large
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_meta_oof, small)
    p_small_oof = clf.predict_proba(X_meta_oof)[:,1]
    p_small_test = clf.predict_proba(X_meta_test)[:,1]
    
    # Train separate meta models
    if small.sum() > 10:  # Need enough samples
        meta_small = Ridge(alpha=1.0, random_state=SEED)
        meta_small.fit(X_meta_oof[small==1], target[small==1])
    else:
        meta_small = Ridge(alpha=1.0, random_state=SEED)
        meta_small.fit(X_meta_oof, target)  # Fallback
    
    if (1-small).sum() > 10:
        meta_large = Ridge(alpha=1.0, random_state=SEED)
        meta_large.fit(X_meta_oof[small==0], target[small==0])
    else:
        meta_large = Ridge(alpha=1.0, random_state=SEED)
        meta_large.fit(X_meta_oof, target)  # Fallback
    
    # Blend predictions
    oof_small = meta_small.predict(X_meta_oof)
    oof_large = meta_large.predict(X_meta_oof)
    meta_oof = p_small_oof * oof_small + (1 - p_small_oof) * oof_large
    
    test_small = meta_small.predict(X_meta_test)
    test_large = meta_large.predict(X_meta_test)
    meta_test = p_small_test * test_small + (1 - p_small_test) * test_large

    # Convert back from log space
    if args.per_unit:
        oof_price = np.expm1(meta_oof) * np.maximum(factor_tr, 1e-3)
        test_price = np.expm1(meta_test) * np.maximum(factor_te, 1e-3)
    else:
        oof_price = np.expm1(meta_oof)
        test_price = np.expm1(meta_test)

    # Optional isotonic calibration (not recommended)
    if args.enable_isotonic:
        print("Applying isotonic calibration...")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_price, train["price"].values.astype(float))
        test_price = iso.transform(test_price)
    
    # Clipping
    test_price = np.maximum(test_price, 0.01)
    hi_thresh = np.percentile(train["price"].values, 99.8)
    test_price = np.clip(test_price, 0.01, hi_thresh * 3.0)

    # Final SMAPE on OOF
    oof_smape = smape(train["price"].values, oof_price)
    print(f"\nOOF SMAPE: {oof_smape:.2f}%")
    
    # Save predictions
    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price.astype(float)})
    sub.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(sub)} rows.")

if __name__ == "__main__":
    main()

