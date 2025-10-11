#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Product Pricing — v3 Final (Data-Driven Optimization)

Based on empirical testing on YOUR data:
- Embeddings are strongest (0.59 MAE)
- Text models work best WITHOUT per-unit normalization
- Isotonic calibration causes value collapse
- Need better post-processing (priors, debiasing)

Key improvements:
✅ Better unit/pack parsing (handles more patterns)
✅ Brand×pack prior blending (guardrails)
✅ Global debiasing (fixes underprediction)
✅ ElasticNet meta-learner (better weighting)
✅ No isotonic (prevents value collapse)
✅ Gated approach (small vs large prices)
"""
import os, re, gc, math, random, warnings, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def build_argparser():
    ap = argparse.ArgumentParser(description="Smart Product Pricing v3 Final")
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
    ap.add_argument("--small_thresh", type=float, default=5.0)
    ap.add_argument("--prior_alpha", type=float, default=30.0, help="Blending weight for brand×pack priors")
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

# ============================================================
# IMPROVED UNIT PARSING
# ============================================================
UNIT_ALIASES = {
    # mass -> kg
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    # volume -> liters
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "cc": "ml",
    "l": "l", "ltr": "l", "ltrs": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "floz": "floz", "fl oz": "floz",
}

TO_BASE = {
    "g": 1e-3, "kg": 1.0, "lb": 0.45359237, "oz": 0.028349523125,   # -> kg
    "ml": 1e-3, "l": 1.0, "floz": 0.0295735295625,                   # -> liters
}

# patterns like: 3x200g, 200 g x 3, (3 x 200 g), 2*500ml, etc.
PACKED_QTY_PAT = re.compile(
    r'(?:(\d{1,3})\s*[x×*]\s*)?(\d+(?:\.\d+)?)\s*(g|gm|gms|gram|grams|kg|kgs|lb|lbs|pound|pounds|oz|ounce|ounces|ml|milliliter|milliliters|cc|l|ltr|ltrs|litre|litres|liter|liters|fl\s*oz|floz)\b'
)

def parse_units(text):
    """Returns total base quantity (kg for mass, liters for volume)"""
    if not isinstance(text, str): return 0.0
    total = 0.0
    for m in PACKED_QTY_PAT.finditer(text):
        mult_raw, val_raw, unit_raw = m.groups()
        mult = int(mult_raw) if mult_raw else 1
        unit_raw = unit_raw.replace(" ", "")  # "fl oz" -> "floz"
        unit_norm = UNIT_ALIASES.get(unit_raw, unit_raw)
        factor = TO_BASE.get(unit_norm)
        if factor is None: continue
        try:
            val = float(val_raw)
            total += mult * val * factor
        except: pass
    return float(total)

# Better pack parsing
PACK_PATTERN = re.compile(r'(?:pack\s*of\s*|x\s*|×\s*|\(\s*)\b(\d{1,3})\b(?:\s*(?:pcs?|pieces?|units?)\s*\)?)?')

def parse_pack(text):
    if not isinstance(text, str): return 1
    m = PACK_PATTERN.search(text)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 200: return n
        except: pass
    return 1

from PIL import Image
def safe_image_open(fp, size=(224,224)):
    try:
        return Image.open(fp).convert("RGB").resize(size)
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
        Z = [np.zeros(dim, dtype=float) if e is None else e.astype(float) for e in out]
        return np.vstack(Z)

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

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
        print(f"Reduced test size to: {len(test)} samples")

    assert "price" in train.columns

    # Feature engineering with improved parsing
    print("\nFeature engineering (improved unit/pack parsing)...")
    for df in (train, test):
        df["catalog_content"] = df["catalog_content"].fillna("").map(normalize_text)
        df["title"] = df["catalog_content"].str.split(".").str[0]
        df["brand"] = df["title"].map(extract_brand)
        df["pack"] = df["catalog_content"].map(parse_pack).astype(int)
        df["unit_base"] = df["catalog_content"].map(parse_units).astype(float)
        df["len_chars"] = df["catalog_content"].str.len().astype(int)
        df["len_words"] = df["catalog_content"].str.split().map(len).astype(int)
        df["has_digits"] = df["catalog_content"].str.contains(r'\d').astype(int)

    # Simple log target (no per-unit for text models based on empirical results)
    y = train["price"].astype(float).values
    target = np.log1p(y)
    print("Using standard log1p target (per-unit hurt text models on your data)")
    
    folds = args.folds
    bins = stratify_bins(train["price"])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(train, bins))

    # ============================================================
    # TEXT FEATURES
    # ============================================================
    print("\nBuilding TF-IDF features...")
    
    # Word-level TF-IDF
    word_tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=args.max_tfidf, strip_accents="unicode")
    X_word = word_tfidf.fit_transform(train["catalog_content"])
    T_word = word_tfidf.transform(test["catalog_content"])
    
    # Char-level TF-IDF
    char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=5, max_features=args.max_char_tfidf)
    X_char = char_tfidf.fit_transform(train["catalog_content"])
    T_char = char_tfidf.transform(test["catalog_content"])
    
    print(f"  Word TF-IDF: {X_word.shape}")
    print(f"  Char TF-IDF: {X_char.shape}")

    # ============================================================
    # EMBEDDINGS (Your strongest signal!)
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

    # Images
    img_tr = infer_image_paths(train, images_dir)
    img_te = infer_image_paths(test, images_dir)
    n_found = sum(1 for p in img_tr+img_te if p and os.path.exists(p))
    found_ratio = n_found / max(1, len(img_tr)+len(img_te))
    
    enable_images = (not args.disable_images) and (args.use_images or found_ratio >= 0.2)

    if enable_images:
        print(f"Images used (~{found_ratio*100:.1f}% found)")
        vit = ViTEncoder("vit_small_patch16_224", use_cuda=use_cuda)
        tr_img = vit.encode([safe_image_open(p) for p in img_tr], 64).astype(np.float32)
        te_img = vit.encode([safe_image_open(p) for p in img_te], 64).astype(np.float32)
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
    # BASE LEARNERS
    # ============================================================
    
    # Base A: Word TF-IDF + Ridge
    print("\n[Base A] Word TF-IDF + Ridge...")
    ridge_word_oof = np.zeros(len(train)); ridge_word_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        m = Ridge(alpha=1.2, random_state=SEED)
        m.fit(X_word[tr_idx], target[tr_idx])
        ridge_word_oof[va_idx] = m.predict(X_word[va_idx])
        ridge_word_test += m.predict(T_word) / len(fold_splits)
        print(f"  Fold {fold} MAE: {mean_absolute_error(target[va_idx], ridge_word_oof[va_idx]):.4f}")
    gc.collect()

    # Base B: Char TF-IDF + Ridge
    print("\n[Base B] Char TF-IDF + Ridge...")
    ridge_char_oof = np.zeros(len(train)); ridge_char_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(X_char[tr_idx], target[tr_idx])
        ridge_char_oof[va_idx] = m.predict(X_char[va_idx])
        ridge_char_test += m.predict(T_char) / len(fold_splits)
        print(f"  Fold {fold} MAE: {mean_absolute_error(target[va_idx], ridge_char_oof[va_idx]):.4f}")
    gc.collect()

    # Base C: Word TF-IDF + LightGBM
    print("\n[Base C] Word TF-IDF + LightGBM...")
    lgbm_word_oof = np.zeros(len(train)); lgbm_word_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        m = lgb.LGBMRegressor(n_estimators=3000, num_leaves=256, learning_rate=0.03, 
                              subsample=0.8, colsample_bytree=0.6, objective="mae", 
                              random_state=SEED, verbose=-1)
        m.fit(X_word[tr_idx], target[tr_idx], eval_set=[(X_word[va_idx], target[va_idx])], 
              callbacks=[lgb.early_stopping(100, verbose=False)])
        lgbm_word_oof[va_idx] = m.predict(X_word[va_idx])
        lgbm_word_test += m.predict(T_word) / len(fold_splits)
        print(f"  Fold {fold} MAE: {mean_absolute_error(target[va_idx], lgbm_word_oof[va_idx]):.4f}")
    gc.collect()

    # Base D: Embeddings + CatBoost (YOUR BEST MODEL!)
    print("\n[Base D] Embeddings + CatBoost (strongest on your data)...")
    cat_oof = np.zeros(len(train)); cat_test = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
        m = CatBoostRegressor(iterations=2000, depth=6, learning_rate=0.035, 
                             loss_function="MAE", eval_metric="MAE", 
                             random_seed=SEED, verbose=False)
        m.fit(X_emb[tr_idx], target[tr_idx], eval_set=(X_emb[va_idx], target[va_idx]), use_best_model=True)
        cat_oof[va_idx] = m.predict(X_emb[va_idx])
        cat_test += m.predict(T_emb) / len(fold_splits)
        print(f"  Fold {fold} MAE: {mean_absolute_error(target[va_idx], cat_oof[va_idx]):.4f}")
    gc.collect()

    # ============================================================
    # GATED META-LEARNER WITH ELASTICNET
    # ============================================================
    print("\n[Meta] Gated ElasticNet meta-learner...")
    
    # Identify small vs large
    small = (train["price"].values <= args.small_thresh).astype(int)
    print(f"  Small (≤${args.small_thresh}): {small.sum()} ({100*small.mean():.1f}%)")
    print(f"  Large (>${args.small_thresh}): {(1-small).sum()} ({100*(1-small).mean():.1f}%)")
    
    # Meta features
    X_meta_oof = np.vstack([ridge_word_oof, ridge_char_oof, lgbm_word_oof, cat_oof]).T
    X_meta_test = np.vstack([ridge_word_test, ridge_char_test, lgbm_word_test, cat_test]).T
    
    # Classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_meta_oof, small)
    p_small_oof = clf.predict_proba(X_meta_oof)[:,1]
    p_small_test = clf.predict_proba(X_meta_test)[:,1]
    
    # ElasticNet meta models (better than Ridge)
    if small.sum() > 10:
        meta_small = ElasticNet(alpha=0.001, l1_ratio=0.2, positive=True, random_state=SEED, max_iter=2000)
        meta_small.fit(X_meta_oof[small==1], target[small==1])
    else:
        meta_small = ElasticNet(alpha=0.001, l1_ratio=0.2, positive=True, random_state=SEED, max_iter=2000)
        meta_small.fit(X_meta_oof, target)
    
    if (1-small).sum() > 10:
        meta_large = ElasticNet(alpha=0.001, l1_ratio=0.2, positive=True, random_state=SEED, max_iter=2000)
        meta_large.fit(X_meta_oof[small==0], target[small==0])
    else:
        meta_large = ElasticNet(alpha=0.001, l1_ratio=0.2, positive=True, random_state=SEED, max_iter=2000)
        meta_large.fit(X_meta_oof, target)
    
    # Blend
    oof_small = meta_small.predict(X_meta_oof)
    oof_large = meta_large.predict(X_meta_oof)
    meta_oof = p_small_oof * oof_small + (1 - p_small_oof) * oof_large
    
    test_small = meta_small.predict(X_meta_test)
    test_large = meta_large.predict(X_meta_test)
    meta_test = p_small_test * test_small + (1 - p_small_test) * test_large

    # Convert to price space
    oof_price = np.expm1(meta_oof)
    test_price = np.expm1(meta_test)

    # ============================================================
    # POST-PROCESSING
    # ============================================================
    print("\nPost-processing...")
    
    # 1. Global debiasing (fixes underprediction)
    ratio = (train["price"].values.sum() + 1e-6) / (oof_price.sum() + 1e-6)
    print(f"  Global bias ratio: {ratio:.3f}")
    test_price *= ratio
    oof_price *= ratio
    
    # 2. Brand×pack prior blending (guardrails)
    print(f"  Applying brand×pack prior blending (alpha={args.prior_alpha})...")
    grp = train.groupby(["brand","pack"], observed=True)["price"].agg(["median","count"])
    grp = grp.rename(columns={"median":"brand_pack_median","count":"brand_pack_count"})
    tst = test.merge(grp, left_on=["brand","pack"], right_index=True, how="left")
    global_med = float(train["price"].median())
    tst["brand_pack_median"] = tst["brand_pack_median"].fillna(global_med)
    tst["brand_pack_count"] = tst["brand_pack_count"].fillna(0)
    
    # Blend weight
    w = tst["brand_pack_count"].clip(0, 200) / (tst["brand_pack_count"].clip(0, 200) + args.prior_alpha)
    test_price = w.values * test_price + (1 - w.values) * tst["brand_pack_median"].values
    
    # 3. Clamp to sensible band
    lo = 0.35 * tst["brand_pack_median"].values
    hi = 2.5 * tst["brand_pack_median"].values
    test_price = np.clip(test_price, lo, hi)
    
    # 4. Final clipping
    test_price = np.maximum(test_price, 0.01)
    hi_thresh = np.percentile(train["price"].values, 99.8)
    test_price = np.clip(test_price, 0.01, hi_thresh * 3.0)

    # Final SMAPE
    oof_smape = smape(train["price"].values, oof_price)
    print(f"\nOOF SMAPE: {oof_smape:.2f}%")
    
    # Save
    sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_price.astype(float)})
    sub.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(sub)} rows.")

if __name__ == "__main__":
    main()

