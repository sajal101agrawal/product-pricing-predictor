# Ensemble Model Enhancement Guide

## How to Improve Ridge + CatBoost + LightGBM Ensemble

This guide provides practical strategies to enhance your multimodal ensemble model quality.

---

## Table of Contents
1. [Add More Base Models](#1-add-more-base-models)
2. [Hyperparameter Tuning](#2-hyperparameter-tuning)
3. [Advanced Feature Engineering](#3-advanced-feature-engineering)
4. [Better Stacking Strategies](#4-better-stacking-strategies)
5. [Improved Cross-Validation](#5-improved-cross-validation)
6. [Data Quality Improvements](#6-data-quality-improvements)
7. [Model-Specific Tips](#7-model-specific-tips)
8. [Quick Wins](#8-quick-wins)

---

## 1. Add More Base Models ⭐⭐⭐

### Current: 3 Models
- Ridge (TF-IDF text)
- CatBoost (embeddings)
- LightGBM (brand/pack anchor)

### Enhanced: 5-7 Models

#### Add XGBoost
```python
import xgboost as xgb

# Add this after CatBoost training (around line 385)
print("Training XGBoost...")
xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))

for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    trX, vaX = X_emb[tr_idx], X_emb[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        tree_method='gpu_hist',  # Use GPU on A100!
        random_state=SEED
    )
    model.fit(trX, ytr, eval_set=[(vaX, yva)], early_stopping_rounds=50, verbose=False)
    xgb_oof[va_idx] = model.predict(vaX)
    xgb_test += model.predict(T_emb) / folds
    
    mae = mean_absolute_error(yva, xgb_oof[va_idx])
    print(f"[Fold {f}] XGBoost MAE(log-space): {mae:.4f}")
```

#### Add ExtraTrees (Fast & Different)
```python
from sklearn.ensemble import ExtraTreesRegressor

print("Training ExtraTrees...")
et_oof = np.zeros(len(train))
et_test = np.zeros(len(test))

for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    trX, vaX = X_emb[tr_idx], X_emb[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=SEED
    )
    model.fit(trX, ytr)
    et_oof[va_idx] = model.predict(vaX)
    et_test += model.predict(T_emb) / folds
    
    mae = mean_absolute_error(yva, et_oof[va_idx])
    print(f"[Fold {f}] ExtraTrees MAE(log-space): {mae:.4f}")
```

#### Add Neural Network
```python
from sklearn.neural_network import MLPRegressor

print("Training Neural Network...")
nn_oof = np.zeros(len(train))
nn_test = np.zeros(len(test))

for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    trX, vaX = X_emb[tr_idx], X_emb[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    # Normalize for neural network
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    trX_scaled = scaler.fit_transform(trX)
    vaX_scaled = scaler.transform(vaX)
    test_scaled = scaler.transform(T_emb)
    
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=SEED
    )
    model.fit(trX_scaled, ytr)
    nn_oof[va_idx] = model.predict(vaX_scaled)
    nn_test += model.predict(test_scaled) / folds
    
    mae = mean_absolute_error(yva, nn_oof[va_idx])
    print(f"[Fold {f}] Neural Net MAE(log-space): {mae:.4f}")
```

#### Update Stacker with All Models
```python
# Instead of 3 models, now use 5-6
base_oof = np.vstack([
    ridge_oof, 
    cat_oof, 
    anchor_oof,
    xgb_oof,      # NEW
    et_oof,       # NEW
    nn_oof        # NEW
]).T

base_test = np.vstack([
    ridge_test, 
    cat_test, 
    anchor_test,
    xgb_test,     # NEW
    et_test,      # NEW
    nn_test       # NEW
]).T
```

**Expected Improvement:** +2-5% better SMAPE

---

## 2. Hyperparameter Tuning ⭐⭐

### Current Parameters are Generic - Tune Them!

#### Option A: Manual Grid Search (Quick)

```python
# Test different CatBoost depths
for depth in [4, 6, 8]:
    for lr in [0.02, 0.035, 0.05]:
        model = CatBoostRegressor(
            iterations=2000,
            depth=depth,
            learning_rate=lr,
            loss_function="MAE",
            random_seed=SEED,
            verbose=False
        )
        # Train and evaluate...
```

#### Option B: Optuna (Automated) ⭐ Recommended

```python
import optuna

def objective(trial):
    # CatBoost hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': SEED,
        'verbose': False
    }
    
    # Cross-validation
    scores = []
    for tr_idx, va_idx in skf.split(train, bins):
        trX, vaX = X_emb[tr_idx], X_emb[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(trX, ytr, eval_set=(vaX, yva), use_best_model=True)
        pred = model.predict(vaX)
        mae = mean_absolute_error(yva, pred)
        scores.append(mae)
    
    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

**Expected Improvement:** +1-3% better SMAPE

---

## 3. Advanced Feature Engineering ⭐⭐⭐

### Add More Features to Improve All Models

#### Price-Related Features
```python
# Add these after line 253 (after basic features)

# Numeric extractions
df["price_numbers"] = df["catalog_content"].str.findall(r'\d+').apply(lambda x: np.mean([int(i) for i in x]) if x else 0)
df["has_mrp"] = df["catalog_content"].str.contains(r'mrp|price|₹|\$', case=False).astype(int)
df["has_discount"] = df["catalog_content"].str.contains(r'off|discount|save|deal', case=False).astype(int)

# Product category keywords
category_keywords = {
    'electronics': r'phone|laptop|tablet|headphone|watch|charger',
    'food': r'oil|rice|flour|snack|biscuit|cookie|chocolate',
    'beauty': r'cream|lotion|shampoo|soap|perfume|makeup',
    'home': r'towel|sheet|blanket|pillow|curtain|mat'
}

for cat, pattern in category_keywords.items():
    df[f"is_{cat}"] = df["catalog_content"].str.contains(pattern, case=False).astype(int)

# Text statistics
df["word_entropy"] = df["catalog_content"].apply(lambda x: len(set(x.split())) / max(1, len(x.split())))
df["avg_word_len"] = df["catalog_content"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)
df["has_brand_name"] = (df["brand"] != "__unknown__").astype(int)

# Unit per pack ratio
df["unit_per_pack"] = df["unit_base"] / df["pack"]
```

#### Image-Based Features (If Images Available)
```python
from PIL import ImageStat

def extract_image_stats(img_path):
    """Extract color/brightness statistics from image"""
    try:
        img = Image.open(img_path).convert("RGB")
        stats = ImageStat.Stat(img)
        return {
            'brightness': np.mean(stats.mean),
            'contrast': np.mean(stats.stddev),
            'dominant_color': max(stats.mean)  # Dominant channel
        }
    except:
        return {'brightness': 0, 'contrast': 0, 'dominant_color': 0}

# Add after image paths inference
if enable_images:
    train['img_brightness'] = [extract_image_stats(p)['brightness'] if p else 0 for p in img_paths_train]
    train['img_contrast'] = [extract_image_stats(p)['contrast'] if p else 0 for p in img_paths_train]
    test['img_brightness'] = [extract_image_stats(p)['brightness'] if p else 0 for p in img_paths_test]
    test['img_contrast'] = [extract_image_stats(p)['contrast'] if p else 0 for p in img_paths_test]
```

#### Target Encoding
```python
from category_encoders import TargetEncoder

# Add brand target encoding
te = TargetEncoder()
for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    ytr = y_log.iloc[tr_idx]
    
    te.fit(tr[['brand']], ytr)
    train.loc[va_idx, 'brand_target_enc'] = te.transform(va[['brand']])['brand']
    test['brand_target_enc'] = te.transform(test[['brand']])['brand']
```

**Expected Improvement:** +3-7% better SMAPE

---

## 4. Better Stacking Strategies ⭐⭐

### Current: Simple Ridge Meta-Learner

### Enhancement Options:

#### Option A: LightGBM as Meta-Learner
```python
# Instead of Ridge for stacking, use LightGBM
meta = lgb.LGBMRegressor(
    n_estimators=500,
    num_leaves=32,
    learning_rate=0.05,
    random_state=SEED
)
meta.fit(base_oof, y_log.values)
pred_log_test = meta.predict(base_test)
```

#### Option B: Weighted Average (Learn Weights)
```python
from scipy.optimize import minimize

def weighted_mae(weights):
    weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
    pred = base_oof @ weights
    return mean_absolute_error(y_log.values, pred)

# Find optimal weights
result = minimize(weighted_mae, x0=np.ones(base_oof.shape[1]) / base_oof.shape[1], method='Nelder-Mead')
optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))

print(f"Optimal weights: {optimal_weights}")
pred_log_test = base_test @ optimal_weights
```

#### Option C: Two-Level Stacking
```python
# Level 1: Train diverse models
# Level 2: Train another set of models on Level 1 predictions
# Level 3: Final meta-learner

# This is advanced - only if you have time
```

**Expected Improvement:** +0.5-2% better SMAPE

---

## 5. Improved Cross-Validation ⭐

### Current: 5-fold Stratified CV

### Enhancements:

#### Option A: More Folds
```bash
# Use 10 folds for more stable predictions
python3 test.py --folds 10
```

#### Option B: Repeated K-Fold
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
# This gives 10 models (5 folds × 2 repeats) - more robust but slower
```

#### Option C: Group-Based CV (By Brand)
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = train['brand'].values  # Ensure same brand doesn't appear in both train/val

for f, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups), 1):
    # Train models...
```

**Expected Improvement:** +0.5-1% better generalization

---

## 6. Data Quality Improvements ⭐⭐

### Clean and Augment Data

#### Remove Outliers
```python
# Before training, identify and handle outliers
from scipy import stats

z_scores = np.abs(stats.zscore(y))
outlier_mask = z_scores > 3

print(f"Found {outlier_mask.sum()} outliers")

# Option 1: Remove outliers
train_clean = train[~outlier_mask].reset_index(drop=True)

# Option 2: Cap outliers
y_capped = y.copy()
lower, upper = y.quantile([0.01, 0.99])
y_capped = np.clip(y_capped, lower, upper)
```

#### Handle Missing Images
```python
# Download missing images or use placeholder embeddings
# Train a model to predict image embeddings from text for missing images

from sklearn.linear_model import Ridge as RidgeRegressor

# Train text -> image embedding predictor
mask_has_img = train['has_image'].values
img_predictor = RidgeRegressor()
img_predictor.fit(train_text_emb[mask_has_img], train_img_emb[mask_has_img])

# Predict missing image embeddings
train_img_emb[~mask_has_img] = img_predictor.predict(train_text_emb[~mask_has_img])
```

#### Text Cleaning
```python
import re

def advanced_text_clean(text):
    # Remove HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    # Expand contractions
    text = text.replace("'", "'")  # Normalize quotes
    return text.strip()

train["catalog_content"] = train["catalog_content"].apply(advanced_text_clean)
test["catalog_content"] = test["catalog_content"].apply(advanced_text_clean)
```

**Expected Improvement:** +1-3% better SMAPE

---

## 7. Model-Specific Tips

### Ridge (TF-IDF)
- ✅ Increase `max_tfidf` to 250,000-500,000
- ✅ Try `ngram_range=(1,3)` for trigrams
- ✅ Tune alpha: try [0.5, 1.0, 1.5, 2.0, 3.0]
- ✅ Add character-level n-grams: `analyzer='char', ngram_range=(3,5)`

### CatBoost
- ✅ Use GPU: `task_type='GPU'` on A100
- ✅ Increase iterations to 3000-5000
- ✅ Try different loss functions: 'MAE', 'RMSE', 'Huber'
- ✅ Enable categorical features if you have them
- ✅ Use `bootstrap_type='Bayesian'` for better regularization

### LightGBM
- ✅ Use GPU: `device='gpu'` on A100
- ✅ Try `boosting_type='dart'` for better regularization
- ✅ Increase `num_leaves` to 128-256 for full dataset
- ✅ Use `feature_fraction=0.7` for more diversity
- ✅ Try `extra_trees=True` for random splits

### XGBoost (if added)
- ✅ Use `tree_method='gpu_hist'` on A100
- ✅ Try `max_depth` of 8-12
- ✅ Use `reg_alpha` and `reg_lambda` for regularization

---

## 8. Quick Wins (Implement These First!) ⭐⭐⭐

### Priority Order:

#### 1. Add XGBoost (15 minutes)
```bash
pip install xgboost
# Add XGBoost code to test.py
```
**Expected gain:** +1-3% SMAPE

#### 2. Increase max_tfidf (1 minute)
```bash
python3 test.py --max_tfidf 300000
```
**Expected gain:** +0.5-1% SMAPE

#### 3. Use GPU for Boosting Models (2 minutes)
```python
# CatBoost
task_type='GPU', devices='0'

# LightGBM  
device='gpu', gpu_platform_id=0, gpu_device_id=0
```
**Expected gain:** 3-5x faster training

#### 4. Add More Feature Engineering (30 minutes)
- Add price-related keywords
- Add category detection
- Add unit_per_pack ratio

**Expected gain:** +2-4% SMAPE

#### 5. Tune Hyperparameters with Optuna (1-2 hours)
```bash
pip install optuna
# Run 50-100 trials
```
**Expected gain:** +1-3% SMAPE

#### 6. Increase Folds (5 minutes)
```bash
python3 test.py --folds 10
```
**Expected gain:** +0.5-1% SMAPE (more stability)

---

## Complete Enhanced Script Example

Here's a minimal example adding XGBoost + better features:

```python
# After line 253 in test.py, add:

# === Enhanced Features ===
for df in (train, test):
    # Price indicators
    df["has_price_word"] = df["catalog_content"].str.contains(r'price|mrp|₹|\$', case=False).astype(int)
    df["has_discount"] = df["catalog_content"].str.contains(r'off|discount|save', case=False).astype(int)
    
    # Category detection
    df["is_food"] = df["catalog_content"].str.contains(r'oil|rice|flour|food', case=False).astype(int)
    df["is_electronics"] = df["catalog_content"].str.contains(r'phone|laptop|cable|charger', case=False).astype(int)
    
    # Enhanced pack features
    df["unit_per_pack"] = df["unit_base"] / np.maximum(df["pack"], 1)
    df["total_volume"] = df["unit_base"] * df["pack"]

# Update tab_cols (line 354)
tab_cols = ["pack", "unit_base", "len_chars", "len_words", "has_digits",
            "has_price_word", "has_discount", "is_food", "is_electronics",
            "unit_per_pack", "total_volume"]

# After CatBoost training (around line 385), add XGBoost:
import xgboost as xgb

print("\nTraining XGBoost...")
xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))

for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    trX, vaX = X_emb[tr_idx], X_emb[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        tree_method='gpu_hist',  # GPU acceleration
        random_state=SEED,
        early_stopping_rounds=50
    )
    model.fit(trX, ytr, eval_set=[(vaX, yva)], verbose=False)
    xgb_oof[va_idx] = model.predict(vaX)
    xgb_test += model.predict(T_emb) / folds
    
    mae = mean_absolute_error(yva, xgb_oof[va_idx])
    print(f"[Fold {f}] XGBoost MAE(log-space): {mae:.4f}")

# Update stacker (line 444)
base_oof = np.vstack([ridge_oof, cat_oof, anchor_oof, xgb_oof]).T
base_test = np.vstack([ridge_test, cat_test, anchor_test, xgb_test]).T
```

---

## Expected Total Improvement

| Enhancement | Time | SMAPE Improvement | Cumulative |
|-------------|------|-------------------|------------|
| Baseline | - | - | 100% |
| Add XGBoost | 15 min | -2% | 98% |
| Increase max_tfidf | 1 min | -1% | 97% |
| Feature engineering | 30 min | -3% | 94% |
| Hyperparameter tuning | 2 hrs | -2% | 92% |
| Add ExtraTrees | 20 min | -1% | 91% |
| Better stacking | 30 min | -1% | 90% |
| **Total** | **~4 hrs** | **-10%** | **90%** |

*Note: Lower SMAPE is better. 90% means 10% improvement.*

---

## Testing Your Improvements

Always validate improvements:

```bash
# Baseline
python3 test.py --max_train 10000 --folds 3 --out_csv baseline.csv

# With enhancement
python3 test.py --max_train 10000 --folds 3 --out_csv enhanced.csv --max_tfidf 300000

# Compare validation MAE in logs
```

Look for consistent improvement across all folds.

---

## Final Recommendations

### For Quick Competition Win (1-2 hours):
1. ✅ Add XGBoost
2. ✅ Increase max_tfidf to 300K
3. ✅ Add 5-10 new features
4. ✅ Use GPU acceleration

### For Maximum Performance (4-8 hours):
1. ✅ All of the above
2. ✅ Add ExtraTrees + Neural Network
3. ✅ Hyperparameter tuning with Optuna
4. ✅ Advanced feature engineering (20+ features)
5. ✅ Better stacking (weighted average or LightGBM meta)
6. ✅ 10-fold CV with multiple repeats

### Don't Bother With:
- ❌ Complex deep learning architectures (license issues + time)
- ❌ Too many models (>7 base models has diminishing returns)
- ❌ Over-tuning on validation set (risk overfitting)

---

**Good luck with your competition! 🚀**

