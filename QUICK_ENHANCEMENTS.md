# Quick Enhancement Summary

## TL;DR - How to Enhance Ensemble Quality

### The Ensemble Power Hierarchy (from most to least impact):

```
1. Model Diversity (Add More Models)           ⭐⭐⭐ +2-5% improvement
2. Feature Engineering (Better Features)       ⭐⭐⭐ +3-7% improvement  
3. Hyperparameter Tuning                       ⭐⭐  +1-3% improvement
4. Better Stacking Strategy                    ⭐⭐  +0.5-2% improvement
5. More Cross-Validation Folds                 ⭐   +0.5-1% improvement
6. Increase max_tfidf                          ⭐   +0.5-1% improvement
```

---

## 🚀 FASTEST Path to Better Results (30 minutes)

### Step 1: Add XGBoost (Best ROI)
```bash
# Install if not already installed
pip install xgboost
```

Add this code block to `test.py` after line 384 (after CatBoost):

```python
# ============= ADD XGBOOST =============
import xgboost as xgb

print("\nTraining XGBoost...")
xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))

for f, (tr_idx, va_idx) in enumerate(skf.split(train, bins), 1):
    trX, vaX = X_emb[tr_idx], X_emb[va_idx]
    ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    print(f"  Fold {f}/{folds} - Training XGBoost...", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        tree_method='gpu_hist',  # Uses A100 GPU!
        random_state=SEED
    )
    model.fit(
        trX, ytr, 
        eval_set=[(vaX, yva)], 
        early_stopping_rounds=50,
        verbose=False
    )
    xgb_oof[va_idx] = model.predict(vaX)
    xgb_test += model.predict(T_emb) / folds
    
    mae = mean_absolute_error(yva, xgb_oof[va_idx])
    print(f"[Fold {f}] XGBoost MAE(log-space): {mae:.4f}")
    del trX, vaX; gc.collect()
```

Then update the stacker at line 444:
```python
# OLD:
base_oof = np.vstack([ridge_oof, cat_oof, anchor_oof]).T
base_test = np.vstack([ridge_test, cat_test, anchor_test]).T

# NEW:
base_oof = np.vstack([ridge_oof, cat_oof, anchor_oof, xgb_oof]).T
base_test = np.vstack([ridge_test, cat_test, anchor_test, xgb_test]).T
```

**Time:** 15 minutes | **Gain:** +2-3% SMAPE

### Step 2: Add Enhanced Features

Add this after line 253 (after basic feature engineering):

```python
# ============= ENHANCED FEATURES =============
for df in (train, test):
    # Price/discount indicators
    df["has_price"] = df["catalog_content"].str.contains(r'price|mrp|₹|\$|rs\.', case=False).astype(int)
    df["has_discount"] = df["catalog_content"].str.contains(r'off|discount|save|deal|offer', case=False).astype(int)
    
    # Category detection
    df["is_food"] = df["catalog_content"].str.contains(r'oil|rice|flour|dal|masala|spice|food|grain', case=False).astype(int)
    df["is_electronics"] = df["catalog_content"].str.contains(r'phone|mobile|laptop|cable|charger|electronic|adapter', case=False).astype(int)
    df["is_beauty"] = df["catalog_content"].str.contains(r'cream|lotion|shampoo|soap|perfume|makeup|beauty', case=False).astype(int)
    df["is_home"] = df["catalog_content"].str.contains(r'towel|sheet|blanket|pillow|curtain|mat|home', case=False).astype(int)
    
    # Quantity features
    df["unit_per_pack"] = df["unit_base"] / np.maximum(df["pack"], 1)
    df["total_volume"] = df["unit_base"] * df["pack"]
    df["is_bulk"] = (df["pack"] > 5).astype(int)
    
    # Text quality features
    df["capital_ratio"] = df["catalog_content"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x)))
    df["digit_ratio"] = df["catalog_content"].apply(lambda x: sum(1 for c in x if c.isdigit()) / max(1, len(x)))
    df["has_brand"] = (df["brand"] != "__unknown__").astype(int)
```

Then update `tab_cols` at line 354:
```python
# OLD:
tab_cols = ["pack","unit_base","len_chars","len_words","has_digits"]

# NEW:
tab_cols = [
    "pack", "unit_base", "len_chars", "len_words", "has_digits",
    "has_price", "has_discount", "is_food", "is_electronics", 
    "is_beauty", "is_home", "unit_per_pack", "total_volume",
    "is_bulk", "capital_ratio", "digit_ratio", "has_brand"
]
```

**Time:** 10 minutes | **Gain:** +2-4% SMAPE

### Step 3: Use GPU Acceleration + Increase max_tfidf

```bash
python3 test.py \
    --max_tfidf 300000 \
    --max_train 75000 \
    --max_test 75000 \
    --folds 5
```

Enable GPU for CatBoost - modify line 369:
```python
# OLD:
model = CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.035,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=SEED,
    verbose=False
)

# NEW:
model = CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.035,
    loss_function="MAE",
    eval_metric="MAE",
    task_type='GPU',         # GPU acceleration
    devices='0',
    random_seed=SEED,
    verbose=False
)
```

**Time:** 2 minutes | **Gain:** +1% SMAPE + 3x faster

---

## 📊 Expected Results

### Before Enhancements:
```
[Fold 1] Ridge MAE(log-space): 0.4200
[Fold 1] CatBoost MAE(log-space): 0.3800
[Fold 1] Anchor LGBM MAE(log-space): 0.3900

Final SMAPE: ~35-40%
```

### After Quick Enhancements:
```
[Fold 1] Ridge MAE(log-space): 0.4000    (-0.02)
[Fold 1] CatBoost MAE(log-space): 0.3500  (-0.03)
[Fold 1] Anchor LGBM MAE(log-space): 0.3700 (-0.02)
[Fold 1] XGBoost MAE(log-space): 0.3600   (NEW!)

Final SMAPE: ~31-35% (10-15% improvement!)
```

---

## 🎯 Run Commands

### Quick Test (10 minutes):
```bash
python3 test.py \
    --max_train 5000 \
    --max_test 500 \
    --folds 3 \
    --max_tfidf 300000 \
    --out_csv test_enhanced_quick.csv
```

### Full Run (2-3 hours):
```bash
python3 test.py \
    --max_train 75000 \
    --max_test 75000 \
    --folds 5 \
    --max_tfidf 300000 \
    --out_csv test_enhanced_full.csv
```

### Competition Run (4-5 hours) - Maximum Quality:
```bash
python3 test.py \
    --max_train 75000 \
    --max_test 75000 \
    --folds 10 \
    --max_tfidf 500000 \
    --out_csv submission.csv \
    2>&1 | tee training_final.log
```

---

## 🔧 Advanced Enhancements (If You Have More Time)

### 1. Add ExtraTrees (Fast)
```python
from sklearn.ensemble import ExtraTreesRegressor

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
    
# Add to stacker
base_oof = np.vstack([ridge_oof, cat_oof, anchor_oof, xgb_oof, et_oof]).T
base_test = np.vstack([ridge_test, cat_test, anchor_test, xgb_test, et_test]).T
```

### 2. Hyperparameter Tuning with Optuna

```python
pip install optuna

# Run this separately to find best params
import optuna

def objective(trial):
    depth = trial.suggest_int('depth', 4, 10)
    lr = trial.suggest_float('learning_rate', 0.01, 0.1)
    
    scores = []
    for tr_idx, va_idx in skf.split(train, bins):
        model = CatBoostRegressor(
            iterations=1000,
            depth=depth,
            learning_rate=lr,
            task_type='GPU',
            verbose=False
        )
        model.fit(X_emb[tr_idx], y_log.iloc[tr_idx])
        pred = model.predict(X_emb[va_idx])
        mae = mean_absolute_error(y_log.iloc[va_idx], pred)
        scores.append(mae)
    
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")

# Use best params in your training
```

### 3. Better Meta-Learner

```python
# Instead of Ridge, use LightGBM for stacking
meta = lgb.LGBMRegressor(
    n_estimators=500,
    num_leaves=32,
    learning_rate=0.05,
    device='gpu',
    random_state=SEED
)
meta.fit(base_oof, y_log.values)
pred_log_test = meta.predict(base_test)
```

---

## 📝 Checklist

Before your final submission:

- [ ] Added XGBoost model
- [ ] Added enhanced features (10+ new features)
- [ ] Increased max_tfidf to 300K+
- [ ] Enabled GPU acceleration
- [ ] Used 5+ folds
- [ ] Trained on full 75K dataset
- [ ] Validated output format matches sample
- [ ] No missing predictions
- [ ] All prices are positive
- [ ] Logged training details
- [ ] Saved model predictions for analysis

---

## 🎓 Key Insights

### Why Ensemble Works:
1. **Diversity** - Each model sees data differently:
   - Ridge: Looks at word frequencies
   - CatBoost: Learns complex embeddings
   - LightGBM: Focuses on brand/pack patterns
   - XGBoost: Different tree-building strategy

2. **Complementary Strengths**:
   - Ridge: Fast, handles high-dim text well
   - CatBoost: Best for embeddings, handles missing data
   - LightGBM: Great with categorical features
   - XGBoost: Robust, different regularization

3. **Error Averaging**:
   - Individual models make different mistakes
   - Ensemble averages out the errors
   - Result: More stable, accurate predictions

### What Makes Biggest Difference:
1. **More diverse models** > More of same model type
2. **Better features** > More training data
3. **Good cross-validation** > Perfect hyperparameters
4. **Understanding data** > Complex architectures

---

## 💡 Pro Tips

1. **Always test on subset first**
   ```bash
   # Test with 5K samples before full 75K run
   python3 test.py --max_train 5000 --max_test 500
   ```

2. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save intermediate results**
   - Save OOF predictions: `np.save('oof_preds.npy', base_oof)`
   - Analyze which model performs best on which samples

4. **Check feature importance**
   ```python
   # For tree models
   importance = model.feature_importances_
   print(f"Top features: {sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]}")
   ```

5. **Ensemble of ensembles**
   - Run script 3 times with different seeds
   - Average the 3 outputs: `(pred1 + pred2 + pred3) / 3`
   - Often gives +0.5-1% improvement!

---

## 🏆 Competition Strategy

### Day 1-2: Quick Setup
- Get baseline running
- Add XGBoost
- Add basic features

### Day 3-4: Enhancement
- Feature engineering
- Hyperparameter tuning
- Experiment with different combinations

### Day 5-6: Refinement
- Add more models (ExtraTrees, NN)
- Better stacking
- Ensemble multiple runs

### Final Day: Polish
- Full 75K training with 10 folds
- Multiple seeds
- Average best 3-5 runs
- Validate format thoroughly

**Good luck! 🚀**

