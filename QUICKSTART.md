# Quick Start Guide

## Product Pricing Predictor - Get Started in 5 Minutes

This guide helps you get started quickly on different environments.

---

## Choose Your Environment

### 🖥️ Ubuntu/Linux Local Setup
👉 See [UBUNTU_SETUP.md](UBUNTU_SETUP.md) for detailed installation instructions

**Quick commands:**
```bash
# Setup
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run sample code
python3 sample_code.py
```

---

### 🚀 Nvidia A100 GPU Setup
👉 See [A100_GPU_SETUP.md](A100_GPU_SETUP.md) for comprehensive GPU instructions

**Quick commands:**
```bash
# One-command quick test
./run_on_a100.sh quick

# Full training run
./run_on_a100.sh full
```

---

## Script Usage Options

### `run_on_a100.sh` - Automated Runner

Three modes available:

#### 1. Quick Test (5-10 minutes)
```bash
./run_on_a100.sh quick
```
- Uses 2,000 training samples
- Tests on 100 samples
- Perfect for testing your setup

#### 2. Medium Run (30-45 minutes)
```bash
./run_on_a100.sh medium
```
- Uses 15,000 training samples
- Tests on 5,000 samples
- Good balance of speed and accuracy

#### 3. Full Training (2-4 hours)
```bash
./run_on_a100.sh full
```
- Uses complete 75,000 training samples
- Tests on all 75,000 samples
- Maximum accuracy for competition submission

---

## Manual Execution

### Basic Run
```bash
source env/bin/activate
python3 temp.py --data_dir dataset --out_csv test_out.csv
```

### With Custom Parameters
```bash
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv \
    --folds 5 \
    --max_train 10000 \
    --max_test 5000 \
    --pca_dim 128
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Path to dataset folder | `dataset` |
| `--images_dir` | Path to images folder | `images` |
| `--out_csv` | Output CSV filename | `test_out.csv` |
| `--folds` | Number of CV folds | `5` |
| `--max_train` | Max training samples | `5000` |
| `--max_test` | Max test samples | `100` |
| `--pca_dim` | PCA dimensions (0=disable) | `128` |
| `--cpu_only` | Force CPU mode | GPU if available |
| `--disable_images` | Disable image features | Auto-detect |
| `--use_images` | Force enable images | Auto-detect |

---

## Verification Commands

### Check GPU
```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Verify Output
```bash
# Check file exists and has correct format
head test_out.csv

# Verify number of predictions
wc -l test_out.csv

# Check for missing values
python3 -c "import pandas as pd; df=pd.read_csv('test_out.csv'); print(df.isnull().sum())"
```

### Validate Submission Format
```bash
python3 << 'EOF'
import pandas as pd

# Load your output
output = pd.read_csv('test_out.csv')
sample = pd.read_csv('dataset/sample_test_out.csv')

# Check columns
assert list(output.columns) == ['sample_id', 'price'], "Column names incorrect!"

# Check types
assert output['price'].dtype in ['float64', 'float32'], "Price must be float!"

# Check no missing values
assert not output.isnull().any().any(), "Missing values detected!"

# Check positive prices
assert (output['price'] > 0).all(), "All prices must be positive!"

print("✓ Output format is valid!")
print(f"  Total predictions: {len(output)}")
print(f"  Price range: ${output['price'].min():.2f} - ${output['price'].max():.2f}")
print(f"  Mean price: ${output['price'].mean():.2f}")
EOF
```

---

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError"**
```bash
source env/bin/activate
pip install -r requirements.txt
```

**2. "CUDA out of memory"**
```bash
python3 temp.py --max_train 2000 --max_test 100 --pca_dim 64
```

**3. "No images found"**
```bash
python3 temp.py --disable_images  # Run text-only mode
```

**4. Script permission denied**
```bash
chmod +x run_on_a100.sh
```

---

## Performance Tips

### For A100 GPU

1. **Use GPU features:** Don't use `--cpu_only` flag
2. **Batch size:** Default settings are optimized for A100
3. **Monitor GPU:** Run `nvidia-smi` in separate terminal
4. **Start small:** Test with quick mode first

### For CPU

1. **Reduce samples:** Use `--max_train 1000 --max_test 100`
2. **Disable images:** Add `--disable_images` flag
3. **Fewer folds:** Use `--folds 2` or `--folds 3`

---

## Example Workflows

### Scenario 1: First Time Setup (Ubuntu + A100)

```bash
# 1. Clone/navigate to project
cd /path/to/product-pricing-predictor

# 2. Setup environment
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Verify GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# 4. Quick test
./run_on_a100.sh quick

# 5. If successful, run full training
./run_on_a100.sh full
```

### Scenario 2: Quick Iteration Development

```bash
# Test your code changes quickly
python3 temp.py \
    --max_train 500 \
    --max_test 50 \
    --folds 2 \
    --disable_images \
    --out_csv test_dev.csv
```

### Scenario 3: Final Submission

```bash
# Full run with logging
./run_on_a100.sh full

# Or manual with detailed logs
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv \
    --folds 5 \
    --max_train 75000 \
    --max_test 75000 \
    2>&1 | tee final_training.log

# Validate output
python3 -c "import pandas as pd; df=pd.read_csv('test_out.csv'); print(f'Predictions: {len(df)}')"
```

---

## Project Structure

```
product-pricing-predictor/
├── dataset/
│   ├── train.csv              # Training data (75k samples)
│   ├── test.csv               # Test data (75k samples)
│   ├── sample_test.csv        # Sample test data
│   └── sample_test_out.csv    # Sample output format
├── images/                    # Downloaded product images
├── src/
│   ├── utils.py               # Helper functions
│   └── example.ipynb          # Jupyter notebook example
├── temp.py                    # Main training script
├── sample_code.py             # Basic sample code
├── run_on_a100.sh            # Automated runner for A100
├── requirements.txt           # Python dependencies
├── UBUNTU_SETUP.md           # Detailed Ubuntu setup
├── A100_GPU_SETUP.md         # Detailed A100 GPU setup
├── QUICKSTART.md             # This file
└── README.md                 # Problem statement
```

---

## Next Steps

1. ✓ **Setup environment** - Follow setup guide for your platform
2. ✓ **Run quick test** - Verify everything works
3. ✓ **Download images** (optional) - Use `src/utils.py`
4. ✓ **Run full training** - Generate `test_out.csv`
5. ✓ **Validate output** - Check format and values
6. ✓ **Submit** - Upload to competition portal
7. ✓ **Document** - Fill in `Documentation_template.md`

---

## Getting Help

- **Setup issues:** Check [UBUNTU_SETUP.md](UBUNTU_SETUP.md#troubleshooting)
- **GPU issues:** Check [A100_GPU_SETUP.md](A100_GPU_SETUP.md#troubleshooting)
- **Model issues:** Review training logs for error messages
- **Format issues:** Compare your output with `dataset/sample_test_out.csv`

---

## Useful Commands Cheatsheet

```bash
# Environment
source env/bin/activate       # Activate venv
deactivate                    # Deactivate venv
pip list                      # List installed packages

# GPU
nvidia-smi                    # Check GPU status
watch -n 1 nvidia-smi        # Monitor GPU real-time

# Training
./run_on_a100.sh quick       # Quick test
./run_on_a100.sh full        # Full training
python3 temp.py --help       # See all options

# Validation
head test_out.csv            # View first lines
wc -l test_out.csv          # Count predictions
python3 -c "import pandas as pd; print(pd.read_csv('test_out.csv').describe())"
```

---

**Last Updated:** October 2025

