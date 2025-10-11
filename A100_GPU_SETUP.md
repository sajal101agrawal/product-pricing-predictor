# Nvidia A100 GPU Setup and Execution Guide

## Running Product Pricing Predictor on Nvidia A100

This guide provides optimized instructions for running the multimodal price prediction model on Nvidia A100 GPUs (tested on A100 40GB/80GB variants).

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [GPU Verification](#gpu-verification)
4. [Running the Model](#running-the-model)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **GPU:** Nvidia A100 (40GB or 80GB)
- **CUDA:** 11.8 or 12.1 (typically pre-installed on cloud instances)
- **Ubuntu:** 20.04 LTS or later
- **Python:** 3.8-3.11
- **RAM:** 32GB+ recommended
- **Storage:** 20GB+ free space

---

## Initial Setup

### Step 1: Check CUDA Installation

First, verify CUDA is installed and accessible:

```bash
nvidia-smi
```

You should see output showing your A100 GPU(s), CUDA version, and driver version.

```bash
nvcc --version
```

If CUDA is not installed, install it:

```bash
# For CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Add CUDA to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Step 2: Navigate to Project Directory

```bash
cd ~/Developer/website-projects/product-pricing-predictor
```

Or if in a different location:
```bash
cd /path/to/product-pricing-predictor
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv env
source env/bin/activate
```

### Step 4: Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install All Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy, pandas, scikit-learn
- lightgbm, xgboost, catboost
- sentence-transformers (for text embeddings)
- timm (for vision models)
- Pillow (for image processing)
- tqdm (for progress bars)

---

## GPU Verification

### Verify PyTorch CUDA Setup

Run this Python script to verify GPU accessibility:

```bash
python3 << 'EOF'
import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU with a simple operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU computation test passed!")
else:
    print("WARNING: CUDA not available!")
EOF
```

Expected output for A100:
```
PyTorch Version: 2.2.0+cu118
CUDA Available: True
CUDA Version: 11.8
Number of GPUs: 1
GPU Name: NVIDIA A100-SXM4-40GB (or 80GB)
GPU Memory: 40.00 GB (or 80.00 GB)
✓ GPU computation test passed!
```

---

## Running the Model

### Option 1: Quick Test Run (Fast Iteration)

For testing with a small subset of data:

```bash
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv \
    --folds 3 \
    --max_train 5000 \
    --max_test 100 \
    --pca_dim 128
```

**Parameters explained:**
- `--max_train 5000`: Use only 5000 training samples
- `--max_test 100`: Use only 100 test samples
- `--folds 3`: Use 3-fold cross-validation
- `--pca_dim 128`: Reduce embeddings to 128 dimensions
- GPU will be used automatically if available

**Estimated time:** 5-10 minutes

### Option 2: Full Training Run (Complete Dataset)

For full training on the complete dataset:

```bash
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv \
    --folds 5 \
    --max_train 75000 \
    --max_test 75000 \
    --pca_dim 128
```

**Estimated time:** 2-4 hours depending on image availability

### Option 3: Without Images (Text-Only)

If images are not downloaded:

```bash
python3 temp.py \
    --data_dir dataset \
    --out_csv test_out.csv \
    --disable_images \
    --folds 5 \
    --max_train 75000 \
    --max_test 75000
```

### Option 4: Force CPU (Not Recommended on A100)

```bash
python3 temp.py \
    --cpu_only \
    --data_dir dataset \
    --out_csv test_out.csv
```

---

## Performance Optimization

### 1. Monitor GPU Usage

Open a separate terminal and run:
```bash
watch -n 1 nvidia-smi
```

This will show real-time GPU utilization, memory usage, and temperature.

### 2. Batch Size Tuning

The script uses default batch sizes:
- Text encoding: 512 samples per batch
- Image encoding: 64 images per batch

These are optimized for A100. If you need to adjust, modify in `temp.py`:

```python
# Line 302 - Text batch size
train_text_emb = encode_text(train["catalog_content"].tolist(), 1024)  # Increase from 512

# Line 174 - Image batch size  
def encode(self, pil_list, batch=128):  # Increase from 64
```

### 3. Multi-GPU Support (If Available)

If you have multiple A100s, you can parallelize:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python3 temp.py --data_dir dataset --out_csv test_out_gpu0.csv &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python3 temp.py --data_dir dataset --out_csv test_out_gpu1.csv &
```

### 4. Mixed Precision Training (Advanced)

For faster training, enable mixed precision (requires code modification):

```python
# Add at the top of temp.py
from torch.cuda.amp import autocast, GradScaler

# Wrap model forward pass with autocast
with autocast():
    feats = self.model(b)
```

### 5. Memory Management

Clear CUDA cache if running multiple experiments:

```python
import torch
torch.cuda.empty_cache()
```

---

## Full Workflow Example

### Complete End-to-End Execution

```bash
#!/bin/bash
# setup_and_run.sh - Complete workflow script

# 1. Activate environment
source env/bin/activate

# 2. Verify GPU
echo "=== GPU Verification ==="
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Download images (optional but recommended)
echo "=== Downloading Images ==="
python3 << 'EOF'
from src.utils import download_images
import pandas as pd
import os

# Download sample test images first
sample_test = pd.read_csv('dataset/sample_test.csv')
print(f"Downloading {len(sample_test)} sample images...")
download_images(sample_test['image_link'], 'images')
print(f"Downloaded {len(os.listdir('images'))} images")
EOF

# 4. Run quick test
echo "=== Running Quick Test ==="
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out_quick.csv \
    --folds 2 \
    --max_train 1000 \
    --max_test 100

# 5. Verify output
echo "=== Verifying Output ==="
head -n 10 test_out_quick.csv

# 6. Run full training
echo "=== Running Full Training ==="
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv \
    --folds 5 \
    --max_train 75000 \
    --max_test 75000

echo "=== Complete! ==="
ls -lh test_out.csv
```

Save as `setup_and_run.sh`, make executable, and run:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

---

## Monitoring and Logging

### 1. Run with Detailed Logging

```bash
python3 temp.py \
    --data_dir dataset \
    --images_dir images \
    --out_csv test_out.csv 2>&1 | tee training.log
```

This saves all output to `training.log` while displaying it.

### 2. GPU Monitoring Script

Save as `monitor_gpu.sh`:

```bash
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status at $(date) ==="
    nvidia-smi
    echo ""
    echo "=== GPU Memory Usage ==="
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
    echo ""
    echo "=== GPU Utilization ==="
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv
    sleep 5
done
```

Run in a separate terminal:
```bash
chmod +x monitor_gpu.sh
./monitor_gpu.sh
```

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Solution 1:** Reduce batch sizes in `temp.py`:
```python
# Line 302
train_text_emb = encode_text(train["catalog_content"].tolist(), 256)  # Reduce from 512

# Line 335
train_img_emb = vit.encode(batch_open(img_paths_train), batch=32)  # Reduce from 64
```

**Solution 2:** Reduce PCA dimensions:
```bash
python3 temp.py --pca_dim 64  # Instead of 128
```

**Solution 3:** Use CPU for embeddings, GPU for boosting:
```bash
python3 temp.py --cpu_only
```

### Issue 2: "RuntimeError: CUDA error: no kernel image is available"

This means PyTorch CUDA version mismatch.

**Solution:**
```bash
# Check CUDA version
nvidia-smi | grep CUDA

# Reinstall PyTorch with matching version
# For CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: GPU Not Being Used

**Check 1:** Verify CUDA is available:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Check 2:** Ensure you're not using `--cpu_only` flag

**Check 3:** Monitor GPU usage:
```bash
nvidia-smi
```

If utilization shows 0%, the code might be running on CPU.

### Issue 4: Slow Image Downloads

**Solution:**
```python
# Modify src/utils.py - reduce parallel connections
with multiprocessing.Pool(20) as pool:  # Reduce from 100
```

Or download images in smaller batches:
```python
import pandas as pd
from src.utils import download_images

df = pd.read_csv('dataset/train.csv')
batch_size = 1000

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    download_images(batch['image_link'], 'images')
    print(f"Downloaded batch {i//batch_size + 1}")
```

### Issue 5: Permission Errors

**Solution:**
```bash
# Ensure directories are writable
chmod -R u+w dataset/ images/
```

---

## Performance Benchmarks (A100 40GB)

Expected execution times:

| Configuration | Training Samples | Test Samples | Time | GPU Memory |
|--------------|------------------|--------------|------|------------|
| Quick test   | 1,000           | 100          | ~3 min | ~5 GB |
| Medium       | 10,000          | 1,000        | ~15 min | ~10 GB |
| Full (no img)| 75,000          | 75,000       | ~45 min | ~8 GB |
| Full (w/img) | 75,000          | 75,000       | ~2-3 hrs | ~15 GB |

*Times vary based on image availability and download time*

---

## Best Practices

1. **Start Small:** Always test with `--max_train 1000 --max_test 100` first
2. **Monitor Resources:** Keep `nvidia-smi` running in a separate terminal
3. **Save Checkpoints:** Run with different seeds and ensemble results
4. **Log Everything:** Use `2>&1 | tee training.log` to save outputs
5. **Verify Output:** Always check `test_out.csv` format matches `sample_test_out.csv`

---

## Cloud Platform Specific Notes

### AWS (p4d.24xlarge with 8x A100)

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python3 temp.py ...
```

### Google Cloud (a2-highgpu-1g with 1x A100)

```bash
# Typically pre-configured with CUDA
python3 temp.py ...
```

### Azure (Standard_ND96asr_v4 with A100)

```bash
# May need to install CUDA drivers
sudo /usr/local/cuda/bin/cuda-install-samples.sh ~
```

---

## Contact and Support

For issues specific to:
- **CUDA/GPU:** Check Nvidia documentation
- **PyTorch:** https://pytorch.org/get-started/
- **Model issues:** Review `training.log` for error messages

---

**Last Updated:** October 2025  
**Tested On:** Nvidia A100 40GB/80GB, Ubuntu 20.04/22.04, CUDA 11.8/12.1

