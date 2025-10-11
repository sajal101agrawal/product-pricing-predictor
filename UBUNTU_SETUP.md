# Ubuntu Installation and Setup Guide

## Product Pricing Predictor - ML Challenge 2025

This guide provides step-by-step instructions for setting up the Product Pricing Predictor project on Ubuntu systems (tested on Ubuntu 20.04 LTS and later).

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Project Setup](#project-setup)
4. [Verification](#verification)
5. [Running the Code](#running-the-code)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

- **OS:** Ubuntu 20.04 LTS or later
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB+ recommended for deep learning models)
- **Storage:** At least 5GB free space
- **Internet:** Required for downloading dependencies and images

---

## Prerequisites Installation

### 1. Update System Packages

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Python 3 and pip

Check if Python 3 is already installed:
```bash
python3 --version
```

If not installed or version is below 3.8, install it:
```bash
sudo apt install python3 python3-pip python3-dev -y
```

Verify installation:
```bash
python3 --version
pip3 --version
```

### 3. Install System Dependencies

Install required system libraries for image processing and scientific computing:

```bash
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libopenblas-dev \
    liblapack-dev
```

### 4. Install Python Virtual Environment (Recommended)

```bash
sudo apt install python3-venv -y
```

---

## Project Setup

### Step 1: Clone or Download the Project

If using git:
```bash
cd ~/Developer/website-projects
git clone <your-repository-url> product-pricing-predictor
cd product-pricing-predictor
```

Or navigate to your existing project directory:
```bash
cd ~/Developer/website-projects/product-pricing-predictor
```

### Step 2: Create Virtual Environment

Create a virtual environment to isolate project dependencies:

```bash
python3 -m venv env
```

Activate the virtual environment:
```bash
source env/bin/activate
```

Your terminal prompt should now show `(env)` at the beginning.

**Note:** You need to activate this environment every time you work on the project.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Dependencies

Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

This will install:
- **Core ML Libraries:** numpy, pandas, scikit-learn
- **Gradient Boosting:** lightgbm, xgboost, catboost
- **Deep Learning:** PyTorch, torchvision, timm
- **NLP:** sentence-transformers
- **Image Processing:** Pillow
- **Utilities:** tqdm (for progress bars)

**Note:** Installing PyTorch and other deep learning libraries may take 5-15 minutes depending on your internet speed.

### Step 5: Verify Installation

Check if all packages are installed correctly:

```bash
python3 -c "import torch; import pandas; import numpy; import sklearn; print('All core packages imported successfully!')"
```

### Step 6: Download Sample Data (If needed)

The dataset should already be in the `dataset/` folder. Verify:

```bash
ls -lh dataset/
```

You should see:
- `train.csv` (training data with ~75k samples)
- `test.csv` (test data with ~75k samples)
- `sample_test.csv` (sample test data)
- `sample_test_out.csv` (sample output format)

---

## Verification

### Test the Setup

#### 1. Test Basic Functionality

Run the sample code to verify everything works:

```bash
python3 sample_code.py
```

This should generate a `dataset/test_out.csv` file with random predictions.

#### 2. Test Image Download (Optional)

Start a Python session and test image downloading:

```bash
python3
```

Then run:
```python
from src.utils import download_images
import pandas as pd

# Load sample test data
sample_test = pd.read_csv('dataset/sample_test.csv')

# Download first 10 images for testing
download_images(sample_test['image_link'].head(10), 'images')

# Verify
import os
print(f"Downloaded {len(os.listdir('images'))} images")
exit()
```

#### 3. Test Jupyter Notebook (Optional)

If you want to use Jupyter notebooks:

```bash
pip install jupyter notebook ipykernel
jupyter notebook
```

Then navigate to `src/example.ipynb` in your browser.

---

## Running the Code

### Basic Usage

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source env/bin/activate
   ```

2. **Run your prediction model**:
   ```bash
   python3 sample_code.py
   ```

3. **Check the output**:
   ```bash
   head dataset/test_out.csv
   ```

### Working with Images

To download product images for your model:

```python
from src.utils import download_images
import pandas as pd

# Load data
train = pd.read_csv('dataset/train.csv')

# Download training images
download_images(train['image_link'], 'images')
```

**Note:** Image downloading uses multiprocessing and may be throttled by the server. You might need to retry if some images fail to download.

### Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

---

## Troubleshooting

### Issue: Permission Denied Errors

**Solution:** Make sure you're not using `sudo` with pip inside the virtual environment. If you see permission errors outside the venv:
```bash
pip install --user <package_name>
```

### Issue: PyTorch Installation Fails

**Solution:** For CPU-only PyTorch (smaller download):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (if you have NVIDIA GPU):
```bash
# First install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Then install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Memory Errors

**Solution:** 
- Close other applications to free up RAM
- Process data in smaller batches
- Use a machine with more RAM for training large models

### Issue: Image Download Failures

**Solution:**
- Some image URLs may be broken or throttled
- Retry the download after a few minutes
- The code handles missing images gracefully with try-except blocks

### Issue: "ModuleNotFoundError"

**Solution:** Make sure your virtual environment is activated:
```bash
source env/bin/activate
```

Then reinstall requirements:
```bash
pip install -r requirements.txt
```

### Issue: Slow Training

**Solution:**
- Use GPU acceleration if available
- Reduce model size or use smaller architectures
- Process fewer images initially for testing
- Consider using pre-trained models from `timm` library

---

## Additional Configuration

### Setting up GPU Support (Optional)

If you have an NVIDIA GPU:

1. **Install NVIDIA drivers**:
   ```bash
   ubuntu-drivers devices
   sudo ubuntu-drivers autoinstall
   ```

2. **Install CUDA** (if not already installed):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

3. **Verify GPU**:
   ```bash
   nvidia-smi
   ```

4. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Test GPU in Python**:
   ```python
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```

---

## Quick Start Commands (Summary)

```bash
# Navigate to project
cd ~/Developer/website-projects/product-pricing-predictor

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run sample code
python3 sample_code.py

# When done
deactivate
```

---

## Support and Resources

- **Project README:** See `README.md` for problem statement and challenge details
- **Documentation Template:** Use `Documentation_template.md` for your submission
- **Sample Code:** Refer to `sample_code.py` for basic structure
- **Utility Functions:** See `src/utils.py` for helper functions

---

## Notes

- Always work within the virtual environment to avoid dependency conflicts
- Keep your virtual environment activated while working on the project
- The `images/` folder is gitignored - you'll need to download images locally
- Test your solution with `sample_test.csv` before running on the full `test.csv`

---

**Last Updated:** October 2025  
**Tested On:** Ubuntu 20.04 LTS, Ubuntu 22.04 LTS

