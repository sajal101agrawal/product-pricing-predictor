#!/bin/bash
# Quick Start Script for Running on Nvidia A100
# Usage: ./run_on_a100.sh [quick|full]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Product Pricing Predictor - A100 Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Are you on a GPU instance?${NC}"
    exit 1
fi

# Display GPU info
echo -e "${YELLOW}GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Activate virtual environment
if [ ! -d "env" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv env
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source env/bin/activate

# Check if dependencies are installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
fi

# Verify CUDA
echo -e "${YELLOW}Verifying CUDA setup...${NC}"
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA Available: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available, will run on CPU (slow!)")
EOF
echo ""

# Determine run mode
MODE=${1:-quick}

if [ "$MODE" == "quick" ]; then
    echo -e "${GREEN}Running QUICK TEST mode (small dataset for testing)${NC}"
    echo -e "${YELLOW}This will take ~5-10 minutes${NC}"
    echo ""
    python3 temp.py \
        --data_dir dataset \
        --images_dir images \
        --out_csv test_out_quick.csv \
        --folds 2 \
        --max_train 0 \
        --max_test 5000 \
        --pca_dim 128
    
    OUTPUT_FILE="test_out_quick.csv"

elif [ "$MODE" == "full" ]; then
    echo -e "${GREEN}Running FULL TRAINING mode (complete dataset)${NC}"
    echo -e "${YELLOW}This will take ~2-4 hours depending on image availability${NC}"
    echo ""
    
    # Ask for confirmation
    read -p "Continue with full training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    python3 temp.py \
        --data_dir dataset \
        --images_dir images \
        --out_csv test_out.csv \
        --folds 5 \
        --max_train 75000 \
        --max_test 75000 \
        --pca_dim 128 \
        2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
    
    OUTPUT_FILE="test_out.csv"

elif [ "$MODE" == "medium" ]; then
    echo -e "${GREEN}Running MEDIUM mode (balanced dataset)${NC}"
    echo -e "${YELLOW}This will take ~30-45 minutes${NC}"
    echo ""
    python3 temp.py \
        --data_dir dataset \
        --images_dir images \
        --out_csv test_out_medium.csv \
        --folds 3 \
        --max_train 15000 \
        --max_test 5000 \
        --pca_dim 128
    
    OUTPUT_FILE="test_out_medium.csv"

else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    echo "Usage: $0 [quick|medium|full]"
    echo "  quick  - Fast test with 2K train, 100 test (~5-10 min)"
    echo "  medium - Medium run with 15K train, 5K test (~30-45 min)"
    echo "  full   - Complete dataset with 75K train, 75K test (~2-4 hrs)"
    exit 1
fi

# Check output
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}✓ Output file created: $OUTPUT_FILE${NC}"
    echo -e "${YELLOW}File size: $(ls -lh $OUTPUT_FILE | awk '{print $5}')${NC}"
    echo -e "${YELLOW}Number of predictions: $(tail -n +2 $OUTPUT_FILE | wc -l)${NC}"
    echo ""
    echo -e "${YELLOW}First 5 predictions:${NC}"
    head -n 6 "$OUTPUT_FILE"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Validate format: python3 -c \"import pandas as pd; df=pd.read_csv('$OUTPUT_FILE'); print(df.info())\""
    echo "2. Check for missing values: python3 -c \"import pandas as pd; print(pd.read_csv('$OUTPUT_FILE').isnull().sum())\""
    echo "3. Submit $OUTPUT_FILE to the competition portal"
else
    echo -e "${RED}ERROR: Output file not created!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Done!${NC}"

