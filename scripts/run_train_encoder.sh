#!/bin/bash

# Train BERT Encoder using Masked Language Modeling (MLM)
# This script performs domain adaptation on the dataset

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

DATASET="ViHSD"
EPOCHS=3
BATCH_SIZE=16
LR=2e-5

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate) LR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo -e "${GREEN}=== Starting Encoder Training (MLM) ===${NC}"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo ""

# Create logs dir
mkdir -p logs/mlm

# Run python script
python src/train_encoder.py \
    --dataset "$DATASET" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Training completed successfully!${NC}"
    echo "Model saved in models/${DATASET}_encoder_mlm"
else
    echo -e "\n${RED}✗ Training failed!${NC}"
    exit 1
fi
