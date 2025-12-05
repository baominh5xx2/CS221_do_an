#!/bin/bash

# Train BERT Encoder with Masked Language Modeling (MLM)
# Domain adaptation step before downstream classification

# Quick usage:
#   ./scripts/run_train_encoder.sh \
#       --dataset ViHSD \
#       --model_name vinai/phobert-base \
#       --epochs 3 \
#       --batch_size 16 \
#       --learning_rate 2e-5 \
#       --weight_decay 0.01 \
#       --mlm_probability 0.15 \
#       --max_length 256 \
#       --output_dir models/ViHSD_encoder_mlm

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# Default params (edit here or override via CLI)
DATASET="ViHSD"
MODEL_NAME="vinai/phobert-base"
EPOCHS=3
BATCH_SIZE=16
LR=2e-5
WEIGHT_DECAY=0.01
MLM_PROB=0.15
MAX_LEN=256
OUTPUT_DIR=""

usage() {
    cat <<EOF
Usage: $0 [--dataset ViHSD|ViCTSD|ViHOS] [--model_name MODEL] [--epochs N] [--batch_size N] [--learning_rate LR] [--weight_decay WD] [--mlm_probability P] [--max_length LEN] [--output_dir PATH]
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate) LR="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --mlm_probability) MLM_PROB="$2"; shift 2 ;;
        --max_length) MAX_LEN="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

echo -e "${GREEN}=== Starting Encoder Training (MLM) ===${NC}"
echo "Dataset       : $DATASET"
echo "Model         : $MODEL_NAME"
echo "Epochs        : $EPOCHS"
echo "Batch Size    : $BATCH_SIZE"
echo "Learning Rate : $LR"
echo "Weight Decay  : $WEIGHT_DECAY"
echo "MLM Prob      : $MLM_PROB"
echo "Max Length    : $MAX_LEN"
echo "Output Dir    : ${OUTPUT_DIR:-auto (models/${DATASET}_encoder_mlm)}"
echo ""

# Create logs dir
mkdir -p logs/mlm

# Run python script
python src/train_encoder.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --mlm_probability $MLM_PROB \
    --max_length $MAX_LEN \
    ${OUTPUT_DIR:+--output_dir "$OUTPUT_DIR"}

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Training completed successfully!${NC}"
    echo "Model saved in ${OUTPUT_DIR:-models/${DATASET}_encoder_mlm}"
else
    echo -e "\n${RED}✗ Training failed!${NC}"
    exit 1
fi
