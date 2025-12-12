#!/bin/bash

# Train PhoBERT/BERT models for Vietnamese Hate Speech Detection
# Encoder-only architecture for classification

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# Default params
DATASET=""
MODEL_NAME="vinai/phobert-base"
EPOCHS="10"
BATCH_SIZE="16"
MAX_LENGTH="256"
LR="2e-5"
WEIGHT_DECAY="0.01"
WARMUP_RATIO="0.1"
PATIENCE="3"
SEED="42"
OUTPUT_DIR=""

usage() {
    cat <<EOF
${GREEN}PhoBERT/BERT Training Script for Vietnamese Hate Speech Detection${NC}

Usage: $0 --dataset DATASET [OPTIONS]

${YELLOW}Required:${NC}
  --dataset DATASET     Dataset name (ViHSD, ViCTSD, ViHOS, ViHSD_processed, Minhbao5xx2/VOZ-HSD_2M)

${YELLOW}Options:${NC}
  --model_name MODEL            HuggingFace model (default: vinai/phobert-base)
  --epochs N                    Number of epochs (default: 10)
  --batch_size N                Batch size (default: 16)
  --max_length LEN              Max sequence length (default: 256)
  --learning_rate LR            Learning rate (default: 2e-5)
  --weight_decay WD             Weight decay (default: 0.01)
  --warmup_ratio RATIO          Warmup ratio (default: 0.1)
  --patience N                  Early stopping patience (default: 3)
  --seed N                      Random seed (default: 42)
  --output_dir PATH             Custom output directory
  -h, --help                    Show this help message

${YELLOW}Model Presets:${NC}
  vinai/phobert-base            PhoBERT base (135M) - Vietnamese
  vinai/phobert-large           PhoBERT large (370M) - Vietnamese
  uitnlp/visobert               ViSoBERT (135M) - Vietnamese social media
  bert-base-multilingual-cased  mBERT (110M) - Multilingual

${YELLOW}Examples:${NC}
  # PhoBERT base on ViHSD
  $0 --dataset ViHSD

  # ViSoBERT on ViHSD_processed
  $0 --dataset ViHSD_processed --model_name uitnlp/visobert --max_length 128

  # PhoBERT on VOZ-HSD
  $0 --dataset Minhbao5xx2/VOZ-HSD_2M --epochs 5

  # Custom hyperparameters
  $0 --dataset ViHSD --epochs 20 --batch_size 32 --learning_rate 3e-5

EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --learning_rate) LR="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo -e "${RED}Unknown arg: $1${NC}"; usage; exit 1 ;;
    esac
done

# Validate required args
if [ -z "$DATASET" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    usage
    exit 1
fi

# Print configuration
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      PhoBERT/BERT Training for Hate Speech Detection        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Dataset              : $DATASET"
echo "  Model                : $MODEL_NAME"
echo "  Epochs               : $EPOCHS"
echo "  Batch Size           : $BATCH_SIZE"
echo "  Max Length           : $MAX_LENGTH"
echo "  Learning Rate        : $LR"
echo "  Weight Decay         : $WEIGHT_DECAY"
echo "  Warmup Ratio         : $WARMUP_RATIO"
echo "  Patience             : $PATIENCE"
echo "  Seed                 : $SEED"
[ -n "$OUTPUT_DIR" ] && echo "  Output Dir           : $OUTPUT_DIR"
echo ""

# Create logs directory
mkdir -p logs/train

# Build command
CMD="python src/train.py"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --model_name \"$MODEL_NAME\""
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --learning_rate $LR"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --warmup_ratio $WARMUP_RATIO"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --seed $SEED"

[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Run training
echo -e "${GREEN}🚀 Starting PhoBERT/BERT training...${NC}"
echo ""

eval $CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ✓ Training Complete!                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ✗ Training Failed!                        ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
