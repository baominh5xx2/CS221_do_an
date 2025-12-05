#!/bin/bash

# Train Encoder with Masked Language Modeling (MLM)
# Domain adaptation step before downstream classification
# Supports PhoBERT, T5, and ViT5 models with preset configurations

# Quick usage:
#   ./scripts/run_train_encoder.sh --dataset ViHSD --model phobert-base
#   ./scripts/run_train_encoder.sh --dataset Minhbao5xx2/VOZ-HSD_2M --split balanced --model t5-base
#   ./scripts/run_train_encoder.sh --dataset ViHSD --model vit5-large

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# Default params
DATASET="ViHSD"
DATASET_SPLIT=""
MODEL_PRESET="phobert-base"
EPOCHS=""
BATCH_SIZE=""
LR=""
WEIGHT_DECAY=""
MLM_PROB=""
MAX_LEN=""
OUTPUT_DIR=""

# Model presets: model_preset -> (model_name, epochs, batch_size, lr, weight_decay, mlm_prob, max_len)
declare -A MODELS
MODELS[phobert-base]="vinai/phobert-base:3:16:2e-5:0.01:0.15:256"
MODELS[phobert-large]="vinai/phobert-large:3:8:1e-5:0.01:0.15:256"
MODELS[t5-base]="google/t5-base:3:8:1e-4:0.01:0.15:512"
MODELS[t5-large]="google/t5-large:2:4:5e-5:0.01:0.15:512"
MODELS[vit5-base]="VietAI/vit5-base:3:8:1e-4:0.01:0.15:512"
MODELS[vit5-large]="VietAI/vit5-large:2:4:5e-5:0.01:0.15:512"

usage() {
    cat <<EOF
Usage: $0 --dataset DATASET [--split SPLIT] --model MODEL [OPTIONS]

Datasets:
  ViHSD, ViCTSD, ViHOS, ViHSD_processed, Minhbao5xx2/VOZ-HSD_2M

Model Presets (auto-configured):
  phobert-base   (vinai/phobert-base, 220M)
  phobert-large  (vinai/phobert-large, 370M)
  t5-base        (google/t5-base, 220M)
  t5-large       (google/t5-large, 770M)
  vit5-base      (VietAI/vit5-base, 220M)
  vit5-large     (VietAI/vit5-large, 770M)

Options:
  --dataset DATASET             Dataset name (required)
  --split SPLIT                 For VOZ-HSD_2M: balanced or hate_only
  --model MODEL                 Model preset (default: phobert-base)
  --epochs N                    Override epochs
  --batch_size N                Override batch size
  --learning_rate LR            Override learning rate
  --weight_decay WD             Override weight decay
  --mlm_probability P           Override MLM probability
  --max_length LEN              Override max length
  --output_dir PATH             Custom output directory
  -h, --help                    Show this help message

Examples:
  $0 --dataset ViHSD --model phobert-base
  $0 --dataset Minhbao5xx2/VOZ-HSD_2M --split balanced --model t5-base
  $0 --dataset ViHSD --model vit5-large --batch_size 2 --epochs 5

EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --split) DATASET_SPLIT="$2"; shift 2 ;;
        --model) MODEL_PRESET="$2"; shift 2 ;;
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

# Validate dataset
if [ -z "$DATASET" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    usage
    exit 1
fi

# Load preset config
if [ -z "${MODELS[$MODEL_PRESET]}" ]; then
    echo -e "${RED}Error: Unknown model preset '$MODEL_PRESET'${NC}"
    echo "Available: ${!MODELS[@]}"
    exit 1
fi

IFS=':' read -r MODEL_NAME DEFAULT_EPOCHS DEFAULT_BATCH DEFAULT_LR DEFAULT_WD DEFAULT_MLM DEFAULT_LEN <<< "${MODELS[$MODEL_PRESET]}"

# Apply overrides (CLI > preset > defaults)
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH}
LR=${LR:-$DEFAULT_LR}
WEIGHT_DECAY=${WEIGHT_DECAY:-$DEFAULT_WD}
MLM_PROB=${MLM_PROB:-$DEFAULT_MLM}
MAX_LEN=${MAX_LEN:-$DEFAULT_LEN}

echo -e "${GREEN}=== Encoder Training (MLM) ===${NC}"
echo "Dataset       : $DATASET"
[ -n "$DATASET_SPLIT" ] && echo "Split         : $DATASET_SPLIT"
echo "Model         : $MODEL_PRESET ($MODEL_NAME)"
echo "Epochs        : $EPOCHS"
echo "Batch Size    : $BATCH_SIZE"
echo "Learning Rate : $LR"
echo "Weight Decay  : $WEIGHT_DECAY"
echo "MLM Prob      : $MLM_PROB"
echo "Max Length    : $MAX_LEN"
echo "Output Dir    : ${OUTPUT_DIR:-auto}"
echo ""

# Create logs dir
mkdir -p logs/mlm

# Build python command
CMD="python src/train_encoder.py --dataset \"$DATASET\" --model_name \"$MODEL_NAME\" --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LR --weight_decay $WEIGHT_DECAY --mlm_probability $MLM_PROB --max_length $MAX_LEN"

[ -n "$DATASET_SPLIT" ] && CMD="$CMD --split \"$DATASET_SPLIT\""
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Run training
eval $CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Training completed!${NC}"
else
    echo -e "\n${RED}✗ Training failed!${NC}"
    exit 1
fi
