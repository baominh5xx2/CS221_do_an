#!/bin/bash

# Train encoder models for supervised classification
# Supports preset model keys OR direct HuggingFace model ids

# Quick usage:
#   ./scripts/run_train_encoder.sh --dataset ViHSD --model phobert-base
#   ./scripts/run_train_encoder.sh --dataset ViHSD_processed --model visobert
#   ./scripts/run_train_encoder.sh --dataset ViHSD_processed --model uitnlp/visobert

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# Default params
DATASET="ViHSD"
MODEL_PRESET="phobert-base"
EPOCHS=""
BATCH_SIZE=""
LR=""
WEIGHT_DECAY=""
MAX_LEN=""
OUTPUT_DIR=""

# Model presets: model_key -> (model_name, epochs, batch_size, lr, weight_decay, max_len)
declare -A MODELS
MODELS[phobert-base]="vinai/phobert-base:10:16:2e-5:0.01:256"
MODELS[phobert-large]="vinai/phobert-large:10:8:1e-5:0.01:256"
MODELS[visobert]="uitnlp/visobert:10:16:1e-5:0.01:128"
MODELS[t5-base]="google/t5-base:10:8:1e-4:0.01:512"
MODELS[t5-large]="google/t5-large:10:4:5e-5:0.01:512"
MODELS[vit5-base]="VietAI/vit5-base:10:8:1e-4:0.01:512"
MODELS[vit5-large]="VietAI/vit5-large:10:4:5e-5:0.01:512"

usage() {
    cat <<EOF
Usage: $0 --dataset DATASET --model MODEL [OPTIONS]

Datasets:
  ViHSD, ViCTSD, ViHOS, ViHSD_processed, Minhbao5xx2/VOZ-HSD_2M

Model Presets (auto-configured):
  phobert-base   (vinai/phobert-base, 220M)
  phobert-large  (vinai/phobert-large, 370M)
  visobert       (uitnlp/visobert, 135M - social media)
  t5-base        (google/t5-base, 220M)
  t5-large       (google/t5-large, 770M)
  vit5-base      (VietAI/vit5-base, 220M)
  vit5-large     (VietAI/vit5-large, 770M)

Options:
  --dataset DATASET             Dataset name (required)
  --model MODEL                 Preset key (e.g. visobert) OR HuggingFace id (e.g. uitnlp/visobert)
  --epochs N                    Override epochs
  --batch_size N                Override batch size
  --learning_rate LR            Override learning rate
  --weight_decay WD             Override weight decay
  --max_length LEN              Override max length
  --output_dir PATH             Custom output directory
  -h, --help                    Show this help message

Examples:
  $0 --dataset ViHSD --model phobert-base
  $0 --dataset ViHSD_processed --model visobert
  $0 --dataset ViHSD_processed --model uitnlp/visobert
  $0 --dataset Minhbao5xx2/VOZ-HSD_2M --model t5-base
  $0 --dataset ViHSD --model vit5-large --batch_size 2 --epochs 5

EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL_PRESET="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --learning_rate) LR="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --max_length) MAX_LEN="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

# Common alias: allow passing HF id for ViSoBERT
if [[ "$MODEL_PRESET" == "uitnlp/visobert" ]]; then
  MODEL_PRESET="visobert"
fi

# Validate dataset
if [ -z "$DATASET" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    usage
    exit 1
fi

# Resolve model: preset key OR direct HuggingFace id
if [ -n "${MODELS[$MODEL_PRESET]}" ]; then
  IFS=':' read -r MODEL_NAME DEFAULT_EPOCHS DEFAULT_BATCH DEFAULT_LR DEFAULT_WD DEFAULT_LEN <<< "${MODELS[$MODEL_PRESET]}"
else
  MODEL_NAME="$MODEL_PRESET"
  DEFAULT_EPOCHS=10
  DEFAULT_BATCH=16
  DEFAULT_LR=2e-5
  DEFAULT_WD=0.01
  DEFAULT_LEN=256
  if [[ "$MODEL_NAME" == *"visobert"* ]]; then
    DEFAULT_LR=1e-5
    DEFAULT_LEN=128
  fi
fi

# Apply overrides (CLI > preset > defaults)
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH}
LR=${LR:-$DEFAULT_LR}
WEIGHT_DECAY=${WEIGHT_DECAY:-$DEFAULT_WD}
MAX_LEN=${MAX_LEN:-$DEFAULT_LEN}

echo -e "${GREEN}=== Encoder Training (Classification) ===${NC}"
echo "Dataset       : $DATASET"
echo "Model         : $MODEL_PRESET ($MODEL_NAME)"
echo "Epochs        : $EPOCHS"
echo "Batch Size    : $BATCH_SIZE"
echo "Learning Rate : $LR"
echo "Weight Decay  : $WEIGHT_DECAY"
echo "Max Length    : $MAX_LEN"
echo "Output Dir    : ${OUTPUT_DIR:-auto}"
echo ""

# Create logs dir
mkdir -p logs/encoder_classification

# Build python command
CMD="python src/train_encoder.py --dataset \"$DATASET\" --model_name \"$MODEL_NAME\" --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LR --weight_decay $WEIGHT_DECAY --max_length $MAX_LEN"

[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Run training
eval $CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Training completed!${NC}"
else
    echo -e "\n${RED}✗ Training failed!${NC}"
    exit 1
fi
