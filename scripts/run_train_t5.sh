#!/bin/bash

# Train T5/ViT5 models for Vietnamese Hate Speech Detection
# Encoder-decoder architecture with text-to-text format

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# Default params
DATASET=""
MODEL_PRESET="vit5-base"
EPOCHS=""
BATCH_SIZE=""
LR=""
MAX_LEN=""
DEV_RATIO=""
OUTPUT_DIR=""

# Advanced params defaults
OPTIM="adafactor"
SCHEDULER="linear"
WARMUP="0.1"
GRAD_ACC="1"
WEIGHT_DECAY="0.01"
LABEL_SMOOTHING="0.0"
PATIENCE="3"
NUM_BEAMS="1"
FP16=""
BF16=""

# Model presets: model -> (hf_name, epochs, batch_size, lr, max_len)
declare -A MODELS
MODELS[t5-small]="google/t5-small:5:16:1e-3:512"
MODELS[t5-base]="google/t5-base:5:8:1e-3:512"
MODELS[t5-large]="google/t5-large:3:4:5e-4:512"
MODELS[vit5-base]="VietAI/vit5-base:5:8:1e-3:512"
MODELS[vit5-large]="VietAI/vit5-large:3:4:5e-4:512"
MODELS[vit5-large-1024]="VietAI/vit5-large-1024-vietnews:3:2:3e-4:768"

usage() {
    cat <<EOF
${GREEN}T5/ViT5 Training Script for Vietnamese Hate Speech Detection${NC}

Usage: $0 --dataset DATASET --model MODEL [OPTIONS]

${YELLOW}Required:${NC}
  --dataset DATASET     Dataset name (ViHSD, ViCTSD, ViHOS, Minhbao5xx2/VOZ-HSD_2M)

${YELLOW}Model Presets (auto-configured):${NC}
  t5-small          google/t5-small       (60M)   batch=16, lr=1e-3, max_len=512
  t5-base           google/t5-base        (220M)  batch=8,  lr=1e-3, max_len=512
  t5-large          google/t5-large       (770M)  batch=4,  lr=5e-4, max_len=512
  vit5-base         VietAI/vit5-base      (220M)  batch=8,  lr=1e-3, max_len=512
  vit5-large        VietAI/vit5-large     (770M)  batch=4,  lr=5e-4, max_len=512
  vit5-large-1024   VietAI/vit5-large-1024-vietnews (770M) batch=2, lr=3e-4, max_len=768

${YELLOW}Basic Options:${NC}
  --model MODEL                 Model preset (default: vit5-base)
  --epochs N                    Override epochs
  --batch_size N                Override batch size
  --learning_rate LR            Override learning rate
  --max_length LEN              Override max length
  --dev_ratio R                 Validation split ratio (default: 0.1)
  --output_dir PATH             Custom output directory

${YELLOW}Advanced Options:${NC}
  --optim NAME                  Optimizer (adafactor, adamw_torch) [default: adafactor]
  --scheduler NAME              LR Scheduler (linear, cosine, constant) [default: linear]
  --warmup RATIO                Warmup ratio [default: 0.1]
  --grad_acc N                  Gradient accumulation steps [default: 1]
  --weight_decay VAL            Weight decay [default: 0.01]
  --label_smoothing VAL         Label smoothing factor [default: 0.0]
  --patience N                  Early stopping patience [default: 3]
  --num_beams N                 Number of beams for generation [default: 1]
  --fp16                        Enable FP16 mixed precision
  --bf16                        Enable BF16 mixed precision (Ampere+ GPUs)
  -h, --help                    Show this help message

${YELLOW}Examples:${NC}
  # ViT5 base on ViHSD (Standard)
  $0 --dataset ViHSD --model vit5-base

  # T5 base with FP16 and Beam Search
  $0 --dataset Minhbao5xx2/re_VOZ-HSD --model t5-base --fp16 --num_beams 3

  # Custom optimization
  $0 --dataset ViHSD --model vit5-base --optim adamw_torch --learning_rate 2e-5 --scheduler cosine

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
        --max_length) MAX_LEN="$2"; shift 2 ;;
        --dev_ratio) DEV_RATIO="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        
        # Advanced args
        --optim) OPTIM="$2"; shift 2 ;;
        --scheduler) SCHEDULER="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --grad_acc) GRAD_ACC="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --label_smoothing) LABEL_SMOOTHING="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --num_beams) NUM_BEAMS="$2"; shift 2 ;;
        --fp16) FP16="--fp16"; shift 1 ;;
        --bf16) BF16="--bf16"; shift 1 ;;
        
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

# Validate model preset
if [ -z "${MODELS[$MODEL_PRESET]}" ]; then
    echo -e "${RED}Error: Unknown model preset '$MODEL_PRESET'${NC}"
    echo "Available models: ${!MODELS[@]}"
    exit 1
fi

# Parse preset config
IFS=':' read -r MODEL_NAME DEFAULT_EPOCHS DEFAULT_BATCH DEFAULT_LR DEFAULT_LEN <<< "${MODELS[$MODEL_PRESET]}"

# Apply overrides (CLI > preset)
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH}
LR=${LR:-$DEFAULT_LR}
MAX_LEN=${MAX_LEN:-$DEFAULT_LEN}
DEV_RATIO=${DEV_RATIO:-0.1}

# Print configuration
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           T5/ViT5 Training for Hate Speech Detection         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Dataset              : $DATASET"
echo "  Model Preset         : $MODEL_PRESET"
echo "  HuggingFace Model    : $MODEL_NAME"
echo "  Epochs               : $EPOCHS"
echo "  Batch Size           : $BATCH_SIZE"
echo "  Learning Rate        : $LR"
echo "  Max Length           : $MAX_LEN"
echo "  Dev Ratio            : $DEV_RATIO"
echo "  Optimizer            : $OPTIM"
echo "  Scheduler            : $SCHEDULER"
echo "  Warmup Ratio         : $WARMUP"
echo "  Grad Accumulation    : $GRAD_ACC"
echo "  Patience             : $PATIENCE"
echo "  Num Beams            : $NUM_BEAMS"
[ -n "$FP16" ] && echo "  Mixed Precision      : FP16"
[ -n "$BF16" ] && echo "  Mixed Precision      : BF16"
[ -n "$OUTPUT_DIR" ] && echo "  Output Dir           : $OUTPUT_DIR"
echo ""

# Create logs directory
mkdir -p logs/t5

# Build command
CMD="python src/train_t5.py"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --model_name \"$MODEL_NAME\""
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LR"
CMD="$CMD --max_length $MAX_LEN"
CMD="$CMD --dev_ratio $DEV_RATIO"

# Advanced params
CMD="$CMD --optim $OPTIM"
CMD="$CMD --lr_scheduler_type $SCHEDULER"
CMD="$CMD --warmup_ratio $WARMUP"
CMD="$CMD --gradient_accumulation_steps $GRAD_ACC"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --label_smoothing_factor $LABEL_SMOOTHING"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --num_beams $NUM_BEAMS"

[ -n "$FP16" ] && CMD="$CMD $FP16"
[ -n "$BF16" ] && CMD="$CMD $BF16"
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir \"$OUTPUT_DIR\""

# Run training
echo -e "${GREEN}🚀 Starting T5 training...${NC}"
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
