#!/bin/bash
# Script to run T5 fine-tuning
# Usage: bash scripts/run_train_t5.sh

# Default values
SAVE_MODEL_NAME="ViHateT5-finetuned"
PRE_TRAINED_CKPT="VietAI/vit5-base"
OUTPUT_DIR="outputs/t5_finetuned"
BATCH_SIZE=32
NUM_EPOCHS=4
LEARNING_RATE=2e-4
GPU="0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --save_model_name)
            SAVE_MODEL_NAME="$2"
            shift 2
            ;;
        --pre_trained_ckpt)
            PRE_TRAINED_CKPT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_train_t5.sh [--save_model_name NAME] [--pre_trained_ckpt CKPT] [--output_dir DIR] [--batch_size N] [--num_epochs N] [--learning_rate LR] [--gpu GPU]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "T5 Fine-tuning Configuration:"
echo "=========================================="
echo "Save model name: $SAVE_MODEL_NAME"
echo "Pre-trained checkpoint: $PRE_TRAINED_CKPT"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "GPU: $GPU"
echo "=========================================="
echo ""

# Run training
python src/train_t5.py \
    --save_model_name "$SAVE_MODEL_NAME" \
    --pre_trained_ckpt "$PRE_TRAINED_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gpu "$GPU"

echo ""
echo "✅ Fine-tuning completed! Model saved to: $OUTPUT_DIR"

