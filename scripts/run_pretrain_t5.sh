#!/bin/bash
# Script to run T5 pretraining with span corruption
# Usage: bash scripts/run_pretrain_t5.sh

# Default values
DATASET_NAME="Minhbao5xx2/re_VOZ-HSD"
SPLIT_NAME="hate_only"
MAX_SAMPLES=50000
OUTPUT_DIR="vihate_t5_pretrain"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --split_name)
            SPLIT_NAME="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train_file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --valid_file)
            VALID_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_pretrain_t5.sh [--dataset_name DATASET] [--split_name SPLIT] [--max_samples N] [--output_dir DIR] [--train_file FILE] [--valid_file FILE]"
            exit 1
            ;;
    esac
done

# Build command
CMD="python src/pre_train_t5.py"

if [ -n "$DATASET_NAME" ]; then
    CMD="$CMD --dataset_name \"$DATASET_NAME\""
fi

if [ -n "$SPLIT_NAME" ]; then
    CMD="$CMD --split_name \"$SPLIT_NAME\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$TRAIN_FILE" ]; then
    CMD="$CMD --train_file \"$TRAIN_FILE\""
fi

if [ -n "$VALID_FILE" ]; then
    CMD="$CMD --valid_file \"$VALID_FILE\""
fi

# Print configuration
echo "=========================================="
echo "T5 Pretraining Configuration:"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Split: $SPLIT_NAME"
echo "Max samples: $MAX_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$TRAIN_FILE" ]; then
    echo "Train file: $TRAIN_FILE"
fi
if [ -n "$VALID_FILE" ]; then
    echo "Valid file: $VALID_FILE"
fi
echo "=========================================="
echo ""

# Run command
eval $CMD

echo ""
echo "✅ Pretraining completed! Model saved to: $OUTPUT_DIR/final"

