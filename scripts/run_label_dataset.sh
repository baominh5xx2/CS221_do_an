#!/bin/bash

# Auto-labeling script for VOZ-HSD dataset with parallel batch processing
# Usage: bash scripts/run_label_dataset.sh

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

# Model path (required)
MODEL_PATH="models/ViHSD_processed_phobert-base_20251204_112736"

# Dataset split to label
SPLIT="train"  # Options: train, validation, test

# Parallel processing settings
TOTAL_BATCHES=10  # Number of batches to split dataset into
# Set to 1 for no parallelization

# Inference settings
BATCH_SIZE=32     # Inference batch size
MAX_LENGTH=256    # Maximum sequence length

# Output directory
OUTPUT_DIR="labeled_data/voz_hsd"

# Optional: limit samples for testing
# MAX_SAMPLES=1000  # Uncomment to limit samples
MAX_SAMPLES=""

# ============================================================================
# SCRIPT LOGIC - DO NOT MODIFY UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================================

echo "========================================="
echo "VOZ-HSD Auto-Labeling Pipeline"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Split: $SPLIT"
echo "Total Batches: $TOTAL_BATCHES"
echo "Batch Size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run labeling for a single batch
run_batch() {
    local batch_idx=$1
    echo "🚀 Starting batch $((batch_idx + 1))/$TOTAL_BATCHES"
    
    # Build command
    cmd="python src/label_dataset.py \
        --model_path $MODEL_PATH \
        --split $SPLIT \
        --batch_idx $batch_idx \
        --total_batches $TOTAL_BATCHES \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR"
    
    # Add max_samples if set
    if [ -n "$MAX_SAMPLES" ]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi
    
    # Run command
    eval $cmd
    
    echo "✅ Completed batch $((batch_idx + 1))/$TOTAL_BATCHES"
    echo ""
}

# Check if running in parallel mode
if [ "$TOTAL_BATCHES" -eq 1 ]; then
    echo "Running in sequential mode (1 batch)..."
    run_batch 0
else
    echo "Running in parallel mode ($TOTAL_BATCHES batches)..."
    echo "Note: Make sure you have enough GPU/CPU resources!"
    echo ""
    
    # Option 1: Sequential execution (safer, one batch at a time)
    # Uncomment this block if you want to run batches one by one
    # for ((i=0; i<$TOTAL_BATCHES; i++)); do
    #     run_batch $i
    # done
    
    # Option 2: Parallel execution (faster, but requires more resources)
    # Run all batches in parallel using background processes
    echo "Starting all batches in parallel..."
    for ((i=0; i<$TOTAL_BATCHES; i++)); do
        run_batch $i &
    done
    
    # Wait for all background processes to complete
    echo "Waiting for all batches to complete..."
    wait
    
    echo "✅ All batches completed!"
fi

# Combine results from all batches
echo ""
echo "========================================="
echo "Combining Results & Cleaning Up"
echo "========================================="

TEMP_DIR="$OUTPUT_DIR/temp"

# Check if temp directory exists
if [ ! -d "$TEMP_DIR" ]; then
    echo "❌ Error: Temp directory not found!"
    exit 1
fi

# Combine labeled data
echo "📦 Combining labeled CSV files..."
OUTPUT_COMBINED="$OUTPUT_DIR/${SPLIT}_labeled.csv"

if ls "$TEMP_DIR/${SPLIT}_batch_"*".csv" 1> /dev/null 2>&1; then
    # Combine all batch CSV files
    head -n 1 "$TEMP_DIR/${SPLIT}_batch_0_${TOTAL_BATCHES}.csv" > "$OUTPUT_COMBINED"
    for ((i=0; i<$TOTAL_BATCHES; i++)); do
        tail -n +2 "$TEMP_DIR/${SPLIT}_batch_${i}_${TOTAL_BATCHES}.csv" >> "$OUTPUT_COMBINED"
    done
    echo "✅ Combined labeled data: $OUTPUT_COMBINED"
else
    echo "⚠️  No batch CSV files found!"
fi

# Combine and summarize metrics
echo ""
echo "📊 Combining metrics..."
METRICS_COMBINED="$OUTPUT_DIR/metrics_summary.csv"

if ls "$TEMP_DIR/metrics_batch_"*".csv" 1> /dev/null 2>&1; then
    # Combine all metrics files
    head -n 1 "$TEMP_DIR/metrics_batch_0_${TOTAL_BATCHES}.csv" > "$METRICS_COMBINED"
    for ((i=0; i<$TOTAL_BATCHES; i++)); do
        tail -n +2 "$TEMP_DIR/metrics_batch_${i}_${TOTAL_BATCHES}.csv" >> "$METRICS_COMBINED"
    done
    echo "✅ Combined metrics: $METRICS_COMBINED"
    
    # Calculate overall statistics
    echo ""
    echo "========================================="
    echo "Overall Statistics"
    echo "========================================="
    python - <<EOF
import pandas as pd

# Read combined metrics
metrics = pd.read_csv("$METRICS_COMBINED")

# Calculate weighted averages
total_samples = metrics['samples'].sum()
weighted_accuracy = (metrics['accuracy'] * metrics['samples']).sum() / total_samples
weighted_precision = (metrics['precision'] * metrics['samples']).sum() / total_samples
weighted_recall = (metrics['recall'] * metrics['samples']).sum() / total_samples
weighted_f1 = (metrics['f1'] * metrics['samples']).sum() / total_samples
total_agreement = metrics['agreement'].sum()
total_disagreement = metrics['disagreement'].sum()

print(f"Total Samples: {total_samples:,}")
print(f"Accuracy: {weighted_accuracy:.4f}")
print(f"Precision (HATE): {weighted_precision:.4f}")
print(f"Recall (HATE): {weighted_recall:.4f}")
print(f"F1 (HATE): {weighted_f1:.4f}")
print(f"Agreement: {total_agreement:,} ({total_agreement/total_samples*100:.2f}%)")
print(f"Disagreement: {total_disagreement:,} ({total_disagreement/total_samples*100:.2f}%)")
EOF
else
    echo "⚠️  No metrics files found!"
fi

# Clean up temp directory
echo ""
echo "🧹 Cleaning up temporary files..."
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "✅ Removed temp directory"
fi

echo ""
echo "========================================="
echo "✨ Labeling Complete!"
echo "========================================="
echo "Final outputs:"
echo "  📄 Labeled data: $OUTPUT_COMBINED"
echo "  📊 Metrics: $METRICS_COMBINED"
echo ""
echo "All temporary batch files have been cleaned up."
