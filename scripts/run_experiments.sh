#!/bin/bash

# Run experiments on all datasets with real-time logging
# This script trains models on all three datasets with detailed progress tracking

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print with timestamp
log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

# Print header
echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║  Vietnamese Hate Speech Detection - Batch Experiments     ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=2e-5
PATIENCE=3
SEED=42

# Datasets to train on
DATASETS=("ViHSD" "ViCTSD" "ViHOS")

# Create logs directory
LOGS_DIR="logs/experiments"
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="$LOGS_DIR/batch_experiment_$TIMESTAMP.log"

log_info "Logging to: $MAIN_LOG"
echo ""

# Log configuration
log "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Patience: $PATIENCE"
echo "  Seed: $SEED"
echo "  Datasets: ${DATASETS[@]}"
echo ""

# Save configuration to log
{
    echo "Batch Experiment Started: $(date)"
    echo "Configuration:"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Patience: $PATIENCE"
    echo "  Seed: $SEED"
    echo ""
} > "$MAIN_LOG"

# Track overall progress
TOTAL_DATASETS=${#DATASETS[@]}
COMPLETED=0
FAILED=0

# Train on each dataset
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_NUM=$((i + 1))
    
    echo ""
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    log_info "Training on ${YELLOW}$DATASET${NC} (${DATASET_NUM}/${TOTAL_DATASETS})"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Create dataset-specific log
    DATASET_LOG="$LOGS_DIR/${DATASET}_$TIMESTAMP.log"
    
    # Record start time
    START_TIME=$(date +%s)
    log "Started training at $(date '+%H:%M:%S')"
    
    # Run training with real-time output and logging
    # Use 'tee' to show output in terminal AND save to log file
    if python src/train.py \
        --dataset "$DATASET" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --patience $PATIENCE \
        --seed $SEED 2>&1 | tee "$DATASET_LOG"; then
        
        # Calculate duration
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))
        
        log_success "Completed training on $DATASET in ${MINUTES}m ${SECONDS}s"
        COMPLETED=$((COMPLETED + 1))
        
        # Log to main log
        echo "[$DATASET] SUCCESS - Duration: ${MINUTES}m ${SECONDS}s" >> "$MAIN_LOG"
    else
        # Training failed
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))
        
        log_error "Failed training on $DATASET after ${MINUTES}m ${SECONDS}s"
        FAILED=$((FAILED + 1))
        
        # Log to main log
        echo "[$DATASET] FAILED - Duration: ${MINUTES}m ${SECONDS}s" >> "$MAIN_LOG"
        
        log_warning "Continuing with next dataset..."
    fi
    
    echo ""
done

# Print summary
echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
log_info "Batch Experiment Summary"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Total datasets: ${TOTAL_DATASETS}"
echo -e "  ${GREEN}Completed: ${COMPLETED}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}Failed: ${FAILED}${NC}"
fi
echo ""
log_info "Results saved in models/ directory"
log_info "Logs saved in $LOGS_DIR/"
echo ""

# Final status
if [ $FAILED -eq 0 ]; then
    log_success "All experiments completed successfully! 🎉"
    echo "All experiments completed successfully!" >> "$MAIN_LOG"
    exit 0
else
    log_warning "Some experiments failed. Check logs for details."
    echo "Some experiments failed." >> "$MAIN_LOG"
    exit 1
fi
