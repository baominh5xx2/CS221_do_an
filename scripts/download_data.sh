#!/bin/bash

# Download datasets script with real-time logging
# This script helps download and prepare the datasets

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
echo -e "${MAGENTA}║  Vietnamese Hate Speech Detection - Data Preparation      ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "Starting data preparation..."
echo ""

# Create data directories
log "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs

if [ -d "data/raw" ] && [ -d "data/processed" ]; then
    log_success "Created data directories"
    echo "  📁 data/raw/"
    echo "  📁 data/processed/"
else
    log_error "Failed to create data directories"
    exit 1
fi

echo ""
log_info "Dataset Information:"
echo ""
echo -e "${YELLOW}Supported Datasets:${NC}"
echo "  1. ViHSD (Vietnamese Hate Speech Detection)"
echo "     Source: visolex/ViHSD (Hugging Face)"
echo "     Type: Multi-class hate speech classification"
echo ""
echo "  2. ViCTSD (Vietnamese Constructive and Toxic Speech Detection)"
echo "     Source: tarudesu/ViCTSD (Hugging Face)"
echo "     Type: Binary toxicity classification"
echo ""
echo "  3. ViHOS (Vietnamese Hate and Offensive Spans)"
echo "     Source: phusroyal/ViHOS (GitHub)"
echo "     Type: Hate span detection"
echo ""

log_info "Datasets will be automatically downloaded when you run the training script."
echo ""

# Check prerequisites
log "Checking prerequisites..."
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    log_success ".env file found"
    
    # Check if HF_TOKEN is set
    if grep -q "HF_TOKEN=" .env; then
        log_success "HF_TOKEN is configured in .env"
    else
        log_warning "HF_TOKEN not found in .env file"
        echo "  Please add your Hugging Face token to .env:"
        echo "  ${CYAN}HF_TOKEN=your_token_here${NC}"
    fi
else
    log_warning ".env file not found"
    echo "  Creating .env from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        log_success "Created .env file from template"
        log_warning "Please edit .env and add your HF_TOKEN"
    else
        log_error ".env.example not found"
    fi
fi

echo ""

# Check if requirements are installed
log "Checking Python dependencies..."
if python -c "import transformers, datasets, torch" 2>/dev/null; then
    log_success "Required Python packages are installed"
else
    log_warning "Some required packages may be missing"
    echo "  Install with: ${CYAN}pip install -r requirements.txt${NC}"
fi

echo ""

# Check Python version
log "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Check if GPU is available
log "Checking GPU availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    log_success "GPU available: $GPU_NAME"
else
    log_warning "No GPU detected. Training will use CPU (slower)"
fi

echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
log_info "Next Steps:"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  1. Ensure HF_TOKEN is set in .env file"
echo "  2. Install dependencies: ${CYAN}pip install -r requirements.txt${NC}"
echo "  3. Run training: ${CYAN}python src/train.py --dataset ViHSD${NC}"
echo "  4. Or use notebook: ${CYAN}jupyter notebook notebooks/train.ipynb${NC}"
echo ""

log_success "Data preparation complete! 🎉"
echo ""
