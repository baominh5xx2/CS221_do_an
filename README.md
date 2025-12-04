# Vietnamese Hate Speech Detection using PhoBERT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of hate speech detection models for Vietnamese language using PhoBERT transformer architecture. The models are evaluated on three benchmark datasets: ViHSD, ViCTSD, and ViHOS.

## 📋 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Hate speech detection is a critical task in natural language processing, especially for low-resource languages like Vietnamese. This project implements a fine-tuned PhoBERT model for detecting hate speech across multiple Vietnamese datasets.

**Key Features:**
- 🚀 State-of-the-art PhoBERT-based architecture
- 📊 Support for multiple Vietnamese hate speech datasets
- 🔧 Configurable training pipeline with early stopping
- 📈 Comprehensive evaluation metrics and logging
- 💾 Automatic model checkpointing
- 🎛️ GPU memory tracking and optimization

## 📚 Datasets

This project supports three Vietnamese hate speech detection datasets:

### 1. ViHSD (Vietnamese Hate Speech Detection)
- **Source:** [visolex/ViHSD](https://huggingface.co/datasets/visolex/ViHSD)
- **Task:** Multi-class hate speech classification
- **Classes:** Multiple hate speech categories

### 2. ViCTSD (Vietnamese Constructive and Toxic Speech Detection)
- **Source:** [tarudesu/ViCTSD](https://huggingface.co/datasets/tarudesu/ViCTSD)
- **Task:** Binary toxicity classification
- **Classes:** NONE (0), TOXIC (1)

### 3. ViHOS (Vietnamese Hate and Offensive Spans)
- **Source:** [ViHOS GitHub](https://github.com/phusroyal/ViHOS)
- **Task:** Hate span detection
- **Classes:** CLEAN (0), HAS_HATE_SPANS (1)

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/vietnamese-hate-speech-detection.git
cd vietnamese-hate-speech-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Hugging Face token:**
```bash
# Create a .env file and add your token
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## 🚀 Quick Start

### Using Jupyter Notebook

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Open `notebooks/train.ipynb`** and run the cells sequentially.

### Using Python Scripts

1. **Train on a specific dataset:**
```bash
python src/train.py --dataset ViHSD --epochs 10 --batch_size 16
```

2. **Evaluate a trained model:**
```bash
python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD
```

3. **Run inference:**
```bash
python src/inference.py --model_path models/ViHSD_phobert --text "Your Vietnamese text here"
```

## 🎓 Training

### Configuration

Training parameters can be configured in `src/config.py` or passed as command-line arguments:

```python
# Key hyperparameters
model_name = "vinai/phobert-base"
max_length = 256
batch_size = 16
epochs = 10
learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
patience = 3  # Early stopping patience
```

### Training Process

The training pipeline includes:
- ✅ Automatic data loading and preprocessing
- ✅ Cosine learning rate scheduling with warmup
- ✅ Early stopping based on validation F1 score
- ✅ GPU memory monitoring
- ✅ Comprehensive metrics logging
- ✅ Model checkpointing

### Example Training Command

```bash
python src/train.py \
    --dataset ViHSD \
    --model_name vinai/phobert-base \
    --max_length 256 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --patience 3 \
    --seed 42
```

## 📊 Evaluation

### Metrics

Models are evaluated using:
- **Accuracy:** Overall classification accuracy
- **Macro F1:** Primary metric for model selection
- **Precision, Recall, F1:** Per-class metrics
- **Classification Report:** Detailed per-class performance

### Running Evaluation

```bash
python src/evaluate.py \
    --model_path models/ViHSD_phobert \
    --dataset ViHSD \
    --output_dir results/
```

Results are saved in CSV format:
- `epoch_metrics.csv`: Per-epoch training metrics
- `run_summary.csv`: Overall run statistics
- `test_report.csv`: Detailed test set classification report

## �️ Bash Scripts

This repository includes several bash scripts to automate common tasks. All scripts feature **real-time logging**, colored output, and automatic log saving.

### 1. Data Preparation
Checks environment, creates necessary directories, and verifies data availability.
```bash
bash scripts/download_data.sh
```

### 2. Batch Experiments
Trains classification models on all three datasets (ViHSD, ViCTSD, ViHOS) sequentially. Useful for reproducing all results in one go.
```bash
bash scripts/run_experiments.sh
```

### 3. Encoder Training (Domain Adaptation)
Fine-tunes the PhoBERT encoder using Masked Language Modeling (MLM) on a specific dataset. This adapts the pretrained model to the specific language/slang of the hate speech datasets.
```bash
# Train encoder on ViHSD
bash scripts/run_train_encoder.sh --dataset ViHSD --epochs 3

# Train encoder on ViCTSD with custom settings
bash scripts/run_train_encoder.sh --dataset ViCTSD --epochs 5 --learning_rate 1e-5
```

## �📈 Results

### Performance Summary

| Dataset | Accuracy | Macro F1 | Precision | Recall |
|---------|----------|----------|-----------|--------|
| ViHSD   | TBD      | TBD      | TBD       | TBD    |
| ViCTSD  | TBD      | TBD      | TBD       | TBD    |
| ViHOS   | TBD      | TBD      | TBD       | TBD    |

*Note: Run the training scripts to populate these results.*

### Training Curves

Training curves and visualizations can be found in the `results/` directory after training.

## 📁 Project Structure

```
vietnamese-hate-speech-detection/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
│
├── data/                      # Data directory (auto-downloaded)
│   ├── raw/                   # Raw datasets
│   └── processed/             # Preprocessed data
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   ├── data_loader.py         # Dataset loading utilities
│   ├── model.py               # Model definitions
│   ├── train.py               # Training script (Classification)
│   ├── train_encoder.py       # Encoder training script (MLM)
│   ├── evaluate.py            # Evaluation script
│   ├── inference.py           # Inference script
│   └── utils.py               # Helper functions
│
├── notebooks/                 # Jupyter notebooks
│   ├── train.ipynb            # Training notebook
│   ├── eda.ipynb              # Exploratory data analysis
│   └── inference_demo.ipynb   # Inference demonstration
│
├── models/                    # Saved models
│   ├── ViHSD_phobert/
│   ├── ViCTSD_phobert/
│   └── ViHOS_phobert/
│
├── results/                   # Experiment results
│   ├── figures/               # Plots and visualizations
│   └── metrics/               # Metric logs
│
├── logs/                      # Execution logs
│   ├── experiments/           # Training logs
│   └── mlm/                   # Encoder training logs
│
├── tests/                     # Unit tests
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_utils.py
│
└── scripts/                   # Utility scripts
    ├── download_data.sh       # Data download script
    ├── run_experiments.sh     # Batch experiment runner
    └── run_train_encoder.sh   # Encoder training runner
```

## 🔬 Reproducibility

To ensure reproducibility:

1. **Set random seeds:**
```python
import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

2. **Use the same environment:**
```bash
pip install -r requirements.txt
```

3. **Document your runs:**
All experiments are automatically logged with timestamps and configurations.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{vietnamese-hate-speech-2024,
  author = {Your Name},
  title = {Vietnamese Hate Speech Detection using PhoBERT},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/vietnamese-hate-speech-detection}}
}
```

### Dataset Citations

**ViHSD:**
```bibtex
@inproceedings{vihsd,
  title={ViHSD: Vietnamese Hate Speech Detection Dataset},
  author={...},
  booktitle={...},
  year={...}
}
```

**ViCTSD:**
```bibtex
@inproceedings{victsd,
  title={ViCTSD: Vietnamese Constructive and Toxic Speech Detection},
  author={...},
  booktitle={...},
  year={...}
}
```

**ViHOS:**
```bibtex
@inproceedings{vihos,
  title={ViHOS: Vietnamese Hate and Offensive Spans},
  author={...},
  booktitle={...},
  year={...}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch) for PhoBERT
- Dataset creators for ViHSD, ViCTSD, and ViHOS
- Hugging Face for the Transformers library

## 📧 Contact

For questions or feedback, please open an issue or contact:
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)

---

**Note:** This is a research project. The models and code are provided as-is for academic and research purposes.
