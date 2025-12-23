# Vietnamese Hate Speech Detection

A comprehensive pipeline for Vietnamese hate speech and toxic speech detection using PhoBERT/ViSoBERT and T5/ViT5 models. The project supports pretraining, fine-tuning, and evaluation with multiple datasets.

## Features

- **Pretraining**: T5 span corruption pretraining on Vietnamese text
- **Fine-tuning**: T5/ViT5 sequence-to-sequence fine-tuning for classification
- **Classification**: BERT-based (PhoBERT/ViSoBERT) classification models
- **Multi-dataset support**: ViHSD, ViCTSD, ViHOS, VOZ-HSD, and custom HuggingFace datasets
- **Comprehensive evaluation**: Automatic test set evaluation with detailed metrics

## Datasets

### Predefined Datasets

- **ViHSD** (multi-class): 3-class hate speech detection
  - Classes: CLEAN, OFFENSIVE, HATE
  - Source: https://huggingface.co/datasets/visolex/ViHSD

- **ViHSD_processed** (binary): Processed binary version
  - Source: https://huggingface.co/datasets/trinhtrantran122/ViHSD_processed

- **ViCTSD** (binary): Toxic speech detection
  - Classes: NONE, TOXIC
  - Source: https://huggingface.co/datasets/tarudesu/ViCTSD

- **ViHOS** (hate spans): Hate span detection task
  - Source: https://github.com/phusroyal/ViHOS

- **VOZ-HSD** (binary): Large-scale hate speech dataset
  - Splits: `balanced`, `hate_only`, `full`
  - Source: https://huggingface.co/datasets/Minhbao5xx2/re_VOZ-HSD

### Custom Datasets

You can load any dataset from HuggingFace Hub by passing the dataset identifier (e.g., `username/dataset_name`). The code will auto-detect text and label columns.

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. HuggingFace Authentication

For private models or pushing to Hub:

```bash
# Linux/Mac
huggingface-cli login
# or
export HF_TOKEN=your_token_here

# Windows (Git Bash)
huggingface-cli login
# or
setx HF_TOKEN "hf_xxx"

# Optional: Use .env file (don't commit)
echo "HF_TOKEN=your_token_here" > .env
```

## Quick Start

### T5 Pretraining (Span Corruption)

Pretrain T5/ViT5 models on Vietnamese text using span corruption objective:

```bash
# Basic usage with default settings
bash scripts/run_pretrain_t5.sh

# Custom dataset and parameters
bash scripts/run_pretrain_t5.sh \
    --dataset_name "Minhbao5xx2/re_VOZ-HSD" \
    --split_name "hate_only" \
    --max_samples 50000 \
    --output_dir "vihate_t5_pretrain"
```

**Parameters:**
- `--dataset_name`: Dataset identifier (e.g., `Minhbao5xx2/re_VOZ-HSD`) or `None` for local files
- `--split_name`: For VOZ-HSD: `balanced`, `hate_only`, or `full` (default: `balanced`)
- `--max_samples`: Maximum number of samples to use (optional)
- `--train_file`: Path to local training text file (one example per line)
- `--valid_file`: Path to local validation text file (one example per line)
- `--output_dir`: Output directory (default: `vihate_t5_pretrain`)

**Note:** The pretraining script is optimized for H200 GPUs (141GB HBM3) with large batch sizes. Adjust `per_device_train_batch_size` in `src/pre_train_t5.py` if using smaller GPUs.

### T5 Fine-tuning

Fine-tune T5/ViT5 models for sequence-to-sequence classification:

```bash
# Basic usage
bash scripts/run_train_t5.sh

# Custom configuration
bash scripts/run_train_t5.sh \
    --save_model_name "ViHateT5-custom" \
    --pre_trained_ckpt "VietAI/vit5-base" \
    --output_dir "outputs/t5_finetuned" \
    --batch_size 32 \
    --num_epochs 4 \
    --learning_rate 2e-4 \
    --gpu "0"
```

**Parameters:**
- `--save_model_name`: Name for the fine-tuned model (required for Hub push)
- `--pre_trained_ckpt`: Pre-trained checkpoint (default: `VietAI/vit5-base`)
- `--output_dir`: Output directory (default: `outputs/t5_finetuned`)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--gpu`: GPU device ID (default: "0")

**Note:** The script trains on ViHSD, ViCTSD, and ViHOS datasets combined, then evaluates on each test set separately.

### BERT-based Classification

Train PhoBERT/ViSoBERT models for classification:

```bash
# Basic usage
bash scripts/run_train_bert.sh --dataset ViHSD

# Full configuration
bash scripts/run_train_bert.sh \
    --dataset ViHSD \
    --model_name "vinai/phobert-base" \
    --max_length 256 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --patience 3 \
    --seed 42 \
    --output_dir "outputs/bert_custom"
```

**Parameters:**
- `--dataset` (required): Dataset name (ViHSD, ViCTSD, ViHOS, ViHSD_processed, Minhbao5xx2/VOZ-HSD_2M, or HuggingFace dataset)
- `--model_name`: Model identifier (default: `vinai/phobert-base`)
  - Options: `vinai/phobert-base`, `vinai/phobert-large`, `uitnlp/visobert`, `bert-base-multilingual-cased`
- `--max_length`: Maximum sequence length (default: 256)
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_ratio`: Warmup ratio (default: 0.1)
- `--patience`: Early stopping patience (default: 3)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory (auto-generated if not specified)

## Direct Python Usage

### T5 Pretraining

```bash
python src/pre_train_t5.py \
    --dataset_name "Minhbao5xx2/re_VOZ-HSD" \
    --split_name "hate_only" \
    --max_samples 50000
```

### T5 Fine-tuning

```bash
python src/train_t5.py \
    --save_model_name "ViHateT5-finetuned" \
    --pre_trained_ckpt "VietAI/vit5-base" \
    --output_dir "outputs/t5_finetuned" \
    --batch_size 32 \
    --num_epochs 4 \
    --learning_rate 2e-4 \
    --gpu "0"
```

### BERT Classification

```bash
python src/train_bert.py \
    --dataset ViHSD \
    --model_name "vinai/phobert-base" \
    --max_length 256 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --patience 3 \
    --seed 42 \
    --output_dir "outputs/bert_custom"
```

## Evaluation

Evaluate a trained model on a specific dataset:

```bash
python src/evaluate.py \
    --model_path "outputs/bert_custom" \
    --dataset ViHSD \
    --output_dir "results/"
```

## Output Files

### T5 Pretraining (`pre_train_t5.py`)

Output directory: `vihate_t5_pretrain/final/`
- Model weights: `pytorch_model.bin` or `model.safetensors`
- Model config: `config.json`
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`

### T5 Fine-tuning (`train_t5.py`)

Output directory: `outputs/t5_finetuned/`
- **Model files**: Same as pretraining
- **`results/evaluation_results.csv`**: Test set evaluation results
  - Columns: `Model`, `Task`, `Accuracy`, `Weighted F1 Score`, `Macro F1 Score`
  - Tasks: ViHSD, ViCTSD, ViHOS

### BERT Classification (`train_bert.py`)

Output directory: `outputs/bert_{dataset}_{timestamp}/`
- **`epoch_metrics.csv`**: Training metrics per epoch
  - Columns: `epoch`, `train_loss`, `val_loss`, `val_acc`, `val_f1`, `epoch_seconds`, `learning_rate`
- **`run_summary.csv`**: Overall training summary
  - Columns: `dataset`, `model`, `timestamp`, `best_val_f1`, `test_loss`, `test_acc`, `test_f1`, `training_minutes`, `epochs_trained`
- **Model files**: Same as above

## Project Structure

```
.
├── src/                    # Core source code
│   ├── pre_train_t5.py    # T5 span corruption pretraining
│   ├── train_t5.py         # T5 fine-tuning for classification
│   ├── train_bert.py       # BERT-based classification
│   ├── data_loader.py      # Dataset loading utilities
│   ├── evaluate.py         # Model evaluation
│   ├── inference.py        # Inference utilities
│   ├── model.py            # Model definitions
│   ├── utils.py            # Utility functions
│   └── t5_data_collator.py # T5 span corruption data collator
├── scripts/                # Training scripts
│   ├── run_pretrain_t5.sh  # T5 pretraining script
│   ├── run_train_t5.sh     # T5 fine-tuning script
│   └── run_train_bert.sh   # BERT training script
├── outputs/                # Model checkpoints
├── results/                # Evaluation results
├── data/                   # Local datasets (optional)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Checkpoints

### Pretrained Models

- **VietAI/vit5-base**: Vietnamese T5 base model
- **VietAI/vit5-large**: Vietnamese T5 large model
- **vinai/phobert-base**: Vietnamese PhoBERT base
- **vinai/phobert-large**: Vietnamese PhoBERT large
- **uitnlp/visobert**: ViSoBERT model

### Fine-tuned Models

After training, models are saved locally and can be pushed to HuggingFace Hub (if `--save_model_name` is provided in T5 training).

## Performance Tips

### GPU Optimization

- **H200 (141GB)**: Use default settings in `pre_train_t5.py` (batch_size=512)
- **A100 (40GB)**: Reduce batch_size to 128-256
- **V100 (16GB)**: Reduce batch_size to 32-64, enable gradient checkpointing
- **Smaller GPUs**: Use gradient accumulation, reduce max_length

### Memory Optimization

- Enable `gradient_checkpointing=True` for memory efficiency
- Use `bf16` or `fp16` mixed precision training
- Reduce `max_length` if encountering OOM errors
- Use `--max_samples` to limit dataset size during development

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: t5_data_collator**
   - Ensure `src/t5_data_collator.py` exists
   - Check Python path includes project root

2. **CUDA Out of Memory**
   - Reduce `batch_size` or `per_device_train_batch_size`
   - Enable `gradient_checkpointing`
   - Reduce `max_length`
   - Use gradient accumulation

3. **HuggingFace Authentication**
   - Run `huggingface-cli login`
   - Or set `HF_TOKEN` environment variable

4. **Dataset Loading Errors**
   - Check dataset name spelling
   - Verify HuggingFace dataset exists
   - Check internet connection for remote datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vietnamese_hate_speech_detection,
  title = {Vietnamese Hate Speech Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/vietnamese-hate-speech}
}
```

## License

MIT License. See `LICENSE` file for details.

**Research use only. No guarantees provided.**

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Datasets: ViHSD, ViCTSD, ViHOS, VOZ-HSD teams
- Models: VietAI, Vinai, UIT-NLP
- HuggingFace for the transformers library
