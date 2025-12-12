# Vietnamese Hate Speech Detection

PhoBERT/ViSoBERT + T5/ViT5 pipelines for Vietnamese hate/toxic speech. Primary metric: macro F1. All runners are provided as bash scripts with full CLI parameters.

## Datasets
Predefined datasets:
- **ViHSD** (multi-class): https://huggingface.co/datasets/visolex/ViHSD
- **ViHSD_processed** (binary): https://huggingface.co/datasets/trinhtrantran122/ViHSD_processed
- **ViCTSD** (binary toxicity): https://huggingface.co/datasets/tarudesu/ViCTSD
- **ViHOS** (hate spans -> binary has_hate): https://github.com/phusroyal/ViHOS
- **VOZ-HSD 2M** (binary): https://huggingface.co/datasets/Minhbao5xx2/VOZ-HSD_2M
  - `hate_only`: 110k hate samples (label=1)
  - `balanced`: class 0/1 balanced

**Custom HuggingFace datasets**: You can also load any dataset from HuggingFace Hub by passing the dataset identifier (e.g., `username/dataset_name`). The code will auto-detect text and label columns.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### HuggingFace token (don't hardcode)
- Linux/Mac: `huggingface-cli login` or `export HF_TOKEN=...`
- Windows: Git Bash `huggingface-cli login` or `setx HF_TOKEN "hf_xxx"`
- `.env` (optional): `echo "HF_TOKEN=..." > .env` and don't commit

## Quick start (scripts)
```bash
# PhoBERT/BERT classification (default: ViHSD, phobert-base)
bash scripts/run_train.sh --dataset ViHSD

# T5/ViT5 classification
bash scripts/run_train_t5.sh --dataset ViHSD --model vit5-base

# Encoder classification
bash scripts/run_train_encoder.sh --dataset ViHSD --model phobert-base

# Auto-label VOZ-HSD with trained classifier
bash scripts/run_label_dataset.sh
```

## Full CLI reference (scripts)

### scripts/run_train.sh — PhoBERT/BERT classification
Required: --dataset. One-liner format:
```bash
bash scripts/run_train.sh --dataset ViHSD --model_name vinai/phobert-base --epochs 10 --batch_size 16 --max_length 256 --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.1 --patience 3 --seed 42 --output_dir models/ViHSD_phobert_custom
```
**Parameter reference:**
- `--dataset` (required): ViHSD | ViCTSD | ViHOS | ViHSD_processed | Minhbao5xx2/VOZ-HSD_2M | or any HuggingFace dataset (e.g., `username/dataset_name`)
- `--model_name`: vinai/phobert-base | vinai/phobert-large | uitnlp/visobert | bert-base-multilingual-cased
- `--epochs`, `--batch_size`, `--max_length`: int
- `--learning_rate`, `--weight_decay`, `--warmup_ratio`: float
- `--patience`: int (early stopping)
- `--seed`: int
- `--output_dir`: optional

### scripts/run_train_t5.sh — T5/ViT5 classification (Seq2SeqTrainer)
Required: --dataset. One-liner format:
```bash
bash scripts/run_train_t5.sh --dataset Minhbao5xx2/VOZ-HSD_2M --model vit5-base --epochs 5 --batch_size 8 --learning_rate 1e-4 --max_length 512 --dev_ratio 0.1 --output_dir models/vit5_custom
```
**Parameter reference:**
- `--dataset` (required): ViHSD | ViCTSD | ViHOS | ViHSD_processed | Minhbao5xx2/VOZ-HSD_2M | or any HuggingFace dataset (e.g., `username/dataset_name`)
- `--model`: t5-small | t5-base | t5-large | vit5-base | vit5-large | vit5-large-1024
- `--epochs`, `--batch_size`, `--max_length`: int
- `--learning_rate`, `--dev_ratio`: float
- `--output_dir`: optional

### scripts/run_train_encoder.sh — encoder classification (Trainer)
Required: --dataset, --model. One-liner format:
```bash
bash scripts/run_train_encoder.sh --dataset ViHSD_processed --model visobert --epochs 10 --batch_size 16 --learning_rate 1e-5 --weight_decay 0.01 --max_length 128 --output_dir models/ViHSD_encoder_cls_custom
```
You can also pass a direct HuggingFace model id:
```bash
bash scripts/run_train_encoder.sh --dataset ViHSD_processed --model uitnlp/visobert --epochs 10 --batch_size 16 --learning_rate 1e-5 --weight_decay 0.01 --max_length 128 --output_dir models/ViHSD_encoder_cls_custom
```
**Parameter reference:**
- `--dataset` (required): ViHSD | ViCTSD | ViHOS | ViHSD_processed | Minhbao5xx2/VOZ-HSD_2M | or any HuggingFace dataset (e.g., `username/dataset_name`)
- `--model` (required): preset key (phobert-base, phobert-large, visobert, t5-base, t5-large, vit5-base, vit5-large) OR a HuggingFace model id
- `--epochs`, `--batch_size`, `--max_length`: int (override preset)
- `--learning_rate`, `--weight_decay`: float (override preset)
- `--output_dir`: optional

### scripts/run_label_dataset.sh — auto-label VOZ-HSD
Edit script variables or export before run: MODEL_PATH (required), SPLIT (train|validation|test), TOTAL_BATCHES (1=sequential, >1=parallel), BATCH_SIZE, MAX_LENGTH, OUTPUT_DIR, MAX_SAMPLES (optional).
```bash
bash scripts/run_label_dataset.sh
```
Manual single-batch:
```bash
python src/label_dataset.py --model_path models/ViHSD_processed_phobert-base_YYYYMMDD_HHMMSS --split train --batch_idx 0 --total_batches 1 --batch_size 32 --max_length 256 --output_dir labeled_data/voz_hsd
```

### scripts/run_experiments.sh — batch classifiers
Runs PhoBERT classification on ViHSD, ViCTSD, ViHOS sequentially. Edit defaults inside script then:
```bash
bash scripts/run_experiments.sh
```

## Direct Python (advanced)
PhoBERT classification:
```bash
python src/train.py --dataset ViHSD_processed --model_name uitnlp/visobert --max_length 128 --batch_size 16 --epochs 15 --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.1 --patience 3 --seed 42 --output_dir models/custom_phobert
```

T5/ViT5 classification:
```bash
python src/train_t5.py --dataset Minhbao5xx2/VOZ-HSD_2M --model_name VietAI/vit5-base --max_length 512 --batch_size 8 --epochs 5 --learning_rate 1e-4 --dev_ratio 0.1 --output_dir models/vit5_custom
```

## Evaluation
```bash
python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD --output_dir results/
```

## Output Files (CSV Tracking)

After training, each model directory contains CSV files for tracking and reproducibility:

### PhoBERT/BERT Classification (`train.py`)
Output directory: `models/{dataset}_{model}_{timestamp}/`
- **`epoch_metrics.csv`**: Training metrics per epoch
  - Columns: `epoch`, `train_loss`, `val_loss`, `val_acc`, `val_f1`, `epoch_seconds`, `learning_rate`
- **`run_summary.csv`**: Overall training summary
  - Columns: `dataset`, `model`, `timestamp`, `best_val_f1`, `test_loss`, `test_acc`, `test_f1`, `training_minutes`, `epochs_trained`

### T5/ViT5 Classification (`train_t5.py`)
Output directory: `models/{dataset}_{model}_{timestamp}/`
- **`training_config.csv`**: Hyperparameters and dataset configuration
  - Columns: `dataset`, `model_name`, `max_length`, `batch_size`, `epochs`, `learning_rate`, `dev_ratio`, `train_samples`, `val_samples`, `test_samples`, `num_labels`, `text_col`, `label_col`
- **`training_history.csv`**: Metrics per epoch/step
  - Columns: `epoch`, `step`, `train_loss`, `train_runtime`, `train_samples_per_second`, `eval_loss`, `eval_accuracy`, `eval_f1_macro`, `eval_gen_len`, `learning_rate`
- **`test_results.csv`**: Final test set evaluation
  - Columns: `eval_loss`, `eval_accuracy`, `eval_f1_macro`, `eval_gen_len`, and other metrics
- **`run_summary.csv`**: Overall training summary
  - Columns: `dataset`, `model_name`, `timestamp`, `train_samples`, `val_samples`, `test_samples`, `best_eval_accuracy`, `best_eval_f1_macro`, `test_accuracy`, `test_f1_macro`, `test_loss`, `training_minutes`, `epochs_trained`, `batch_size`, `learning_rate`, `max_length`

### Auto-labeling (`label_dataset.py`)
Output directory: `labeled_data/{split}/`
- **`{split}_labeled.csv`**: Labeled dataset with predictions
  - Columns: Original dataset columns + `predicted_label`, `original_label`
- **`metrics_summary.csv`**: Batch metrics summary (when using parallel processing)
  - Columns: `batch_idx`, `total_batches`, `split`, `samples`, `accuracy`, `precision`, `recall`, `f1`, `agreement`, `disagreement`

### Model Files
All training scripts also save:
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `config.json`: Model configuration
- `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`: Tokenizer files
- `logs/`: TensorBoard logs directory

## Project layout
```
src/            # core code (train, evaluate, inference, data)
scripts/        # runners (train, experiments, labeling, encoder MLM)
notebooks/      # train.ipynb, eda.ipynb, inference_demo.ipynb
models/         # saved checkpoints
results/        # metrics/plots
logs/           # execution logs
```

## License
MIT. See `LICENSE`.

Research-only; no guarantees.
