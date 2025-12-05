# Vietnamese Hate Speech Detection (PhoBERT)

Minimal intro: fine-tune PhoBERT (and compatible HF models) to flag Vietnamese hate/toxic speech on ViHSD, ViCTSD, and ViHOS. Primary metric: macro F1. Pipelines include cosine LR with warmup, early stopping, GPU VRAM tracking, and CSV logging (per-epoch, run summary, test report).

## Contents
- Overview
- Datasets
- Install
- Quick Start (scripts + notebook)
- Scripts (detailed usage)
- Training / Evaluation
- Project layout
- License

## Overview
- Task: classify/flag Vietnamese hate or toxicity; macro F1 used for model selection.
- Models: PhoBERT-base by default; any HF checkpoint works.
- Data: ViHSD (multi-class), ViCTSD (binary toxicity), ViHOS (span -> binary has_hate).
- Logging: `epoch_metrics.csv`, `run_summary.csv`, `test_report.csv`; GPU peak VRAM per epoch.

## Datasets
- ViHSD: https://huggingface.co/datasets/visolex/ViHSD — multi-class.
- ViCTSD: https://huggingface.co/datasets/tarudesu/ViCTSD — binary toxicity.
- ViHOS: https://github.com/phusroyal/ViHOS — span labels converted to binary has_hate.
- **VOZ-HSD 2M**: https://huggingface.co/datasets/Minhbao5xx2/VOZ-HSD_2M — 2M Vietnamese samples with 2 splits:
  - `hate_only`: All hate speech samples (class 1 only)
  - `balanced`: Balanced dataset with equal class 0 and class 1 samples

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "HF_TOKEN=your_hf_token" > .env
```

## Quick Start
### Notebook
Open `notebooks/train.ipynb` and run cells in order (EDA -> config -> train -> eval -> logging).

### Scripts (common cases)
```bash
# Classification training
python src/train.py --dataset ViHSD --model_name vinai/phobert-base --epochs 10 --batch_size 16

# Train on VOZ-HSD 2M (hate only)
python src/train.py --dataset Minhbao5xx2/VOZ-HSD_2M --split hate_only --epochs 5 --batch_size 16

# Train on VOZ-HSD 2M (balanced)
python src/train.py --dataset Minhbao5xx2/VOZ-HSD_2M --split balanced --epochs 5 --batch_size 16

# Evaluation
python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD --output_dir results/

# Inference
python src/inference.py --model_path models/ViHSD_phobert --text "Noi dung can du doan"

# Encoder domain adaptation (MLM)
bash scripts/run_train_encoder.sh --dataset ViHSD --epochs 3 --batch_size 16 --learning_rate 2e-5

# Batch all datasets (classification)
bash scripts/run_experiments.sh
```
```

## Scripts (detailed)

### `scripts/run_train_encoder.sh` — MLM domain adaptation
Adapts PhoBERT on a chosen dataset with Masked Language Modeling.
```bash
bash scripts/run_train_encoder.sh \
  --dataset ViHSD \          # ViHSD | ViCTSD | ViHOS | ViHSD_processed
  --model_name vinai/phobert-base \ 
  --epochs 3 \               # MLM epochs
  --batch_size 16 \          # MLM batch size
  --learning_rate 2e-5 \     # MLM LR
  --weight_decay 0.01 \      # optional
  --mlm_probability 0.15 \   # mask ratio
  --max_length 256 \         # max tokens
  --output_dir models/ViHSD_encoder_mlm  # optional override
```
Note: PhoBERT requires Vietnamese word segmentation; `underthesea` is enforced in code when using PhoBERT.

### `scripts/run_experiments.sh` — train classifiers on all datasets
Runs classification training sequentially for ViHSD, ViCTSD, ViHOS with shared hyperparameters. Edit defaults inside if needed.
```bash
bash scripts/run_experiments.sh
```
Outputs: `models/<dataset>_phobert` plus CSV metrics.

### `scripts/run_label_dataset.sh` — auto-label VOZ-HSD
Splits inference into batches and merges outputs.
- Edit in-script vars: `MODEL_PATH`, `TOTAL_BATCHES` (default 10), `BATCH_SIZE` (default 32).
```bash
bash scripts/run_label_dataset.sh
```
Manual single-batch:
```bash
python src/label_dataset.py \
  --model_path models/ViHSD_processed_phobert-base_YYYYMMDD_HHMMSS \
  --split train --batch_idx 0 --total_batches 1 --batch_size 32
```
## Training (classification)

### Classification with PhoBERT
Key flags for `src/train.py`:
```bash
python src/train.py \
  --dataset ViHSD \            # ViHSD | ViCTSD | ViHOS | ViHSD_processed | Minhbao5xx2/VOZ-HSD_2M
  --model_name vinai/phobert-base \
  --max_length 256 \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --patience 3 \
  --seed 42
```
Pipeline highlights:
- Auto dataset load + column resolution.
- Cosine LR with warmup; macro F1 for early stopping.
- GPU peak VRAM + LR per epoch tracked.
- Best checkpoint saved to `models/<dataset>_phobert`.
- Logs: `epoch_metrics.csv`, `run_summary.csv`, `test_report.csv`.

### T5 Models
T5/ViT5 training uses a separate script with text-to-text format (encoder-decoder architecture).

**Supported Models:**
| Model | HuggingFace | Params | Default Config |
|-------|-------------|--------|----------------|
| `t5-small` | google/t5-small | 60M | batch=16, lr=1e-4, max_len=512 |
| `t5-base` | google/t5-base | 220M | batch=8, lr=1e-4, max_len=512 |
| `t5-large` | google/t5-large | 770M | batch=4, lr=5e-5, max_len=512 |
| `vit5-base` | VietAI/vit5-base | 220M | batch=8, lr=1e-4, max_len=512 |
| `vit5-large` | VietAI/vit5-large | 770M | batch=4, lr=5e-5, max_len=512 |
| `vit5-large-1024` | VietAI/vit5-large-1024-vietnews | 770M | batch=2, lr=3e-5, max_len=768 |

**Training with bash script (recommended):**
```bash
# ViT5 base on ViHSD (simplest)
bash scripts/run_train_t5.sh --dataset ViHSD --model vit5-base

# T5 base on VOZ-HSD balanced
bash scripts/run_train_t5.sh --dataset Minhbao5xx2/VOZ-HSD_2M --split balanced --model t5-base

# Custom dev ratio
bash scripts/run_train_t5.sh --dataset ViHSD --model vit5-base --dev_ratio 0.15

# T5 small (fastest, for quick testing)
bash scripts/run_train_t5.sh --dataset ViHSD --model t5-small --epochs 2

# Full custom config
bash scripts/run_train_t5.sh \
  --dataset Minhbao5xx2/VOZ-HSD_2M \
  --split hate_only \
  --model vit5-base \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 5e-3 \
  --max_length 256 \
  --dev_ratio 0.1 \
  --output_dir models/vit5_balanced_custom
```

**Direct Python usage:**
```bash
python src/train_t5.py \
  --dataset Minhbao5xx2/VOZ-HSD_2M \
  --split balanced \
  --model_name VietAI/vit5-base \
  --max_length 512 \
  --batch_size 8 \
  --epochs 5 \
  --learning_rate 1e-4 \
  --dev_ratio 0.1
```

**T5 Key Notes:**
- Uses text-to-text format: input "classify: <text>" → output "hate" or "not_hate"
- ViT5 models (Vietnamese-pretrained) recommended for best results
- max_length=512 is default; use 768 for longer texts

## Evaluation
```bash
python src/evaluate.py \
  --model_path models/ViHSD_phobert \
  --dataset ViHSD \
  --output_dir results/
```
Writes classification report and metrics to the output dir.

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

For questions or feedback, please open an issue or contact:
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)

---

**Note:** This is a research project. The models and code are provided as-is for academic and research purposes.
