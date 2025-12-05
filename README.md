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

# Evaluation
python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD --output_dir results/

# Inference
python src/inference.py --model_path models/ViHSD_phobert --text "Noi dung can du doan"

# Encoder domain adaptation (MLM)


# Batch all datasets (classification)
## 📧 Contact
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
Key flags for `src/train.py`:
```bash
python src/train.py \
  --dataset ViHSD \            # ViHSD | ViCTSD | ViHOS | ViHSD_processed
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
