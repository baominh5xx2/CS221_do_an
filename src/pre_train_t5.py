
# pretrain_t5_span_corruption.py
# pip install -U transformers datasets accelerate torch

import os
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from data_loader import load_dataset_by_name
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch  # For torch.compile (optional optimization)

# file này bạn tải từ: huggingface/olm-training (link ở trên)
from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths  # noqa

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "VietAI/vit5-base"   # continual pretrain từ ViT5-base là 1 setting họ dùng trong ablation [file:23]
TEXT_COL = "text"


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ViHate T5 with span corruption")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name to load via load_dataset_by_name (e.g., 'Minhbao5xx2/re_VOZ-HSD')")
    parser.add_argument("--split_name", type=str, default=None, help="Split name for VOZ-HSD datasets: 'balanced', 'hate_only', or 'full' (default: 'balanced')")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (for limiting dataset size)")
    parser.add_argument("--train_file", type=str, default="data/train.txt", help="Path to training text file (one example per line)")
    parser.add_argument("--valid_file", type=str, default="data/valid.txt", help="Path to validation text file (one example per line)")
    return parser.parse_args()


args = parse_args()

# Validate dataset files exist only if not using dataset_name
if not args.dataset_name:
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}. Provide --train_file pointing to your dataset or use --dataset_name.")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}. Provide --valid_file pointing to your dataset or use --dataset_name.")

# ====== 1) Load data ======
# Option A (local): mỗi dòng 1 comment. Option B: load from existing codebase datasets via --dataset_name
if args.dataset_name:
    print(f"  Loading dataset from codebase/huggingface: {args.dataset_name}")
    train_df, val_df, test_df, metadata = load_dataset_by_name(
        args.dataset_name,
        split_name=args.split_name,
        max_samples=args.max_samples
    )

    # Determine text column
    text_col = metadata.get("text_col", TEXT_COL) if isinstance(metadata, dict) else TEXT_COL

    # Rename selected text column to TEXT_COL expected by tokenizer pipeline
    if text_col not in train_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataset returned by load_dataset_by_name")

    train_df = train_df[[text_col]].rename(columns={text_col: TEXT_COL}).reset_index(drop=True)
    val_df = val_df[[text_col]].rename(columns={text_col: TEXT_COL}).reset_index(drop=True)

    ds = DatasetDict({
        "train": HFDataset.from_pandas(train_df),
        "validation": HFDataset.from_pandas(val_df),
    })
else:
    # Fallback: load local text files (one example per line)
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    ds = load_dataset(
        "text",
        data_files={"train": str(train_path), "validation": str(valid_path)},
    )

# ====== 2) Load model/tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Optional: Compile model for H200 (PyTorch 2.0+, có thể tăng tốc 10-20%)
# Uncomment nếu dùng PyTorch >= 2.0 và muốn tối ưu thêm
# try:
#     model = torch.compile(model, mode="reduce-overhead")
#     print("  ✅ Model compiled with torch.compile")
# except Exception as e:
#     print(f"  ⚠️  torch.compile not available: {e}")

# ====== 3) T5 span-corruption setup ======
# Thường dùng noise_density=0.15, mean_span_len=3.0 (giống T5/mT5 phổ biến)
noise_density = 0.15
mean_noise_span_length = 3.0
input_length = 256  # paper dùng maxlen 256 cho model base [file:23]

expanded_input_length, target_length = compute_t5_input_and_target_lengths(
    inputs_length=input_length,
    noise_density=noise_density,
    mean_noise_span_length=mean_noise_span_length,
)

def tokenize_fn(batch):
    # Tokenize raw text thành fixed-length "expanded_input_length"
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=expanded_input_length,
        return_attention_mask=False,
    )

tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)

data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=noise_density,
    mean_noise_span_length=mean_noise_span_length,
    input_length=input_length,
    target_length=target_length,
    pad_token_id=tokenizer.pad_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ====== 4) Train args (optimized for H200 GPU - 141GB HBM3) ======
# H200 có thể handle batch size rất lớn, tối ưu để tận dụng memory và throughput
# NOTE: Nếu gặp OOM, giảm per_device_train_batch_size xuống (256, 128, 64...) hoặc tăng gradient_accumulation_steps
training_args = TrainingArguments(
    output_dir="vihate_t5_pretrain",
    num_train_epochs=10,
    learning_rate=5e-3,
    weight_decay=0.001,
    warmup_steps=2000,
    # H200: tăng batch size lớn để tận dụng 141GB memory
    # Có thể tăng lên 1024+ nếu model nhỏ, hoặc giảm nếu OOM
    per_device_train_batch_size=512,  # Tăng từ 128 lên 512 cho H200
    per_device_eval_batch_size=512,
    gradient_accumulation_steps=1,  # Giảm xuống 1 vì batch size đã lớn (tăng nếu OOM)
    # DataLoader optimizations cho H200
    dataloader_num_workers=8,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster GPU transfer
    # Memory & speed optimizations
    gradient_checkpointing=True,  # Trade compute for memory (cho phép batch size lớn hơn)
    bf16=True,  # H200 hỗ trợ tốt bf16
    bf16_full_eval=True,  # Use bf16 cho evaluation
    # Training settings
    evaluation_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    # Optimizer settings (fused Adam cho H200)
    optim="adamw_torch_fused",  # Fused optimizer cho tốc độ cao hơn
    # Additional H200 optimizations
    ddp_find_unused_parameters=False,  # Faster DDP nếu dùng multi-GPU
    remove_unused_columns=False,  # Keep columns for data collator
    # Max steps (optional, comment out if using epochs)
    # max_steps=100000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("vihate_t5_pretrain/final")
tokenizer.save_pretrained("vihate_t5_pretrain/final")