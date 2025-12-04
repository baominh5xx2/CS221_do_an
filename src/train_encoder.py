"""
Train BERT Encoder using Masked Language Modeling (MLM).

This script performs domain adaptation by fine-tuning the encoder on the dataset
using the standard Masked Language Modeling (MLM) objective.
This helps the model understand the specific language/slang of the dataset better
before being used for downstream tasks (classification).

Usage:
    python src/train_encoder.py --dataset ViHSD --epochs 3 --batch_size 16
"""

import argparse
import math
import os
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from config import TrainConfig
from data_loader import load_dataset_by_name
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT Encoder (MLM)")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["ViHSD", "ViCTSD", "ViHOS"],
                       help="Dataset to train on")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base",
                       help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                       help="Ratio of tokens to mask for MLM")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    set_seed(args.seed)
    
    # Setup paths
    if args.output_dir is None:
        output_dir = Path("models") / f"{args.dataset}_encoder_mlm"
    else:
        output_dir = Path(args.output_dir)
        
    print(f"Training Encoder (MLM) on {args.dataset}...")
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    
    # 1. Load Data
    train_df, val_df, test_df, metadata = load_dataset_by_name(args.dataset)
    text_col = metadata["text_col"]
    
    # Convert pandas to Hugging Face Dataset
    # We only need the text column for MLM
    train_dataset = Dataset.from_pandas(train_df[[text_col]])
    val_dataset = Dataset.from_pandas(val_df[[text_col]])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
    # 3. Preprocessing
    def tokenize_function(examples):
        return tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True
        )
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
    
    # Data Collator for MLM (automatically masks tokens)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=args.mlm_probability
    )
    
    # 4. Trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(Path("logs") / "mlm"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/mlflow
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        dataloader_num_workers=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 5. Train
    print("Starting training...")
    train_result = trainer.train()
    
    # 6. Save
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    
    # Evaluate perplexity
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"Validation Perplexity: {perplexity:.2f}")
    
    # Save metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    
    print("Done! Encoder domain adaptation complete.")

if __name__ == "__main__":
    main()
