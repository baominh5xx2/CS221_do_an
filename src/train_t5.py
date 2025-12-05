"""
Training script for T5/ViT5 models on Vietnamese hate speech detection.
Based on the original ViHateT5 paper implementation using Seq2SeqTrainer.

Usage:
    python src/train_t5.py --dataset ViHSD --model_name VietAI/vit5-base --epochs 5
    python src/train_t5.py --dataset Minhbao5xx2/VOZ-HSD_2M --split balanced --model_name google/t5-base
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from datasets import Dataset

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = 'False'

from data_loader import load_dataset_by_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train T5/ViT5 for hate speech detection")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (ViHSD, ViCTSD, ViHOS, Minhbao5xx2/VOZ-HSD_2M)")
    parser.add_argument("--split", type=str, default=None,
                       help="For VOZ-HSD_2M: 'balanced' or 'hate_only'")
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base",
                       help="T5 model (google/t5-base, VietAI/vit5-base, VietAI/vit5-large)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for model")
    
    return parser.parse_args()


def map_data_vihsd(df):
    """Map ViHSD dataset to source-target format."""
    map_labels = {
        0: "CLEAN",
        1: "OFFENSIVE",
        2: "HATE",
    }
    df = df.copy()
    df["source"] = df["free_text"].apply(lambda x: "hate-speech-detection: " + str(x))
    df["target"] = df["label_id"].apply(lambda x: map_labels[x])
    return df[["source", "target"]]


def map_data_victsd(df):
    """Map ViCTSD dataset to source-target format."""
    map_labels = {
        0: "NONE",
        1: "TOXIC",
    }
    df = df.copy()
    df["source"] = df["Comment"].apply(lambda x: "toxic-speech-detection: " + str(x))
    df["target"] = df["Toxicity"].apply(lambda x: map_labels[x])
    return df[["source", "target"]]


def map_data_vihos(df):
    """Map ViHOS dataset to source-target format."""
    def process_spans(lst):
        if lst == '[]' or pd.isna(lst) or lst == '':
            return []
        lst = [int(x) for x in str(lst).strip("[]'").split(', ') if x.strip()]
        if not lst:
            return []
        result = []
        temp = [lst[0]]
        for i in range(1, len(lst)):
            if lst[i] == lst[i-1] + 1:
                temp.append(lst[i])
            else:
                result.append(temp)
                temp = [lst[i]]
        result.append(temp)
        return result

    def add_tags(text, indices):
        if indices == '[]' or pd.isna(indices) or indices == '':
            return text
        indices = process_spans(indices)
        if not indices:
            return text
        for i in range(len(indices)):
            text = text[:indices[i][0]] + "[HATE]" + text[indices[i][0]:]
            text = text[:indices[i][-1]+7] + "[HATE]" + text[indices[i][-1]+7:]
            for j in range(i + 1, len(indices)):
                indices[j] = [x + 12 for x in indices[j]]
        return text

    df = df.copy()
    df["source"] = df["content"].apply(lambda x: "hate-spans-detection: " + str(x))
    df["target"] = [add_tags(str(df['content'].iloc[i]), df['index_spans'].iloc[i]) 
                    for i in range(len(df))]
    return df[["source", "target"]]


def map_data_vozhsd(df):
    """Map VOZ-HSD 2M dataset to source-target format."""
    map_labels = {
        0: "NONE",
        1: "HATE",
    }
    df = df.copy()
    df["source"] = df["texts"].apply(lambda x: "hate-speech-detection: " + str(x))
    df["target"] = df["labels"].apply(lambda x: map_labels[x])
    return df[["source", "target"]]


def map_dataset(df, dataset_name):
    """Map dataset to source-target format based on dataset name."""
    if "VOZ-HSD" in dataset_name or "VOZ_HSD" in dataset_name:
        return map_data_vozhsd(df)
    elif dataset_name == "ViHSD":
        return map_data_vihsd(df)
    elif dataset_name == "ViCTSD":
        return map_data_victsd(df)
    elif dataset_name == "ViHOS":
        return map_data_vihos(df)
    else:
        # Default: assume binary classification with 'texts' and 'labels' columns
        return map_data_vozhsd(df)


def main():
    """Main training function for T5 using Seq2SeqTrainer."""
    args = parse_args()
    
    # Device info
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Print config
    print("\n" + "=" * 80)
    print("T5 Training Configuration (Seq2SeqTrainer):")
    print("=" * 80)
    print(f"  Dataset        : {args.dataset}")
    if args.split:
        print(f"  Split          : {args.split}")
    print(f"  Model          : {args.model_name}")
    print(f"  Max Length     : {args.max_length}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Learning Rate  : {args.learning_rate}")
    print(f"  Dev Ratio      : {args.dev_ratio}")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📚 Loading {args.dataset} dataset...")
    train_df, val_df, test_df, metadata = load_dataset_by_name(args.dataset, args.split, args.dev_ratio)
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples  : {len(val_df)}")
    print(f"  Test samples : {len(test_df)}")
    
    # Map to source-target format
    print("\n🔄 Mapping datasets to source-target format...")
    train_df = map_dataset(train_df, args.dataset)
    val_df = map_dataset(val_df, args.dataset)
    test_df = map_dataset(test_df, args.dataset)
    
    # Drop NA
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    # Convert to HuggingFace Dataset
    train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_data = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_data = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # Load tokenizer and model
    print(f"\n🤖 Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Check if model is Flax-based (like ViHateT5)
    if "ViHateT5" in args.model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, from_flax=True)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Preprocess function
    def preprocess_function(sample, padding="max_length"):
        model_inputs = tokenizer(
            sample["source"], 
            max_length=args.max_length, 
            padding=padding, 
            truncation=True
        )
        labels = tokenizer(
            sample["target"], 
            max_length=args.max_length, 
            padding=padding, 
            truncation=True
        )
        # Replace pad_token_id with -100 for loss calculation
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize datasets
    print("\n🔢 Tokenizing datasets...")
    train_tokenized = train_data.map(
        preprocess_function, 
        batched=True, 
        remove_columns=train_data.column_names
    )
    val_tokenized = val_data.map(
        preprocess_function, 
        batched=True, 
        remove_columns=val_data.column_names
    )
    test_tokenized = test_data.map(
        preprocess_function, 
        batched=True, 
        remove_columns=test_data.column_names
    )
    print(f"  Keys of tokenized dataset: {list(train_tokenized.features)}")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        model_short = args.model_name.split("/")[-1]
        dataset_short = args.dataset.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/{dataset_short}_{model_short}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments (following original paper)
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate F1 macro
        try:
            f1_macro = f1_score(decoded_labels, decoded_preds, average='macro')
        except:
            # If labels don't match exactly, fallback to simple comparison
            f1_macro = sum([1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()]) / len(decoded_preds)
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        gen_len_mean = np.mean(prediction_lens)
        
        return {
            'f1_macro': round(f1_macro * 100, 4), 
            'gen_len': round(gen_len_mean, 4)
        }
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    print(f"  Test F1 Macro: {test_results.get('eval_f1_macro', 'N/A')}")
    print(f"  Test Loss    : {test_results.get('eval_loss', 'N/A')}")
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save test results
    results_df = pd.DataFrame([test_results])
    results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
