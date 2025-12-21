"""
Training script for T5/ViT5 models on Vietnamese hate speech detection.
Based on the original ViHateT5 paper implementation using Seq2SeqTrainer.

Usage:
    python src/train_t5.py --dataset ViHSD --model_name VietAI/vit5-base --epochs 5
    python src/train_t5.py --dataset Minhbao5xx2/VOZ-HSD_2M --model_name google/t5-base
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
import nltk
import torch
import psutil

# Download punkt tokenizer for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainerCallback,
)

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = 'False'

from data_loader import load_dataset_by_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train T5/ViT5 for hate speech detection")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name(s). For single dataset: ViHSD, ViCTSD, ViHOS, etc. "
                            "For multi-dataset training: comma-separated list like 'ViHSD,ViCTSD,ViHOS'")
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base",
                       help="T5 model (google/t5-base, VietAI/vit5-base, VietAI/vit5-large, tarudesu/ViHateT5-base)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.005,
                       help="Warmup ratio (0.0 = no warmup)")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                       help="Learning rate scheduler type (constant = no scheduling, keeps LR fixed)")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                       help="Optimizer (adafactor, adamw_torch, etc.)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay if we apply some")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0,
                       help="Label smoothing factor")
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
    # Convert label to int (handle float labels like 0.0, 1.0, 2.0)
    df["target"] = df["label_id"].apply(lambda x: map_labels[int(x)])
    return df[["source", "target"]]


def map_data_victsd(df):
    """Map ViCTSD dataset to source-target format."""
    map_labels = {
        0: "NONE",
        1: "TOXIC",
    }
    df = df.copy()
    df["source"] = df["Comment"].apply(lambda x: "toxic-speech-detection: " + str(x))
    # Convert label to int (handle float labels like 0.0, 1.0)
    df["target"] = df["Toxicity"].apply(lambda x: map_labels[int(x)])
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
    # Convert label to int (handle float labels like 0.0, 1.0)
    df["target"] = df["labels"].apply(lambda x: map_labels[int(x)])
    return df[["source", "target"]]


def map_data_generic(df, text_col, label_col, num_labels=None):
    """Map generic dataset to source-target format using detected columns."""
    df = df.copy()
    
    # Create label mapping based on number of labels
    if num_labels is None:
        # Auto-detect number of unique labels
        unique_labels = sorted(df[label_col].unique())
        num_labels = len(unique_labels)
    
    # Create label mapping
    if num_labels == 2:
        # Binary classification
        map_labels = {0: "NONE", 1: "HATE"}
    elif num_labels == 3:
        # Three-class classification
        map_labels = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    else:
        # Multi-class: use label values as strings
        unique_labels = sorted(df[label_col].unique())
        map_labels = {i: f"CLASS_{i}" for i in unique_labels}
    
    # Map labels
    df["source"] = df[text_col].apply(lambda x: "hate-speech-detection: " + str(x))
    # Convert label to int (handle float labels) before mapping
    df["target"] = df[label_col].apply(lambda x: map_labels.get(int(x), f"CLASS_{int(x)}"))
    
    return df[["source", "target"]]


def map_dataset(df, dataset_name, metadata=None):
    """Map dataset to source-target format based on dataset name or metadata."""
    # Handle Minhbao5xx2/re_VOZ-HSD specifically (has 'texts' and 'labels' columns)
    if dataset_name == "Minhbao5xx2/re_VOZ-HSD":
        return map_data_vozhsd(df)
    elif "VOZ-HSD" in dataset_name or "VOZ_HSD" in dataset_name:
        return map_data_vozhsd(df)
    elif dataset_name == "ViHSD":
        return map_data_vihsd(df)
    elif dataset_name == "ViCTSD":
        return map_data_victsd(df)
    elif dataset_name == "ViHOS":
        return map_data_vihos(df)
    elif metadata is not None:
        # Use metadata from HuggingFace dataset
        text_col = metadata.get("text_col", "text")
        label_col = metadata.get("label_col", "label")
        num_labels = metadata.get("num_labels", None)
        return map_data_generic(df, text_col, label_col, num_labels)
    else:
        # Default: assume binary classification with 'texts' and 'labels' columns
        return map_data_vozhsd(df)


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def get_gpu_memory_reserved():
    """Get reserved GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0

def get_ram_usage():
    """Get current RAM usage in GB."""
    return psutil.Process().memory_info().rss / 1e9

def main():
    """Main training function for T5 using Seq2SeqTrainer."""
    args = parse_args()
    
    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total VRAM: {total_vram:.1f} GB")
    
    # Track memory
    initial_vram = get_gpu_memory()
    initial_ram = get_ram_usage()
    print(f"   Initial VRAM used: {initial_vram:.2f} GB")
    print(f"   Initial RAM used: {initial_ram:.2f} GB")
    
    # Parse dataset list (support multi-dataset training)
    dataset_list = [d.strip() for d in args.dataset.split(",")]
    is_multi_dataset = len(dataset_list) > 1
    
    # Print config
    print("\n" + "=" * 80)
    print("T5 Training Configuration (Seq2SeqTrainer):")
    print("=" * 80)
    print(f"  Dataset(s)     : {args.dataset}")
    if is_multi_dataset:
        print(f"  Mode           : Multi-task training ({len(dataset_list)} datasets)")
    else:
        print(f"  Mode           : Single dataset training")
    print(f"  Model          : {args.model_name}")
    print(f"  Max Length     : {args.max_length}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Learning Rate  : {args.learning_rate}")
    print(f"  Scheduler      : {args.lr_scheduler_type}")
    print(f"  Warmup Ratio   : {args.warmup_ratio}")
    print(f"  Optimizer      : {args.optim}")
    
    # Store individual test datasets for per-dataset evaluation
    individual_test_dfs = {}
    individual_test_metadata = {}
    # Store individual validation datasets for per-epoch evaluation
    individual_val_dfs = {}
    individual_val_metadata = {}
    
    if is_multi_dataset:
        print(f"\n📚 Loading {len(dataset_list)} datasets for multi-task training...")
        print(f"  Datasets: {', '.join(dataset_list)}")
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for dataset_name in dataset_list:
            print(f"\n  Loading {dataset_name}...")
            train_df_single, val_df_single, test_df_single, metadata_single = load_dataset_by_name(dataset_name, dev_ratio=args.dev_ratio)
            
            # Map to source-target format
            train_df_single = map_dataset(train_df_single, dataset_name, metadata_single)
            val_df_single = map_dataset(val_df_single, dataset_name, metadata_single)
            test_df_single = map_dataset(test_df_single, dataset_name, metadata_single)
            
            # Drop NA
            train_df_single = train_df_single.dropna()
            val_df_single = val_df_single.dropna()
            test_df_single = test_df_single.dropna()
            
            print(f"    Train: {len(train_df_single)}, Val: {len(val_df_single)}, Test: {len(test_df_single)}")
            
            # Store individual test datasets for later evaluation
            individual_test_dfs[dataset_name] = test_df_single.copy()
            individual_test_metadata[dataset_name] = metadata_single.copy()
            # Store individual validation datasets for per-epoch evaluation
            individual_val_dfs[dataset_name] = val_df_single.copy()
            individual_val_metadata[dataset_name] = metadata_single.copy()
            
            train_dfs.append(train_df_single)
            val_dfs.append(val_df_single)
            test_dfs.append(test_df_single)
        
        # Concatenate all datasets
        print(f"\n🔄 Concatenating datasets...")
        train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
        test_df = pd.concat(test_dfs, axis=0, ignore_index=True)
        
        print(f"  Final Train samples: {len(train_df)}")
        print(f"  Final Val samples  : {len(val_df)}")
        print(f"  Final Test samples : {len(test_df)}")
        
        # Create combined metadata
        metadata = {
            "name": "+".join(dataset_list),
            "text_col": "source",  # Already mapped
            "label_col": "target",  # Already mapped
            "num_labels": "multi-task"
        }
    else:
        # Single dataset training (original behavior)
        dataset_name = dataset_list[0]
        print(f"\n📚 Loading {dataset_name} dataset...")
        train_df, val_df, test_df, metadata = load_dataset_by_name(dataset_name, dev_ratio=args.dev_ratio)
        
        # Check if dev_ratio was used (for datasets without predefined splits)
        uses_dev_ratio = (
            "VOZ-HSD_2M" in dataset_name or 
            dataset_name == "Minhbao5xx2/re_VOZ-HSD" or
            (dataset_name.count("/") == 1 and dataset_name not in ["ViHSD", "ViCTSD", "ViHOS", "ViHSD_processed"])
        )
        
        if uses_dev_ratio:
            print(f"  Dev Ratio      : {args.dev_ratio} (used for splitting)")
        else:
            print(f"  Dev Ratio      : {args.dev_ratio} (not used - dataset has predefined splits)")
        print("=" * 80)
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples  : {len(val_df)}")
        print(f"  Test samples : {len(test_df)}")
        
        # Map to source-target format
        print("\n🔄 Mapping datasets to source-target format...")
        train_df = map_dataset(train_df, dataset_name, metadata)
        val_df = map_dataset(val_df, dataset_name, metadata)
        test_df = map_dataset(test_df, dataset_name, metadata)
    
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
        print("  Loading from Flax weights (ViHateT5)...")
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
    
    # Tokenize individual validation datasets for per-epoch evaluation
    individual_val_tokenized = {}
    if is_multi_dataset and individual_val_dfs:
        print("\n🔢 Tokenizing individual validation datasets for per-epoch evaluation...")
        for dataset_name in dataset_list:
            val_df_single = individual_val_dfs[dataset_name]
            val_data_single = Dataset.from_pandas(val_df_single.reset_index(drop=True))
            val_tokenized_single = val_data_single.map(
                preprocess_function,
                batched=True,
                remove_columns=val_data_single.column_names
            )
            individual_val_tokenized[dataset_name] = val_tokenized_single
            print(f"  Tokenized {dataset_name} validation: {len(val_tokenized_single)} samples")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_short = args.model_name.split("/")[-1]
        if is_multi_dataset:
            dataset_short = "+".join([d.split("/")[-1] for d in dataset_list])
        else:
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        report_to="none",  # Disable wandb/tensorboard logging (use local logs only)
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Valid labels for normalization
    VALID = {"CLEAN", "OFFENSIVE", "HATE", "NONE", "TOXIC"}
    
    def norm(s):
        """Normalize label: strip, uppercase, take first word, validate."""
        s = (s or "").strip().upper()
        s = s.split()[0] if s else ""
        return s if s in VALID else "OTHER"
    
    # Custom callback to evaluate on individual datasets after each epoch
    class PerDatasetEvalCallback(TrainerCallback):
        def __init__(self, individual_val_tokenized, tokenizer, norm_func):
            self.individual_val_tokenized = individual_val_tokenized
            self.tokenizer = tokenizer
            self.norm = norm_func
            self.vocab_size = len(tokenizer)
            self.trainer = None  # Will be set by trainer
        
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            """Store trainer reference when training begins."""
            # Trainer is passed in kwargs
            if 'trainer' in kwargs:
                self.trainer = kwargs['trainer']
        
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            """Evaluate on each individual dataset after each epoch."""
            if not self.individual_val_tokenized:
                return
            
            # Get trainer from kwargs or use stored reference
            trainer = kwargs.get('trainer', self.trainer)
            if trainer is None:
                return
            
            print("\n" + "=" * 80)
            print(f"📊 PER-DATASET VALIDATION RESULTS (Epoch {int(state.epoch)})")
            print("=" * 80)
            
            per_dataset_f1 = {}
            
            for dataset_name, val_tokenized_single in self.individual_val_tokenized.items():
                # Get predictions for F1 calculation
                predictions = trainer.predict(val_tokenized_single)
                preds = predictions.predictions
                labels = predictions.label_ids
                
                # Decode predictions (stable approach - no nltk)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Normalize labels (strip and uppercase for comparison)
                decoded_preds = [pred.strip().upper() for pred in decoded_preds]
                decoded_labels = [label.strip().upper() for label in decoded_labels]
                
                # Calculate F1 macro
                f1_macro = f1_score(decoded_labels, decoded_preds, average='macro', zero_division=0) * 100
                
                # Calculate accuracy
                acc = accuracy_score(decoded_labels, decoded_preds) * 100
                
                # Get loss from evaluation
                eval_results = trainer.evaluate(eval_dataset=val_tokenized_single)
                loss = eval_results.get('eval_loss', 0.0)
                
                per_dataset_f1[dataset_name] = f1_macro
                
                print(f"  {dataset_name:15s} | Accuracy: {acc:6.2f}% | Macro F1: {f1_macro:6.2f}% | Loss: {loss:.4f}")
            
            print("=" * 80)
            
            # Store in trainer state for logging
            if not hasattr(state, 'per_dataset_f1_history'):
                state.per_dataset_f1_history = []
            state.per_dataset_f1_history.append({
                'epoch': int(state.epoch),
                **per_dataset_f1
            })
    
    # Compute metrics function (following VIHOS_T5.py approach - stable and simple)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Safety: handle tuple/3D array (following VIHOS_T5.py)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = np.array(predictions)
        
        # If 3D (logits), take argmax
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)
        
        # Clip to valid vocab range
        predictions = np.clip(predictions, 0, len(tokenizer) - 1).astype(np.int64)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, len(tokenizer) - 1).astype(np.int64)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Normalize labels (strip and uppercase for comparison)
        decoded_preds = [pred.strip().upper() for pred in decoded_preds]
        decoded_labels = [label.strip().upper() for label in decoded_labels]
        
        # Calculate F1 macro
        f1_macro = f1_score(decoded_labels, decoded_preds, average='macro', zero_division=0)
        
        # Exact match (following VIHOS_T5.py)
        exact_match = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l) / len(decoded_preds) if decoded_preds else 0.0
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        gen_len_mean = np.mean(prediction_lens)
        
        return {
            'f1_macro': round(f1_macro * 100, 4), 
            'exact_match': round(exact_match * 100, 2),
            'gen_len': round(gen_len_mean, 4)
        }
    
    # Create trainer with callbacks
    callbacks = []
    callback_instance = None
    if is_multi_dataset and individual_val_tokenized:
        # Add callback to evaluate on individual datasets after each epoch
        callback_instance = PerDatasetEvalCallback(individual_val_tokenized, tokenizer, norm)
        callbacks.append(callback_instance)
        print("\n✅ Added per-dataset evaluation callback - will evaluate on each dataset after every epoch")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Set trainer reference in callback after trainer is created
    if callback_instance is not None:
        callback_instance.trainer = trainer
    
    # Train
    print("\n🚀 Starting training...")
    import time
    train_start_time = time.time()
    
    # Track peak memory during training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    trainer.train()
    
    train_end_time = time.time()
    total_train_time = (train_end_time - train_start_time) / 60  # minutes
    
    # Get peak memory
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    final_vram = get_gpu_memory()
    final_ram = get_ram_usage()
    
    print(f"\n💾 Memory Usage:")
    print(f"   Peak VRAM: {peak_vram:.2f} GB")
    print(f"   Final VRAM: {final_vram:.2f} GB")
    print(f"   Final RAM: {final_ram:.2f} GB")
    print(f"   Training Time: {total_train_time:.2f} minutes")
    
    # Evaluate on test set with detailed metrics
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    print(f"  Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.2f}")
    print(f"  Test F1 Macro: {test_results.get('eval_f1_macro', 'N/A'):.2f}")
    print(f"  Test Loss    : {test_results.get('eval_loss', 'N/A'):.4f}")
    
    # Generate predictions for classification report (following VIHOS_T5.py approach)
    print("\n📋 Generating detailed predictions for classification report...")
    predictions = trainer.predict(test_tokenized)
    preds = predictions.predictions
    labels = predictions.label_ids
    
    # Safety: handle tuple/3D array (following VIHOS_T5.py)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.array(preds)
    if preds.ndim == 3:
        preds = preds.argmax(axis=-1)
    
    # Clip to valid vocab range
    preds = np.clip(preds, 0, len(tokenizer) - 1).astype(np.int64)
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = np.clip(labels, 0, len(tokenizer) - 1).astype(np.int64)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize labels (strip and uppercase for comparison)
    y_pred = [pred.strip().upper() for pred in decoded_preds]
    y_true = [label.strip().upper() for label in decoded_labels]
    
    # Get unique labels
    unique_labels = sorted(set(y_true + y_pred))
    
    # Print classification report
    print("\n" + "=" * 80)
    print("📊 CLASSIFICATION REPORT (Combined Test Set)")
    print("=" * 80)
    report = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0, digits=4)
    print(report)
    
    # Print confusion matrix
    print("\n" + "=" * 80)
    print("📊 CONFUSION MATRIX (Combined Test Set)")
    print("=" * 80)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)
    print("=" * 80)
    
    # Save classification report and confusion matrix
    report_dict = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_dict).transpose()
    # Round numeric columns to 4 decimal places
    numeric_cols = ['precision', 'recall', 'f1-score']
    for col in numeric_cols:
        if col in report_df.columns:
            report_df[col] = report_df[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    report_df.to_csv(f"{output_dir}/classification_report_combined.csv", float_format='%.4f')
    print(f"\n💾 Saved classification report to {output_dir}/classification_report_combined.csv")
    
    cm_df.to_csv(f"{output_dir}/confusion_matrix_combined.csv")
    print(f"💾 Saved confusion matrix to {output_dir}/confusion_matrix_combined.csv")
    
    # Save detailed predictions for error analysis
    predictions_df = pd.DataFrame({
        'text': test_df['source'].values[:len(y_pred)],
        'true_label': y_true,
        'pred_label': y_pred,
        'correct': [1 if p == t else 0 for p, t in zip(y_pred, y_true)]
    })
    predictions_df.to_csv(f"{output_dir}/predictions_detailed_combined.csv", index=False)
    print(f"💾 Saved detailed predictions to {output_dir}/predictions_detailed_combined.csv")
    
    # Evaluate on each individual dataset if multi-dataset training
    per_dataset_results = []
    if is_multi_dataset and individual_test_dfs:
        print("\n" + "=" * 80)
        print("📊 EVALUATING ON INDIVIDUAL DATASETS")
        print("=" * 80)
        
        for dataset_name in dataset_list:
            print(f"\n{'='*80}")
            print(f"📊 Evaluating on {dataset_name}...")
            print(f"{'='*80}")
            
            test_df_single = individual_test_dfs[dataset_name]
            metadata_single = individual_test_metadata[dataset_name]
            
            # Convert to HuggingFace Dataset and tokenize
            test_data_single = Dataset.from_pandas(test_df_single.reset_index(drop=True))
            test_tokenized_single = test_data_single.map(
                preprocess_function,
                batched=True,
                remove_columns=test_data_single.column_names
            )
            
            # Evaluate
            test_results_single = trainer.evaluate(test_tokenized_single)
            
            # Generate predictions (following VIHOS_T5.py approach)
            predictions_single = trainer.predict(test_tokenized_single)
            preds_single = predictions_single.predictions
            labels_single = predictions_single.label_ids
            
            # Safety: handle tuple/3D array (following VIHOS_T5.py)
            if isinstance(preds_single, tuple):
                preds_single = preds_single[0]
            
            preds_single = np.array(preds_single)
            if preds_single.ndim == 3:
                preds_single = preds_single.argmax(axis=-1)
            
            # Clip to valid vocab range
            preds_single = np.clip(preds_single, 0, len(tokenizer) - 1).astype(np.int64)
            
            # Decode predictions
            decoded_preds_single = tokenizer.batch_decode(preds_single, skip_special_tokens=True)
            
            # Decode labels
            labels_single = np.where(labels_single != -100, labels_single, tokenizer.pad_token_id)
            labels_single = np.clip(labels_single, 0, len(tokenizer) - 1).astype(np.int64)
            decoded_labels_single = tokenizer.batch_decode(labels_single, skip_special_tokens=True)
            
            # Normalize labels (strip and uppercase for comparison)
            y_pred_single = [pred.strip().upper() for pred in decoded_preds_single]
            y_true_single = [label.strip().upper() for label in decoded_labels_single]
            
            # Calculate metrics
            acc_single = accuracy_score(y_true_single, y_pred_single) * 100
            f1_macro_single = f1_score(y_true_single, y_pred_single, average="macro", zero_division=0) * 100
            f1_weighted_single = f1_score(y_true_single, y_pred_single, average="weighted", zero_division=0) * 100
            
            # Get unique labels
            unique_labels_single = sorted(set(y_true_single + y_pred_single))
            
            # Print results
            print(f"\n  Accuracy: {acc_single:.2f}%")
            print(f"  Macro F1: {f1_macro_single:.2f}%")
            print(f"  Weighted F1: {f1_weighted_single:.2f}%")
            print(f"  Loss: {test_results_single.get('eval_loss', 'N/A'):.4f}")
            
            # Classification report
            print(f"\n  Classification Report:")
            report_single = classification_report(y_true_single, y_pred_single, labels=unique_labels_single, zero_division=0, digits=4)
            print(report_single)
            
            # Save per-dataset results
            report_dict_single = classification_report(y_true_single, y_pred_single, labels=unique_labels_single, zero_division=0, output_dict=True, digits=4)
            report_df_single = pd.DataFrame(report_dict_single).transpose()
            for col in numeric_cols:
                if col in report_df_single.columns:
                    report_df_single[col] = report_df_single[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
            report_df_single.to_csv(f"{output_dir}/classification_report_{dataset_name}.csv", float_format='%.4f')
            
            cm_single = confusion_matrix(y_true_single, y_pred_single, labels=unique_labels_single)
            cm_df_single = pd.DataFrame(cm_single, index=unique_labels_single, columns=unique_labels_single)
            cm_df_single.to_csv(f"{output_dir}/confusion_matrix_{dataset_name}.csv")
            
            predictions_df_single = pd.DataFrame({
                'text': test_df_single['source'].values[:len(y_pred_single)],
                'true_label': y_true_single,
                'pred_label': y_pred_single,
                'correct': [1 if p == t else 0 for p, t in zip(y_pred_single, y_true_single)]
            })
            predictions_df_single.to_csv(f"{output_dir}/predictions_detailed_{dataset_name}.csv", index=False)
            
            print(f"  💾 Saved results to {output_dir}/classification_report_{dataset_name}.csv")
            
            # Store results for summary
            per_dataset_results.append({
                'Dataset': dataset_name,
                'Accuracy': round(acc_single, 4),
                'Macro_F1': round(f1_macro_single, 4),
                'Weighted_F1': round(f1_weighted_single, 4),
                'Loss': round(test_results_single.get('eval_loss', 0), 4),
                'Test_Samples': len(test_df_single)
            })
        
        # Save summary of all datasets
        if per_dataset_results:
            summary_df = pd.DataFrame(per_dataset_results)
            summary_df.to_csv(f"{output_dir}/per_dataset_results_summary.csv", index=False)
            print(f"\n{'='*80}")
            print("📊 PER-DATASET RESULTS SUMMARY")
            print(f"{'='*80}")
            print(summary_df.to_string(index=False))
            print(f"{'='*80}")
            print(f"💾 Saved summary to {output_dir}/per_dataset_results_summary.csv")
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    print("  Saving training configuration...")
    config_dict = {
        "dataset": args.dataset,
        "is_multi_dataset": is_multi_dataset,
        "num_datasets": len(dataset_list) if is_multi_dataset else 1,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        "dev_ratio": args.dev_ratio,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "num_labels": metadata.get("num_labels", "unknown"),
        "text_col": metadata.get("text_col", "unknown"),
        "label_col": metadata.get("label_col", "unknown"),
    }
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(f"{output_dir}/training_config.csv", index=False)
    print(f"    Saved to {output_dir}/training_config.csv")
    
    # Save training history (epoch metrics)
    print("  Saving training history...")
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        history_rows = []
        for log_entry in trainer.state.log_history:
            if 'epoch' in log_entry:
                row = {
                    "epoch": log_entry.get("epoch", None),
                    "step": log_entry.get("step", None),
                }
                # Add train metrics
                if "train_loss" in log_entry:
                    row["train_loss"] = log_entry["train_loss"]
                if "train_runtime" in log_entry:
                    row["train_runtime"] = log_entry["train_runtime"]
                if "train_samples_per_second" in log_entry:
                    row["train_samples_per_second"] = log_entry["train_samples_per_second"]
                
                # Add eval metrics
                if "eval_loss" in log_entry:
                    row["eval_loss"] = log_entry["eval_loss"]
                if "eval_accuracy" in log_entry:
                    row["eval_accuracy"] = log_entry["eval_accuracy"]
                if "eval_f1_macro" in log_entry:
                    row["eval_f1_macro"] = log_entry["eval_f1_macro"]
                if "eval_gen_len" in log_entry:
                    row["eval_gen_len"] = log_entry["eval_gen_len"]
                
                # Add learning rate
                if "learning_rate" in log_entry:
                    row["learning_rate"] = log_entry["learning_rate"]
                
                history_rows.append(row)
        
        if history_rows:
            history_df = pd.DataFrame(history_rows)
            history_df.to_csv(f"{output_dir}/training_history.csv", index=False)
            print(f"    Saved to {output_dir}/training_history.csv ({len(history_rows)} entries)")
        else:
            print("    ⚠️  No training history found")
    else:
        print("    ⚠️  No training history available")
    
    # Save test results
    print("  Saving test results...")
    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv(f"{output_dir}/test_results.csv", index=False)
    print(f"    Saved to {output_dir}/test_results.csv")
    
    # Save run summary
    print("  Saving run summary...")
    
    # Calculate training time if available
    training_time = total_train_time  # Use the measured time
    
    # Get best eval metrics
    best_eval_f1 = None
    best_eval_acc = None
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        eval_entries = [log for log in trainer.state.log_history if "eval_f1_macro" in log]
        if eval_entries:
            best_eval_f1 = max([log["eval_f1_macro"] for log in eval_entries])
        eval_acc_entries = [log for log in trainer.state.log_history if "eval_accuracy" in log]
        if eval_acc_entries:
            best_eval_acc = max([log["eval_accuracy"] for log in eval_acc_entries])
    
    # Get per-class metrics from classification report
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    for label in unique_labels:
        if label in report_dict and label not in ['accuracy', 'macro avg', 'weighted avg']:
            per_class_f1[f"f1_{label}"] = report_dict[label].get('f1-score', 0.0) * 100
            per_class_precision[f"precision_{label}"] = report_dict[label].get('precision', 0.0) * 100
            per_class_recall[f"recall_{label}"] = report_dict[label].get('recall', 0.0) * 100
    
    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "best_eval_accuracy": best_eval_acc,
        "best_eval_f1_macro": best_eval_f1,
        "test_accuracy": test_results.get('eval_accuracy', None),
        "test_f1_macro": test_results.get('eval_f1_macro', None),
        "test_loss": test_results.get('eval_loss', None),
        "training_minutes": training_time,
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "max_length": args.max_length,
        "peak_vram_gb": peak_vram,
        "final_vram_gb": final_vram,
        "final_ram_gb": final_ram,
        "total_vram_gb": total_vram if torch.cuda.is_available() else 0.0,
    }
    
    # Add per-class metrics
    summary.update(per_class_f1)
    summary.update(per_class_precision)
    summary.update(per_class_recall)
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/run_summary.csv", index=False)
    print(f"    Saved to {output_dir}/run_summary.csv")
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Files saved:")
    print(f"     - training_config.csv (hyperparameters & dataset info)")
    print(f"     - training_history.csv (metrics per epoch)")
    print(f"     - test_results.csv (final test metrics)")
    print(f"     - run_summary.csv (overall summary + memory usage)")
    print(f"     - classification_report.csv (per-class precision/recall/f1)")
    print(f"     - confusion_matrix.csv (confusion matrix)")
    print(f"     - predictions_detailed.csv (all predictions for error analysis)")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Training Time: {training_time:.2f} minutes")
    print(f"   Peak VRAM: {peak_vram:.2f} GB / {total_vram if torch.cuda.is_available() else 0.0:.1f} GB")
    print(f"   Test Accuracy: {test_results.get('eval_accuracy', 0):.2f}%")
    print(f"   Test F1 Macro: {test_results.get('eval_f1_macro', 0):.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()