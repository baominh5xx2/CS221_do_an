"""
Evaluation script for trained models.

Supports both classification models (PhoBERT, ViSoBERT, etc.) and T5 models.

Usage:
    # Evaluate with local model:
    python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD
    
    # Evaluate with Hugging Face classification model:
    python src/evaluate.py --model_name username/model-name --dataset ViHSD
    
    # Evaluate with Hugging Face T5 model (e.g., ViHateT5):
    python src/evaluate.py --model_name tarudesu/ViHateT5-base --dataset ViHSD
"""

import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

from data_loader import load_dataset_by_name, build_torch_dataset
from model import load_trained_model
from utils import evaluate


def is_t5_model(model_path_or_name: str) -> bool:
    """Check if model is a T5 model based on name."""
    model_lower = model_path_or_name.lower()
    return "t5" in model_lower or "vit5" in model_lower or "vihatet5" in model_lower


def map_dataset_for_t5(df, dataset_name, metadata):
    """Map dataset to source-target format for T5 models."""
    # Import mapping functions from train_t5
    try:
        # Try importing as module first
        from train_t5 import map_dataset
        return map_dataset(df, dataset_name, metadata)
    except ImportError:
        # Fallback: import from file path
        import sys
        import importlib.util
        from pathlib import Path
        
        # Try to find train_t5.py in the same directory as this file
        current_file = Path(__file__).resolve()
        train_t5_path = current_file.parent / "train_t5.py"
        
        if not train_t5_path.exists():
            # Try src/train_t5.py from workspace root
            train_t5_path = current_file.parent.parent / "src" / "train_t5.py"
        
        if train_t5_path.exists():
            spec = importlib.util.spec_from_file_location("train_t5", str(train_t5_path))
            train_t5_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_t5_module)
            return train_t5_module.map_dataset(df, dataset_name, metadata)
        else:
            raise FileNotFoundError(f"Could not find train_t5.py to import mapping functions")


def normalize_label(s):
    """Normalize label: strip, uppercase, take first word, validate."""
    VALID = {"CLEAN", "OFFENSIVE", "HATE", "NONE", "TOXIC"}
    s = (s or "").strip().upper()
    s = s.split()[0] if s else ""
    return s if s in VALID else "OTHER"


def evaluate_t5(model, tokenizer, eval_df, dataset_name, metadata, device, batch_size=16, max_length=256):
    """Evaluate T5 model using generation."""
    model.eval()
    
    # Map dataset to source-target format
    mapped_df = map_dataset_for_t5(eval_df.copy(), dataset_name, metadata)
    
    all_preds = []
    all_labels = []
    
    # Generate predictions
    with torch.no_grad():
        for i in tqdm(range(0, len(mapped_df), batch_size), desc="Generating predictions"):
            batch_sources = mapped_df["source"].iloc[i:i+batch_size].tolist()
            batch_targets = mapped_df["target"].iloc[i:i+batch_size].tolist()
            
            # Tokenize inputs
            inputs = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                do_sample=False
            )
            
            # Decode predictions
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(decoded_preds)
            all_labels.extend(batch_targets)
    
    # Normalize labels
    y_pred = [normalize_label(p) for p in all_preds]
    y_true = [normalize_label(t) for t in all_labels]
    
    return np.array(y_pred), np.array(y_true), 0.0  # Loss not available for generation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_path", type=str,
                            help="Path to trained model directory (local)")
    model_group.add_argument("--model_name", type=str,
                            help="Hugging Face model name/identifier (e.g., 'username/model-name')")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset to evaluate on (ViHSD, ViCTSD, ViHOS, Minhbao5xx2/VOZ-HSD_2M)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print(f"Evaluation Configuration:")
    print("=" * 80)
    if args.model_path:
        print(f"  Model path (local): {args.model_path}")
        model_source = args.model_path
    else:
        print(f"  Model name (Hugging Face): {args.model_name}")
        model_source = args.model_name
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Load dataset first to get metadata (needed for Hugging Face models)
    print(f"\n📚 Loading {args.dataset} dataset...")
    train_df, val_df, test_df, metadata = load_dataset_by_name(args.dataset)
    
    # Determine model type and load accordingly
    model_source = args.model_path if args.model_path else args.model_name
    is_t5 = is_t5_model(model_source)
    
    # Load model
    if args.model_name:
        print(f"\n🤖 Loading model from Hugging Face: {args.model_name}...")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        if is_t5:
            print(f"  Detected T5 model, loading T5ForConditionalGeneration...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            # Check if ViHateT5 (Flax-based)
            if "ViHateT5" in args.model_name:
                print(f"  Loading from Flax weights (ViHateT5)...")
                # Load on CPU first to avoid meta tensor issues
                model = T5ForConditionalGeneration.from_pretrained(
                    args.model_name, 
                    from_flax=True,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float32
                )
                # Ensure model is on CPU (not meta) before moving to target device
                model = model.to("cpu")
            else:
                model = T5ForConditionalGeneration.from_pretrained(args.model_name)
            
            # Move to target device after ensuring model is fully loaded
            if device_str != "cpu":
                model = model.to(device_str)
            model.eval()
        else:
            from model import build_model
            model, tokenizer = build_model(
                args.model_name, 
                num_labels=metadata["num_labels"],
                device=device_str
            )
            model.eval()
    else:
        print(f"\n🤖 Loading model from local path: {args.model_path}...")
        
        if is_t5:
            print(f"  Detected T5 model, loading T5ForConditionalGeneration...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            # Try loading normally first, if fails try from_flax
            try:
                model = T5ForConditionalGeneration.from_pretrained(
                    args.model_path,
                    low_cpu_mem_usage=False
                )
            except:
                print(f"  Trying to load from Flax weights...")
                model = T5ForConditionalGeneration.from_pretrained(
                    args.model_path, 
                    from_flax=True,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float32
                )
                # Ensure model is on CPU (not meta) before moving to target device
                model = model.to("cpu")
            
            # Move to target device after ensuring model is fully loaded
            if device_str != "cpu":
                model = model.to(device_str)
            model.eval()
        else:
            model, tokenizer = load_trained_model(args.model_path)
    
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    print(f"  Device: {device}")
    
    # Select split
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    
    print(f"  Evaluating on {args.split} split: {len(eval_df)} samples")
    
    # Evaluate based on model type
    if is_t5:
        print(f"\n🔍 Evaluating T5 model (generation-based)...")
        preds, labels, loss = evaluate_t5(
            model, tokenizer, eval_df, args.dataset, metadata, 
            device, args.batch_size, args.max_length
        )
    else:
        # Build dataset for classification models
        eval_dataset = build_torch_dataset(
            eval_df, metadata["text_col"], metadata["label_col"],
            tokenizer, args.max_length
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
        
        print(f"\n🔍 Evaluating classification model...")
        preds, labels, loss = evaluate(model, eval_loader, device)
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    
    # Get unique labels for classification report
    unique_labels = sorted(set(list(labels) + list(preds)))
    
    print("\n" + "=" * 80)
    print(f"Results on {args.dataset} ({args.split} split):")
    print("=" * 80)
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print("=" * 80)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, labels=unique_labels, digits=4, zero_division=0))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_source": model_source,
        "dataset": args.dataset,
        "split": args.split,
        "loss": loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    
    results_df = pd.DataFrame([results])
    results_file = output_dir / f"eval_{args.dataset}_{args.split}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n💾 Saved results to {results_file}")
    
    # Save detailed classification report
    report_dict = classification_report(labels, preds, labels=unique_labels, digits=4, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_file = output_dir / f"report_{args.dataset}_{args.split}.csv"
    report_df.to_csv(report_file)
    
    print(f"💾 Saved classification report to {report_file}")
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
