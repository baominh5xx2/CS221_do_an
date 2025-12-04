"""
Evaluation script for trained models.

Usage:
    python src/evaluate.py --model_path models/ViHSD_phobert --dataset ViHSD
"""

import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data_loader import load_dataset_by_name, build_torch_dataset
from model import load_trained_model
from utils import evaluate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["ViHSD", "ViCTSD", "ViHOS"],
                       help="Dataset to evaluate on")
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
    print(f"  Model path: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Load model
    print(f"\n🤖 Loading model from {args.model_path}...")
    model, tokenizer = load_trained_model(args.model_path)
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    print(f"  Device: {device}")
    
    # Load dataset
    print(f"\n📚 Loading {args.dataset} dataset...")
    train_df, val_df, test_df, metadata = load_dataset_by_name(args.dataset)
    
    # Select split
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    
    print(f"  Evaluating on {args.split} split: {len(eval_df)} samples")
    
    # Build dataset
    eval_dataset = build_torch_dataset(
        eval_df, metadata["text_col"], metadata["label_col"],
        tokenizer, args.max_length
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # Evaluate
    print(f"\n🔍 Evaluating...")
    preds, labels, loss = evaluate(model, eval_loader, device)
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    
    print("\n" + "=" * 80)
    print(f"Results on {args.dataset} ({args.split} split):")
    print("=" * 80)
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print("=" * 80)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_path": args.model_path,
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
    report_dict = classification_report(labels, preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_file = output_dir / f"report_{args.dataset}_{args.split}.csv"
    report_df.to_csv(report_file)
    
    print(f"💾 Saved classification report to {report_file}")
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
