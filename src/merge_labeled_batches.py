"""
Merge multiple labeled CSV files from parallel batch processing.

Usage:
    python src/merge_labeled_batches.py --input_dir labeled_data/voz_hsd --split train --total_batches 10
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge labeled batch files")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing batch files")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split name")
    parser.add_argument("--total_batches", type=int, required=True,
                       help="Total number of batches")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    
    print(f"Merging {args.total_batches} batch files from {input_dir}...")
    
    # Collect all batch files
    all_dfs = []
    for i in range(args.total_batches):
        batch_file = input_dir / f"{args.split}_batch_{i}_{args.total_batches}.csv"
        
        if not batch_file.exists():
            print(f"⚠️  Warning: Batch file not found: {batch_file}")
            continue
        
        df = pd.read_csv(batch_file)
        all_dfs.append(df)
        print(f"  Loaded batch {i}: {len(df):,} samples")
    
    # Merge all dataframes
    if len(all_dfs) == 0:
        print("❌ Error: No batch files found!")
        return
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✅ Merged {len(all_dfs)} batches: {len(merged_df):,} total samples")
    
    # Calculate overall metrics
    if "original_label" in merged_df.columns and "predicted_label" in merged_df.columns:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(merged_df["original_label"], merged_df["predicted_label"])
        precision, recall, f1, _ = precision_recall_fscore_support(
            merged_df["original_label"], merged_df["predicted_label"],
            average="binary", pos_label=1
        )
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
    
    # Save merged file
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_dir / f"{args.split}_labeled_combined.csv"
    
    merged_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
