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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_loader import load_dataset_by_name, build_torch_dataset
from model import load_trained_model
from utils import evaluate


def is_t5_model(model_path_or_name: str) -> bool:
    """Check if model is a T5 model based on name."""
    model_lower = (model_path_or_name or "").lower()
    return ("t5" in model_lower) or ("vit5" in model_lower) or ("vihatet5" in model_lower)


def map_dataset_for_t5(df, dataset_name, metadata):
    """Map dataset to source-target format for T5 models."""
    try:
        from train_t5 import map_dataset
        return map_dataset(df, dataset_name, metadata)
    except ImportError:
        import importlib.util

        current_file = Path(__file__).resolve()
        train_t5_path = current_file.parent / "train_t5.py"

        if not train_t5_path.exists():
            train_t5_path = current_file.parent.parent / "src" / "train_t5.py"

        if not train_t5_path.exists():
            raise FileNotFoundError("Could not find train_t5.py to import mapping functions")

        spec = importlib.util.spec_from_file_location("train_t5", str(train_t5_path))
        train_t5_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_t5_module)
        return train_t5_module.map_dataset(df, dataset_name, metadata)


def normalize_label(s: str) -> str:
    """Normalize label: strip, uppercase, take first word, validate."""
    VALID = {"CLEAN", "OFFENSIVE", "HATE", "NONE", "TOXIC"}
    s = (s or "").strip().upper()
    s = s.split()[0] if s else ""
    return s if s in VALID else "OTHER"


def _has_meta_tensors(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if getattr(p, "is_meta", False) or p.device.type == "meta":
            return True
    for b in model.buffers():
        if getattr(b, "is_meta", False) or b.device.type == "meta":
            return True
    return False


def load_t5_model(model_id_or_path: str, device: torch.device):
    """
    Robust loader to avoid meta-tensor issues:
    - Force low_cpu_mem_usage=False (avoid meta init in many cases)
    - Avoid manual random init of missing weights
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    # 1) Prefer PyTorch weights if available
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
    except Exception:
        # 2) Fallback to Flax conversion
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id_or_path,
            from_flax=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )

    # Safety check: if still meta, fail fast with actionable message
    if _has_meta_tensors(model):
        raise RuntimeError(
            "Model contains meta tensors after loading. "
            "Try upgrading transformers/torch, and ensure low_cpu_mem_usage=False "
            "and that the repo provides compatible weights (PyTorch or convertible Flax)."
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_t5(
    model,
    tokenizer,
    eval_df,
    dataset_name,
    metadata,
    device: torch.device,
    batch_size=16,
    max_length=256,
):
    """Evaluate T5 model using generation (batched)."""
    model.eval()

    mapped_df = map_dataset_for_t5(eval_df.copy(), dataset_name, metadata)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(mapped_df), batch_size), desc="Generating predictions"):
            batch_sources = mapped_df["source"].iloc[i : i + batch_size].tolist()
            batch_targets = mapped_df["target"].iloc[i : i + batch_size].tolist()

            enc = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Labels are short; keep generation bounded
            out_ids = model.generate(
                **enc,
                max_new_tokens=8,
                num_beams=1,
            )

            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            all_preds.extend(decoded)
            all_labels.extend(batch_targets)

    y_pred = [normalize_label(p) for p in all_preds]
    y_true = [normalize_label(t) for t in all_labels]

    return np.array(y_pred), np.array(y_true), 0.0  # No loss for pure generation eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_path", type=str, help="Path to trained model directory (local)")
    model_group.add_argument(
        "--model_name",
        type=str,
        help="Hugging Face model name/identifier (e.g., 'username/model-name')",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to evaluate on (ViHSD, ViCTSD, ViHOS, Minhbao5xx2/VOZ-HSD_2M)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Evaluation Configuration:")
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

    print(f"\nLoading {args.dataset} dataset...")
    train_df, val_df, test_df, metadata = load_dataset_by_name(args.dataset)

    model_source = args.model_path if args.model_path else args.model_name
    is_t5 = is_t5_model(model_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if args.model_name:
        print(f"\nLoading model from Hugging Face: {args.model_name} ...")
        if is_t5:
            print("  Detected T5 model, loading AutoModelForSeq2SeqLM...")
            model, tokenizer = load_t5_model(args.model_name, device)
        else:
            from model import build_model

            model, tokenizer = build_model(
                args.model_name,
                num_labels=metadata["num_labels"],
                device=str(device),
            )
            model.eval()
    else:
        print(f"\nLoading model from local path: {args.model_path} ...")
        if is_t5:
            print("  Detected T5 model, loading AutoModelForSeq2SeqLM...")
            model, tokenizer = load_t5_model(args.model_path, device)
        else:
            model, tokenizer = load_trained_model(args.model_path, device=str(device))
            model.eval()

    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    print(f"  Evaluating on {args.split} split: {len(eval_df)} samples")

    if is_t5:
        print("\nEvaluating T5 model (generation-based)...")
        preds, labels, loss = evaluate_t5(
            model=model,
            tokenizer=tokenizer,
            eval_df=eval_df,
            dataset_name=args.dataset,
            metadata=metadata,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        eval_dataset = build_torch_dataset(
            eval_df,
            metadata["text_col"],
            metadata["label_col"],
            tokenizer,
            args.max_length,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

        print("\nEvaluating classification model...")
        preds, labels, loss = evaluate(model, eval_loader, str(device))

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

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
    print(f"\nSaved results to {results_file}")

    report_dict = classification_report(
        labels,
        preds,
        labels=unique_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_file = output_dir / f"report_{args.dataset}_{args.split}.csv"
    report_df.to_csv(report_file)
    print(f"Saved classification report to {report_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
