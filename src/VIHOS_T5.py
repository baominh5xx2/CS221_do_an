"""
Train ViHOS (span detection) - with eval metrics during training.
"""

import os
import ast
import argparse
import unicodedata
import pandas as pd
import numpy as np
import torch
import psutil
from datetime import datetime
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "False"
from data_loader import load_dataset_by_name  # noqa: E402


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

def parse_args():
    p = argparse.ArgumentParser(description="Train T5 for ViHOS span detection")
    p.add_argument("--model_name", default="VietAI/vit5-base")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--eval_batch_size", type=int, default=128)
    
    # New args
    p.add_argument("--optim", type=str, default="adafactor")
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing_factor", type=float, default=0.0)
    p.add_argument("--num_beams", type=int, default=1)
    
    return p.parse_args()

def process_spans(lst):
    if lst == '[]' or pd.isna(lst) or lst == '': return []
    lst = [int(x) for x in str(lst).strip("[]'").split(', ') if x.strip()]
    if not lst: return []
    result, temp = [], [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1: temp.append(lst[i])
        else: result.append(temp); temp = [lst[i]]
    result.append(temp)
    return result

def add_tags(text, indices):
    if indices == "[]" or pd.isna(indices) or indices == "":
        return text
    indices = process_spans(indices)
    if not indices:
        return text
    for i in range(len(indices)):
        text = text[: indices[i][0]] + "[HATE]" + text[indices[i][0] :]
        text = text[: indices[i][-1] + 7] + "[HATE]" + text[indices[i][-1] + 7 :]
        for j in range(i + 1, len(indices)):
            indices[j] = [x + 12 for x in indices[j]]
    return text

def map_vihos(df):
    df = df.copy()
    df["source"] = df["content"].apply(lambda x: "hate-spans-detection: " + str(x))
    df["target"] = [add_tags(str(df['content'].iloc[i]), df['index_spans'].iloc[i]) for i in range(len(df))]
    return df[["source", "target"]]


# ----- Evaluation helpers -----
def extract_spans(original, output):
    """Extract character indices from generated text with [HATE] ... [HATE] tags."""
    tag = "[hate]"
    output = unicodedata.normalize("NFC", output.lower())
    original = unicodedata.normalize("NFC", original.lower())

    subs = []
    start = output.find(tag)
    while start != -1:
        end = output.find(tag, start + len(tag))
        if end != -1:
            subs.append(output[start + len(tag) : end])
            start = output.find(tag, end + len(tag))
        else:
            break

    if not subs:
        return "[]"

    indices = []
    for sub in subs:
        pos = original.find(sub)
        while pos != -1:
            indices.extend(range(pos, pos + len(sub)))
            pos = original.find(sub, pos + 1)

    return str(sorted(set(indices)))


def digitize(df, preds_spans):
    """Convert predicted span strings to binary vectors aligned with content length."""
    preds = [ast.literal_eval(x) for x in preds_spans]
    labels = [ast.literal_eval(x) for x in df["index_spans"]]
    lens = [len(x) for x in df["content"]]

    bin_preds, bin_labels = [], []
    for i in range(len(lens)):
        bin_preds.append([1 if idx in preds[i] else 0 for idx in range(lens[i])])
        bin_labels.append([1 if idx in labels[i] else 0 for idx in range(lens[i])])

    return bin_labels, bin_preds


def evaluate_vihos(model, tokenizer, df, device, batch_size=16, max_length=256):
    """Evaluate spans on a dataframe using generation + span extraction."""
    model.eval()
    df = df.copy()
    df["source"] = df["content"].apply(lambda x: "hate-spans-detection: " + str(x))

    outputs = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch = df["source"].iloc[i : i + batch_size].tolist()
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_length=max_length)
        outputs.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    pred_spans = [extract_spans(df["content"].iloc[i], outputs[i]) for i in range(len(df))]
    labels_bin, preds_bin = digitize(df, pred_spans)

    accs, f1w, f1m = [], [], []
    for i in range(len(labels_bin)):
        accs.append(accuracy_score(labels_bin[i], preds_bin[i]))
        f1w.append(f1_score(labels_bin[i], preds_bin[i], average="weighted", zero_division=0))
        f1m.append(f1_score(labels_bin[i], preds_bin[i], average="macro", zero_division=0))

    return np.mean(accs) * 100, np.mean(f1w) * 100, np.mean(f1m) * 100

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total VRAM: {total_vram:.1f} GB")
    
    # Track initial memory
    initial_vram = get_gpu_memory()
    initial_ram = get_ram_usage()
    print(f"   Initial VRAM: {initial_vram:.2f} GB")
    print(f"   Initial RAM: {initial_ram:.2f} GB")

    # Load ViHOS
    train_df, val_df, test_df, _ = load_dataset_by_name("ViHOS", dev_ratio=args.dev_ratio)
    print(f"📚 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print(f"⚙️  Config:")
    print(f"   Model: {args.model_name}")
    print(f"   Batch: {args.batch_size}")
    print(f"   LR: {args.lr}")
    print(f"   Optim: {args.optim}")
    print(f"   Sched: {args.lr_scheduler_type}")

    # Keep original copies for evaluation
    val_df_orig = val_df.copy()
    test_df_orig = test_df.copy()

    # Map and clean
    train_df, val_df, test_df = map_vihos(train_df), map_vihos(val_df), map_vihos(test_df)
    train_df, val_df, test_df = train_df.dropna(), val_df.dropna(), test_df.dropna()

    # Print samples
    print("\n📋 Sample training examples:")
    for i in range(min(3, len(train_df))):
        print(f"  [{i+1}] Source: {train_df.iloc[i]['source'][:80]}...")
        print(f"      Target: {train_df.iloc[i]['target'][:80]}...")

    # HF datasets
    train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_data = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    def prep(sample):
        inp = tokenizer(sample["source"], max_length=args.max_length, padding="max_length", truncation=True)
        lbl = tokenizer(sample["target"], max_length=args.max_length, padding="max_length", truncation=True)
        lbl["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in lbl["input_ids"]]
        inp["labels"] = lbl["input_ids"]
        return inp

    train_tok = train_data.map(prep, batched=True, remove_columns=train_data.column_names)
    val_tok = val_data.map(prep, batched=True, remove_columns=val_data.column_names)

    out = args.output_dir or f"models/ViHOS_{args.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out, exist_ok=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=out,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        label_smoothing_factor=args.label_smoothing_factor,
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Vẫn dùng loss để pick best
        greater_is_better=False,
        predict_with_generate=True,  # Enable generation trong eval
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        report_to="none",
    )

    model.config.use_cache = False
    
    # Compute metrics (PROXY - không phải metric chuẩn)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Safety: handle tuple/3D array
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
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Exact match
        exact_match = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()) / len(decoded_preds) if decoded_preds else 0.0
        
        # Token-level F1 (proxy)
        token_matches = []
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_tokens = set(pred.split())
            label_tokens = set(label.split())
            if len(label_tokens) > 0:
                overlap = len(pred_tokens & label_tokens) / len(label_tokens)
                token_matches.append(overlap)
        
        token_f1 = np.mean(token_matches) if token_matches else 0.0
        
        # Debug first eval
        if not hasattr(compute_metrics, '_printed'):
            print(f"\n🔍 Sample predictions (first 3):")
            for i in range(min(3, len(decoded_preds))):
                print(f"  Pred: {decoded_preds[i][:60]}...")
                print(f"  True: {decoded_labels[i][:60]}...")
            compute_metrics._printed = True
        
        return {
            'exact_match': round(exact_match * 100, 2),
            'token_f1': round(token_f1 * 100, 2),
        }
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Training (with proxy metrics)...")
    import time
    train_start_time = time.time()
    
    # Reset peak memory stats
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
    
    trainer.save_model(out)
    tokenizer.save_pretrained(out)

    # --- Save Training Config & History ---
    print("  Saving training configuration...")
    config_dict = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        "dev_ratio": args.dev_ratio,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }
    pd.DataFrame([config_dict]).to_csv(f"{out}/training_config.csv", index=False)
    print(f"    Saved to {out}/training_config.csv")

    print("  Saving training history...")
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        history_rows = []
        for log_entry in trainer.state.log_history:
            if 'epoch' in log_entry:
                # Filter out some keys to keep it clean, but keep metrics
                row = {k: v for k, v in log_entry.items() if k not in ['total_flos', 'train_steps_per_second']}
                history_rows.append(row)
        
        if history_rows:
            pd.DataFrame(history_rows).to_csv(f"{out}/training_history.csv", index=False)
            print(f"    Saved to {out}/training_history.csv ({len(history_rows)} entries)")

    # ---- Auto evaluation on DEV & TEST (REAL metrics) ----
    print("\n" + "=" * 80)
    print("📊 Final Evaluation (REAL Process 1 & 2 metrics)")
    print("=" * 80)

    model.to(device)

    print("\n🔮 Evaluating on DEV set...")
    dev_acc, dev_f1w, dev_f1m = evaluate_vihos(
        model, tokenizer, val_df_orig, device, batch_size=args.eval_batch_size, max_length=args.max_length
    )

    print("\n🔮 Evaluating on TEST set...")
    test_acc, test_f1w, test_f1m = evaluate_vihos(
        model, tokenizer, test_df_orig, device, batch_size=args.eval_batch_size, max_length=args.max_length
    )

    print("\n" + "=" * 80)
    print("📈 DEV Results (REAL):")
    print("=" * 80)
    print(f"  Accuracy    : {dev_acc:.2f}")
    print(f"  Weighted F1 : {dev_f1w:.2f}")
    print(f"  Macro F1    : {dev_f1m:.2f}")

    print("\n" + "=" * 80)
    print("📈 TEST Results (REAL):")
    print("=" * 80)
    print(f"  Accuracy    : {test_acc:.2f}")
    print(f"  Weighted F1 : {test_f1w:.2f}")
    print(f"  Macro F1    : {test_f1m:.2f}")
    print("=" * 80)

    # Save evaluation results
    results = pd.DataFrame(
        [
            {"split": "dev", "accuracy": dev_acc, "f1_weighted": dev_f1w, "f1_macro": dev_f1m},
            {"split": "test", "accuracy": test_acc, "f1_weighted": test_f1w, "f1_macro": test_f1m},
        ]
    )
    results.to_csv(f"{out}/evaluation_results.csv", index=False)
    print(f"\n💾 Saved evaluation results to {out}/evaluation_results.csv")
    
    # Generate detailed predictions for error analysis (on test set)
    print("\n📋 Generating detailed predictions for error analysis...")
    test_df_full = test_df_orig.copy()
    test_df_full["source"] = test_df_full["content"].apply(lambda x: "hate-spans-detection: " + str(x))
    
    test_outputs = []
    for i in tqdm(range(0, len(test_df_full), args.eval_batch_size), desc="Generating test predictions"):
        batch = test_df_full["source"].iloc[i : i + args.eval_batch_size].tolist()
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out_ids = model.generate(**inp, max_length=args.max_length)
        test_outputs.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
    
    # Extract predicted spans
    pred_spans_test = [extract_spans(test_df_full["content"].iloc[i], test_outputs[i]) for i in range(len(test_df_full))]
    
    # Create binary labels for classification
    test_has_hate_true = [1 if (test_df_full["index_spans"].iloc[i] != '[]' and 
                                  test_df_full["index_spans"].iloc[i] != '' and 
                                  not pd.isna(test_df_full["index_spans"].iloc[i])) else 0 
                          for i in range(len(test_df_full))]
    test_has_hate_pred = [1 if (pred != '[]' and pred != '') else 0 for pred in pred_spans_test]
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'content': test_df_full['content'].values,
        'true_spans': test_df_full['index_spans'].values,
        'pred_spans': pred_spans_test,
        'generated_text': test_outputs,
        'has_hate_true': test_has_hate_true,
        'has_hate_pred': test_has_hate_pred,
        'correct': [1 if p == t else 0 for p, t in zip(test_has_hate_pred, test_has_hate_true)]
    })
    predictions_df.to_csv(f"{out}/predictions_detailed.csv", index=False)
    print(f"💾 Saved detailed predictions to {out}/predictions_detailed.csv")
    
    # Classification report for binary hate/no-hate
    print("\n" + "=" * 80)
    print("📊 CLASSIFICATION REPORT (Binary: Has Hate Spans)")
    print("=" * 80)
    report = classification_report(test_has_hate_true, test_has_hate_pred, 
                                   target_names=['No Hate', 'Has Hate'], zero_division=0, digits=4)
    print(report)
    
    # Confusion matrix
    print("\n" + "=" * 80)
    print("📊 CONFUSION MATRIX (Binary: Has Hate Spans)")
    print("=" * 80)
    cm = confusion_matrix(test_has_hate_true, test_has_hate_pred)
    cm_df = pd.DataFrame(cm, index=['No Hate', 'Has Hate'], columns=['No Hate', 'Has Hate'])
    print(cm_df)
    print("=" * 80)
    
    # Save classification report and confusion matrix
    report_dict = classification_report(test_has_hate_true, test_has_hate_pred, 
                                       target_names=['No Hate', 'Has Hate'], 
                                       zero_division=0, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_dict).transpose()
    # Round numeric columns to 4 decimal places
    numeric_cols = ['precision', 'recall', 'f1-score']
    for col in numeric_cols:
        if col in report_df.columns:
            report_df[col] = report_df[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    report_df.to_csv(f"{out}/classification_report.csv", float_format='%.4f')
    print(f"\n💾 Saved classification report to {out}/classification_report.csv")
    
    cm_df.to_csv(f"{out}/confusion_matrix.csv")
    print(f"💾 Saved confusion matrix to {out}/confusion_matrix.csv")

    # Save run summary
    print("  Saving run summary...")
    
    # Get per-class metrics
    no_hate_f1 = report_dict['No Hate']['f1-score'] * 100 if 'No Hate' in report_dict else 0.0
    has_hate_f1 = report_dict['Has Hate']['f1-score'] * 100 if 'Has Hate' in report_dict else 0.0
    no_hate_precision = report_dict['No Hate']['precision'] * 100 if 'No Hate' in report_dict else 0.0
    has_hate_precision = report_dict['Has Hate']['precision'] * 100 if 'Has Hate' in report_dict else 0.0
    no_hate_recall = report_dict['No Hate']['recall'] * 100 if 'No Hate' in report_dict else 0.0
    has_hate_recall = report_dict['Has Hate']['recall'] * 100 if 'Has Hate' in report_dict else 0.0
    
    summary = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "dev_accuracy": dev_acc,
        "dev_f1_weighted": dev_f1w,
        "dev_f1_macro": dev_f1m,
        "test_accuracy": test_acc,
        "test_f1_weighted": test_f1w,
        "test_f1_macro": test_f1m,
        "training_minutes": total_train_time,
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "peak_vram_gb": peak_vram,
        "final_vram_gb": final_vram,
        "final_ram_gb": final_ram,
        "total_vram_gb": total_vram if torch.cuda.is_available() else 0.0,
        "binary_accuracy": accuracy_score(test_has_hate_true, test_has_hate_pred) * 100,
        "no_hate_f1": no_hate_f1,
        "has_hate_f1": has_hate_f1,
        "no_hate_precision": no_hate_precision,
        "has_hate_precision": has_hate_precision,
        "no_hate_recall": no_hate_recall,
        "has_hate_recall": has_hate_recall,
    }
    pd.DataFrame([summary]).to_csv(f"{out}/run_summary.csv", index=False)
    print(f"    Saved to {out}/run_summary.csv")

    print(f"\n✅ Done! Model & results saved to: {out}")
    print(f"⚠️  Note: 'exact_match' & 'token_f1' during training are PROXY metrics.")
    print(f"   REAL metrics are shown above (Process 1 & 2 from paper).")
    print(f"\n📊 Summary:")
    print(f"   Training Time: {total_train_time:.2f} minutes")
    print(f"   Peak VRAM: {peak_vram:.2f} GB / {total_vram if torch.cuda.is_available() else 0.0:.1f} GB")
    print(f"   Test Accuracy (Span): {test_acc:.2f}%")
    print(f"   Test F1 Macro (Span): {test_f1m:.2f}%")
    print(f"   Binary Accuracy (Has Hate): {summary['binary_accuracy']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
