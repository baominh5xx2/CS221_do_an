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
from datetime import datetime
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "False"
from data_loader import load_dataset_by_name  # noqa: E402


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
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    
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

    # Load ViHOS
    train_df, val_df, test_df, _ = load_dataset_by_name("ViHOS", dev_ratio=args.dev_ratio)
    print(f"📚 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print(f"⚙️  Config:")
    print(f"   Model: {args.model_name}")
    print(f"   Batch: {args.batch_size}")
    print(f"   LR: {args.lr}")
    print(f"   Optim: {args.optim}")
    print(f"   Sched: {args.lr_scheduler_type}")
    print(f"   FP16: {args.fp16}, BF16: {args.bf16}")

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
        fp16=args.fp16,
        bf16=args.bf16,
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
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)

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

    print(f"\n✅ Done! Model & results saved to: {out}")
    print(f"⚠️  Note: 'exact_match' & 'token_f1' during training are PROXY metrics.")
    print(f"   REAL metrics are shown above (Process 1 & 2 from paper).")


if __name__ == "__main__":
    main()
