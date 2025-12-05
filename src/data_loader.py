"""
Data loading utilities for Vietnamese hate speech datasets.
"""

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None  # Will raise later if PhoBERT requires it


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length, use_word_seg=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_word_seg = use_word_seg

        if self.use_word_seg and word_tokenize is None:
            raise ImportError(
                "PhoBERT requires Vietnamese word segmentation. Install underthesea to proceed."
            )
    
    def __len__(self):
        return len(self.texts)
    
    def _maybe_segment(self, text: str) -> str:
        if not self.use_word_seg:
            return text
        return " ".join(word_tokenize(text))
    
    def __getitem__(self, idx):
        text = self._maybe_segment(str(self.texts[idx]))
        label = int(self.labels[idx])
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_vihsd() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHSD dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    vihsd = load_dataset("visolex/ViHSD")
    
    train_set = vihsd.filter(lambda ex: ex["type"] == "train")
    val_set = vihsd.filter(lambda ex: ex["type"] == "validation")
    test_set = vihsd.filter(lambda ex: ex["type"] == "test")
    
    train_df = train_set["train"].to_pandas()
    val_df = val_set["train"].to_pandas()
    test_df = test_set["train"].to_pandas()
    
    metadata = {
        "name": "ViHSD",
        "text_col": "free_text",
        "label_col": "label_id",
        "num_labels": int(pd.concat([
            train_df["label_id"],
            val_df["label_id"],
            test_df["label_id"]
        ]).nunique())
    }
    
    return train_df, val_df, test_df, metadata


def load_victsd() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViCTSD dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    train_set = load_dataset("tarudesu/ViCTSD", split="train")
    val_set = load_dataset("tarudesu/ViCTSD", split="validation")
    test_set = load_dataset("tarudesu/ViCTSD", split="test")
    
    train_df = train_set.to_pandas()
    val_df = val_set.to_pandas()
    test_df = test_set.to_pandas()
    
    metadata = {
        "name": "ViCTSD",
        "text_col": "Comment",
        "label_col": "Toxicity",
        "num_labels": 2  # Binary: 0=NONE, 1=TOXIC
    }
    
    return train_df, val_df, test_df, metadata


def load_vihos() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHOS dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    base = "https://raw.githubusercontent.com/phusroyal/ViHOS/master/"
    
    data_files = {
        "train": base + "data/Span_Extraction_based_version/train.csv",
        "validation": base + "data/Span_Extraction_based_version/dev.csv",
        "test": base + "data/Test_data/test.csv",
    }
    
    vihos = load_dataset("csv", data_files=data_files)
    
    train_df = vihos["train"].to_pandas()
    val_df = vihos["validation"].to_pandas()
    test_df = vihos["test"].to_pandas()
    
    # Create binary label: has hate span or not
    def has_hate_span(spans_str):
        if pd.isna(spans_str) or spans_str == "[]" or spans_str == "":
            return 0  # CLEAN
        return 1  # HAS HATE
    
    train_df["has_hate"] = train_df["index_spans"].apply(has_hate_span)
    val_df["has_hate"] = val_df["index_spans"].apply(has_hate_span)
    test_df["has_hate"] = test_df["index_spans"].apply(has_hate_span)
    
    metadata = {
        "name": "ViHOS",
        "text_col": "content",
        "label_col": "has_hate",
        "num_labels": 2  # Binary: 0=CLEAN, 1=HAS_HATE_SPANS
    }
    
    return train_df, val_df, test_df, metadata


def load_vihsd_processed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load ViHSD processed dataset (trinhtrantran122/ViHSD_processed).
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    base_url = "https://huggingface.co/datasets/trinhtrantran122/ViHSD_processed/resolve/main/"
    data_files = {
        "train": base_url + "train_processed.csv",
        "validation": base_url + "dev_processed.csv",
        "test": base_url + "test_processed.csv",
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Map string labels to integers
    # Based on inspection: 'none' -> 0, 'hate' -> 1
    label_map = {"none": 0, "hate": 1}
    
    def map_label(label):
        # Default to -1 if unknown, but we expect only none/hate
        return label_map.get(str(label).strip(), -1)
        
    train_df["label_id"] = train_df["label"].apply(map_label)
    val_df["label_id"] = val_df["label"].apply(map_label)
    test_df["label_id"] = test_df["label"].apply(map_label)
    
    metadata = {
        "name": "ViHSD_processed",
        "text_col": "free_text",
        "label_col": "label_id",
        "num_labels": 2
    }
    
    return train_df, val_df, test_df, metadata


def load_dataset_by_name(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: One of "ViHSD", "ViCTSD", "ViHOS"
    
    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    
    Raises:
        ValueError: If dataset_name is not recognized
    """
    loaders = {
        "ViHSD": load_vihsd,
        "ViCTSD": load_victsd,
        "ViHOS": load_vihos,
        "ViHSD_processed": load_vihsd_processed,
    }
    
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(loaders.keys())}"
        )
    
    return loaders[dataset_name]()


def build_torch_dataset(df: pd.DataFrame, text_col: str, label_col: str, 
                        tokenizer, max_length: int) -> TextDataset:
    """
    Build PyTorch Dataset from DataFrame.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        label_col: Name of label column
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        TextDataset instance
    """
    use_word_seg = "phobert" in str(getattr(tokenizer, "name_or_path", "")).lower()

    return TextDataset(
        df[text_col].tolist(),
        df[label_col].tolist(),
        tokenizer,
        max_length,
        use_word_seg=use_word_seg,
    )
