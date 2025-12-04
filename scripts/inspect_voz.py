
from datasets import load_dataset
import pandas as pd

try:
    print("Loading dataset tarudesu/VOZ-HSD...")
    dataset = load_dataset("tarudesu/VOZ-HSD")
    print("Dataset loaded successfully.")
    print(dataset)
    
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
        print("Train columns:", df.columns)
        print("Sample data:")
        print(df.head())
        if 'label' in df.columns or 'toxicity' in df.columns:
             print("Label distribution:")
             # Check for common label columns
             col = 'label' if 'label' in df.columns else 'toxicity'
             print(df[col].value_counts())
    
except Exception as e:
    print(f"Error loading dataset: {e}")
