"""
Inference script for hate speech detection.

Usage:
    python src/inference.py --model_path models/ViHSD_phobert --text "Your text here"
"""

import argparse
import torch
from model import load_trained_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on text")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--text", type=str, required=True,
                       help="Text to classify")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    
    return parser.parse_args()


def predict(model, tokenizer, text: str, max_length: int = 256, device: str = "cuda"):
    """
    Run inference on a single text.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text
        max_length: Maximum sequence length
        device: Device to run on
    
    Returns:
        Tuple of (predicted_label, probabilities)
    """
    model.eval()
    
    # Tokenize
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(logits, dim=-1).item()
    
    return pred_label, probs.cpu().numpy()[0]


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 80)
    print(f"Inference Configuration:")
    print("=" * 80)
    print(f"  Model path: {args.model_path}")
    print(f"  Text: {args.text}")
    print("=" * 80)
    
    # Load model
    print(f"\n🤖 Loading model from {args.model_path}...")
    model, tokenizer = load_trained_model(args.model_path)
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    print(f"  Device: {device}")
    
    # Run inference
    print(f"\n🔍 Running inference...")
    pred_label, probs = predict(model, tokenizer, args.text, args.max_length, device)
    
    print("\n" + "=" * 80)
    print(f"Results:")
    print("=" * 80)
    print(f"  Predicted label: {pred_label}")
    print(f"  Probabilities:")
    for i, prob in enumerate(probs):
        print(f"    Class {i}: {prob:.4f}")
    print("=" * 80)
    
    # Interpretation
    if len(probs) == 2:
        if pred_label == 0:
            print("\n✅ Classification: CLEAN (No hate speech detected)")
        else:
            print("\n⚠️  Classification: HATE SPEECH DETECTED")
    else:
        print(f"\n📊 Classification: Class {pred_label}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
