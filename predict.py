"""
Inference utility — run any of the fine-tuned models on new headlines.

Usage:
    # Interactive mode
    python predict.py distilbert

    # Single headline from CLI
    python predict.py distilbert "You won't BELIEVE what happened next!"

    # Batch mode: pipe a file of headlines, one per line
    cat headlines.txt | python predict.py distilbert
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config


CLASS_NAMES = ["not clickbait", "clickbait"]


class ClickbaitDetector:
    """Thin wrapper around a fine-tuned model for single- and batch-predictions."""

    def __init__(self, model_key: str):
        if model_key not in config.MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(config.MODELS.keys())}"
            )

        # Path where main.py saved the fine-tuned checkpoint
        model_path = os.path.join(config.OUTPUT_DIR, model_key, "final_model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No fine-tuned model at {model_path}. "
                f"Run `python main.py {model_key}` first."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # disable dropout, etc. — we're not training

    @torch.no_grad()
    def predict(self, headlines):
        """
        Classify one headline (str) or many (list[str]).

        Returns a list of dicts with label, confidence, and both class probabilities.
        """
        if isinstance(headlines, str):
            headlines = [headlines]

        enc = self.tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        results = []
        for headline, pred, prob in zip(headlines, preds, probs):
            results.append({
                "headline":            headline,
                "label":               CLASS_NAMES[pred],
                "confidence":          float(prob[pred]),
                "prob_not_clickbait":  float(prob[0]),
                "prob_clickbait":      float(prob[1]),
            })
        return results


def _print_one(result: dict):
    emoji = "🎣" if result["label"] == "clickbait" else "📰"
    print(f"\n{emoji}  {result['headline']!r}")
    print(f"   → {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"   → P(clickbait)={result['prob_clickbait']:.3f}   "
          f"P(not clickbait)={result['prob_not_clickbait']:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <model_key> [headline]")
        print(f"  model_key ∈ {list(config.MODELS.keys())}")
        sys.exit(1)

    model_key = sys.argv[1]
    detector = ClickbaitDetector(model_key)

    # Case 1: headline passed on the CLI
    if len(sys.argv) >= 3:
        headline = " ".join(sys.argv[2:])
        _print_one(detector.predict(headline)[0])
        return

    # Case 2: headlines being piped in via stdin
    if not sys.stdin.isatty():
        headlines = [line.strip() for line in sys.stdin if line.strip()]
        for r in detector.predict(headlines):
            _print_one(r)
        return

    # Case 3: interactive REPL
    print(f"Loaded {model_key}. Enter headlines (blank line or Ctrl-C to exit).")
    try:
        while True:
            headline = input("\n> ").strip()
            if not headline:
                break
            _print_one(detector.predict(headline)[0])
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
