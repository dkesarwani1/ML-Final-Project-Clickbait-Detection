import os
import sys
import torch

import config
from data_loader import load_and_split, tokenize_for_model
from model_utils import (
    train_one_model,
    save_comparison,
    print_comparison_table,
)


def main():
    # Which models to run — either one specific key from CLI, or all of them.
    if len(sys.argv) > 1:
        requested = sys.argv[1]
        if requested not in config.MODELS:
            print(f"Unknown model '{requested}'. "
                  f"Available: {list(config.MODELS.keys())}")
            sys.exit(1)
        models_to_run = {requested: config.MODELS[requested]}
    else:
        models_to_run = config.MODELS

    # Quick hardware sanity check
    if torch.cuda.is_available():
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print("Using device: mps (Apple Silicon GPU)")
        print("  Note: fp16 is unavailable on MPS — training runs in fp32.")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")
        print("  Consider running this on Google Colab, Kaggle, or any CUDA machine.")

    # Load + split ONCE
    raw_splits = load_and_split()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    all_results = {}
    for key, model_name in models_to_run.items():
        tokenized, tokenizer = tokenize_for_model(raw_splits, model_name)

        output_dir = os.path.join(config.OUTPUT_DIR, key)
        os.makedirs(output_dir, exist_ok=True)

        results = train_one_model(model_name, tokenized, tokenizer, output_dir)
        all_results[key] = results

        # Free memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Only write/print a comparison if we trained multiple models.
    if len(all_results) > 1:
        print_comparison_table(all_results)
        save_comparison(all_results)


if __name__ == "__main__":
    main()