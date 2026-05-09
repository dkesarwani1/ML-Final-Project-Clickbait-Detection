import os
import json
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config


def compute_metrics(eval_pred):
    """
    Hook that Trainer calls after every eval pass. Returns the metrics to log.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_training_args(output_dir: str, num_train_examples: int) -> TrainingArguments:
    """
    Build a TrainingArguments object with sensible defaults.
    """
    steps_per_epoch = max(1, num_train_examples // config.BATCH_SIZE)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,

        seed=config.RANDOM_SEED,
    )


def train_one_model(model_name: str, tokenized_data, tokenizer, output_dir: str) -> dict:
    """
    Fine-tune a single transformer on the clickbait dataset.
    """
    print(f"\n{'='*70}")
    print(f"  Training {model_name}")
    print(f"{'='*70}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        num_train_examples=len(tokenized_data["train"]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    test_metrics = trainer.evaluate(tokenized_data["test"], metric_key_prefix="test")

    preds = trainer.predict(tokenized_data["test"])
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                pred=0   pred=1")
    print(f"  true=0 (neg)  {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  true=1 (pos)  {cm[1,0]:>6}   {cm[1,1]:>6}")

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["non-clickbait", "clickbait"],
        digits=4,
    ))

    # Save model + tokenizer
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved fine-tuned model and tokenizer to {final_path}")

    results = {
        "model_name": model_name,
        "test_accuracy": float(test_metrics.get("test_accuracy", 0)),
        "test_precision": float(test_metrics.get("test_precision", 0)),
        "test_recall": float(test_metrics.get("test_recall", 0)),
        "test_f1": float(test_metrics.get("test_f1", 0)),
        "test_loss": float(test_metrics.get("test_loss", 0)),
        "confusion_matrix": cm.tolist(),
    }
    return results


def save_comparison(all_results: dict, path: str = config.METRICS_FILE):
    """Write the full comparison across models to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved to {path}")


def print_comparison_table(all_results: dict):
    """Pretty-print the final model comparison as a table."""
    print("\n" + "=" * 70)
    print(" FINAL MODEL COMPARISON (held-out test set)")
    print("=" * 70)
    print(f"{'Model':<14} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>8}")
    print("-" * 70)
    for key, r in all_results.items():
        print(f"{key:<14} "
              f"{r['test_accuracy']:>10.4f} "
              f"{r['test_precision']:>11.4f} "
              f"{r['test_recall']:>9.4f} "
              f"{r['test_f1']:>8.4f}")
    print("=" * 70)

    best = max(all_results.items(), key=lambda kv: kv[1]["test_f1"])
    print(f"\nBest model by F1: {best[0]} ({best[1]['test_f1']:.4f})")