"""
Data loading and preprocessing for the clickbait classification task.

Pipeline:
    1. Pull christinacdl/clickbait_notclickbait_dataset from HuggingFace Hub.
    2. Normalize column names (the Hub sometimes ships slightly different schemas).
    3. Concatenate whatever splits the dataset ships with, then carve into
       train/val/test ourselves with a fixed seed. Re-splitting (rather than
       trusting the upstream splits) keeps results reproducible across dataset
       revisions and lets MAX_SAMPLES work uniformly.
    4. Tokenize with the model-specific tokenizer (BERT, DistilBERT, and RoBERTa
       all use different vocabularies, so each needs its own pass).
"""

from typing import Tuple
from datasets import (
    load_dataset,
    DatasetDict,
    ClassLabel,
    concatenate_datasets,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config


# Column names the dataset might ship with, in priority order.
# christinacdl/clickbait_notclickbait_dataset uses 'text' + 'label'. Other
# clickbait datasets on the Hub use 'title'/'headline' + 'clickbait'/'is_clickbait',
# so we keep all the variants here for portability.
TEXT_COL_CANDIDATES  = ["text", "title", "headline", "sentence"]
LABEL_COL_CANDIDATES = ["labels", "label", "clickbait", "is_clickbait", "target", "y"]


def _normalize_columns(ds: DatasetDict) -> DatasetDict:
    """
    Standardize column names to ('text', 'labels') regardless of what the
    dataset ships with, and cast labels to a ClassLabel feature so that
    stratified splitting works.
    """
    sample_split = next(iter(ds.values()))
    cols = sample_split.column_names

    # --- text column ---
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in cols), None)
    if text_col is None:
        raise ValueError(f"No recognized text column found. Columns: {cols}")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    # --- label column ---
    label_col = next((c for c in LABEL_COL_CANDIDATES if c in cols), None)
    if label_col is None:
        raise ValueError(f"No recognized label column found. Columns: {cols}")
    if label_col != "labels":
        ds = ds.rename_column(label_col, "labels")

    # --- ensure labels is a ClassLabel feature ---
    # train_test_split's stratify_by_column requires ClassLabel, not a raw int column.
    sample_split = next(iter(ds.values()))
    if not isinstance(sample_split.features["labels"], ClassLabel):
        ds = ds.cast_column(
            "labels",
            ClassLabel(num_classes=2, names=["not_clickbait", "clickbait"]),
        )

    return ds


def load_and_split() -> DatasetDict:
    """
    Load the full dataset and create a stratified train/val/test split.

    Returns a DatasetDict with keys 'train', 'validation', 'test'.

    The dataset (christinacdl/clickbait_notclickbait_dataset) ships with
    pre-defined train/validation/test splits, but we re-carve our own anyway:
      - keeps results reproducible across upstream dataset revisions
      - lets MAX_SAMPLES work uniformly regardless of source structure
      - guarantees identical splits across all three models we benchmark

    Stratification keeps the class balance consistent across splits — important
    since clickbait detection metrics (precision, recall, F1) are class-sensitive.
    """
    print(f"Loading dataset: {config.DATASET_NAME}")
    raw = load_dataset(config.DATASET_NAME)
    raw = _normalize_columns(raw)

    # Concatenate every split the dataset ships with — could be just 'train',
    # or 'train'+'test', or 'train'+'validation'+'test', etc.
    incoming_splits = list(raw.keys())
    print(f"  Source dataset ships with splits: {incoming_splits}")
    if len(incoming_splits) == 1:
        full = raw[incoming_splits[0]]
    else:
        full = concatenate_datasets([raw[s] for s in incoming_splits])
    print(f"  Combined: {len(full)} total examples")

    if config.MAX_SAMPLES is not None:
        # Shuffle first — the raw dataset may be ordered by label or by split
        # (clickbait first, then non-clickbait). Taking the first N rows of an
        # ordered set gives a badly skewed sample. Seeded for reproducibility.
        full = full.shuffle(seed=config.RANDOM_SEED)
        full = full.select(range(min(config.MAX_SAMPLES, len(full))))
        print(f"  (capped to {len(full)} examples via MAX_SAMPLES, shuffled)")

    # First carve off test, then carve val out of what remains.
    test_split = full.train_test_split(
        test_size=config.TEST_SIZE,
        seed=config.RANDOM_SEED,
        stratify_by_column="labels",
    )
    val_rel = config.VAL_SIZE / (1.0 - config.TEST_SIZE)  # relative to remainder
    trainval_split = test_split["train"].train_test_split(
        test_size=val_rel,
        seed=config.RANDOM_SEED,
        stratify_by_column="labels",
    )

    ds = DatasetDict({
        "train":      trainval_split["train"],
        "validation": trainval_split["test"],
        "test":       test_split["test"],
    })

    # Quick sanity print — class balance per split
    print("\nSplit sizes and class distribution:")
    for name, split in ds.items():
        labels = split["labels"]
        pos = sum(labels)
        print(f"  {name:<11} n={len(split):>6}  "
              f"clickbait={pos:>5} ({pos/len(split)*100:.1f}%)  "
              f"non-clickbait={len(split)-pos:>5}")

    return ds


def tokenize_for_model(
    ds: DatasetDict,
    model_name: str,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase]:
    """
    Tokenize every split for a specific model. Returns tokenized dataset + tokenizer
    (kept around for inference later).

    Each transformer uses a different subword vocabulary, so this has to be redone
    per model — can't tokenize once and reuse across BERT/DistilBERT/RoBERTa.
    """
    print(f"\nTokenizing with {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],  # drop raw text; keep only tensors + labels
        desc=f"Tokenizing for {model_name}",
    )

    # Tell HF to hand PyTorch tensors to the Trainer.
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized, tokenizer


def prepare_data(model_name: str) -> Tuple[DatasetDict, PreTrainedTokenizerBase]:
    """End-to-end helper: load → split → tokenize for a single model."""
    ds = load_and_split()
    return tokenize_for_model(ds, model_name)