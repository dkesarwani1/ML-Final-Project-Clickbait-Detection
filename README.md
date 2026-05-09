# Clickbait Detection Using Machine Learning Algorithms

**CS 4375 — Group 19**
Deepansh Kesarwani · David Schmidt · Abdullah Abdulatif · Sameer Vashisth

Binary classification of news headlines as **clickbait** or **not clickbait**, using three fine-tuned transformer models: BERT, DistilBERT, and RoBERTa.

---

## Project structure

```
clickbait-detection/
├── config.py          # Hyperparameters, model names, paths
├── data_loader.py     # HF dataset loading + stratified splitting + tokenization
├── model_utils.py     # Training loop + evaluation metrics
├── main.py            # Entry point — trains + compares all three models
├── predict.py         # Inference on new headlines
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. Without one, training will take several hours per model. On Colab/Kaggle with a T4/V100, expect ~10–20 minutes per model.

---

## Dataset

Pulled automatically from HuggingFace Hub on first run:

- **Source:** [`christinacdl/clickbait_notclickbait_dataset`](https://huggingface.co/datasets/christinacdl/clickbait_notclickbait_dataset)
- **Size:** 54,753 English headlines (deduplicated)
- **Task:** binary classification (1 = clickbait, 0 = non-clickbait)
- **Columns:** `text` + `label`
- **Upstream splits:** train (43,802) / validation (2,191) / test (8,760)

`data_loader.py` concatenates the upstream splits and re-carves an 80/10/10 train/val/test split with stratification on the label and a fixed seed (42). This keeps splits reproducible across dataset revisions and guarantees every model in the comparison sees the exact same rows. Set `MAX_SAMPLES = None` in `config.py` to use the full dataset (default is capped at 1000 for fast smoke-tests — bump it up for the real run).

---

## Training

```bash
# Train all three models and print a comparison table at the end
python main.py

# Or train just one
python main.py distilbert
python main.py bert
python main.py roberta
```

Outputs land in `./results/<model_key>/`:

- `final_model/` — fine-tuned weights + tokenizer (loadable with `AutoModel.from_pretrained`)
- `logs/` — per-step training logs
- `checkpoint-*/` — intermediate checkpoints (the best one is loaded at end)

After all three finish, `./results/comparison.json` contains the side-by-side metrics.

---

## Inference

After training, classify any headline:

```bash
# One-off
python predict.py distilbert "You won't BELIEVE what happened next!"

# Interactive REPL
python predict.py distilbert

# Batch from a file
cat headlines.txt | python predict.py distilbert
```

Output gives the predicted label, confidence, and both class probabilities.

---

## Hyperparameters

Defaults in `config.py`, following the standard BERT fine-tuning recipe (Devlin et al. 2019):

| Hyperparameter | Value | Notes |
|---|---|---|
| Max sequence length | 128 | Headlines are short; 128 tokens is comfortably above the longest. |
| Batch size | 32 | Drop to 16 if you hit OOM on a smaller GPU. |
| Learning rate | 2e-5 | Canonical for BERT-style fine-tuning. |
| Epochs | 3 | Early stopping (patience 2) kicks in if val F1 plateaus. |
| Weight decay | 0.01 | |
| Warmup ratio | 10% | Linear warmup then linear decay. |
| Optimizer | AdamW | Default in HF Trainer. |
| Mixed precision | fp16 | Enabled automatically if CUDA is available. |

---

## Evaluation

Reported on the held-out test split:

- **Accuracy** — overall correctness
- **Precision** — of headlines flagged as clickbait, how many actually are
- **Recall** — of actual clickbait headlines, how many we caught
- **F1** — harmonic mean (our primary metric for model selection)
- **Confusion matrix** — full breakdown
- **Per-class classification report**

F1 is the primary metric — it balances flagging legit headlines (precision) against missing actual clickbait (recall).

---

## Why these three models?

| Model | Params | Why it's in the comparison |
|---|---|---|
| **DistilBERT** | 66M | 40% smaller than BERT, ~60% faster, retains ~97% of BERT's performance. Ideal for short headlines where full BERT may be overkill. |
| **BERT** | 110M | Standard bidirectional baseline — the reference point every NLP paper compares against. |
| **RoBERTa** | 125M | Same architecture as BERT but trained longer on more data with better hyperparameters. Stronger semantic representations, especially useful for the "deceptive clickbait" cases from Scott (2023) that lack obvious lexical signals. |

---

## References

- Potthast et al. (2016) — *Clickbait Detection* (Webis Clickbait Corpus)
- Scott, K. (2023) — "Pragmatics and Clickbait" — *Journal of Pragmatics*
- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Liu et al. (2019) — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Sanh et al. (2019) — *DistilBERT, a distilled version of BERT*# ML-Final-Project-Clickbait-Detection
