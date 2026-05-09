"""
Configuration for Clickbait Detection project.
Centralizes all hyperparameters, model names, and paths so experiments stay consistent
across files.
"""

# --- Dataset ---
DATASET_NAME = "christinacdl/clickbait_notclickbait_dataset"
# Dataset columns: the actual names are auto-detected in data_loader.py via the
# candidate lists there (this dataset uses 'text' + 'label'), so these constants
# are kept only for documentation / reference.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Cap the total number of examples before splitting. None = use the full dataset.
# Useful for quick smoke-tests, e.g. MAX_SAMPLES = 1000
MAX_SAMPLES = 10000

# Split ratios (dataset ships with a single 'train' split; we carve it up)
VAL_SIZE = 0.10
TEST_SIZE = 0.10
RANDOM_SEED = 42

# --- Models ---
# Three transformers from the proposal, ordered roughly by size / capacity
MODELS = {
    "distilbert": "distilbert-base-uncased",  # 40% smaller, fast baseline
    "bert":       "bert-base-uncased",        # standard contextual baseline
    "roberta":    "roberta-base",             # deeper semantic representation
}

# --- Tokenization ---
# Headlines range 6–135 chars → ~30 wordpiece tokens typically. 128 is safe.
MAX_SEQ_LENGTH = 128

# --- Training ---
NUM_EPOCHS = 3            # BERT fine-tuning typically converges in 2–4 epochs
BATCH_SIZE = 32           # Lower to 16 if you hit OOM on the GPU
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5      # Canonical BERT fine-tuning LR (Devlin et al. 2019)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1        # 10% warmup of total steps
NUM_LABELS = 2            # binary: clickbait / not-clickbait

# --- Output paths ---
OUTPUT_DIR = "./results"
LOG_DIR = "./logs"
METRICS_FILE = "./results/comparison.json"