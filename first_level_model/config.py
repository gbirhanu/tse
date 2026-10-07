"""
Level-one model (RoBERTa) hyperparameters.

Paths are no longer set here. Data, pretrained model and output folders are
passed to main.py on the command line, so the same code runs on Kaggle,
Colab or a local machine.
"""

# One fold split and one training run per seed (5 seeds x 5 folds = 25 models)
SEEDS = [25, 42, 2252021, 3162021, 3058]
N_FOLDS = 5

EPOCHS = 4
LEARNING_RATE = 4e-5
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
# 106 because extra spaces in the tweets are kept; shorter did not fit all tweets
MAX_LEN = 106

N_LAST_HIDDEN = 12   # hidden layers averaged/max-pooled by the head
HIGH_DROPOUT = 0.5   # multi-sample dropout in the head
SOFT_ALPHA = 0.6     # weight of the true start/end in the soft (KL) targets
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
ADAM_EPS = 1e-6      # same default as the old transformers.AdamW

# The 2021 code marked the target one character to the right of the selected
# text. False = fixed targets. True = reproduce the old behaviour.
LEGACY_TARGET_SHIFT = False
