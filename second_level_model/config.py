"""
Level-two model (character-level RNN) hyperparameters.
Paths are passed to main.py; see paths.py for the folder layout.
"""

MODEL_SEED = 2021
FOLD_SEED = 25          # seed of the level-two fold split (was SEED0)
N_FOLDS = 5
EPOCHS = 20
LR = 5e-3
WARMUP_PROP = 0.0
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 512
MAX_LEN = 150           # characters; the longest tweet has 142
SENT_EMBED_DIM = 16
CHAR_EMBED_DIM = 8
FT_LSTM_DIM = 16
LSTM_DIM = 64
USE_MSD = True
SWA_FIRST_EPOCH = 5     # weights are averaged from this epoch on (SWA)

# Loss
loss_config = {'smoothing': True,
               'eps': 0.1}

# Postprocessing: this did not help much, so neutral texts are not replaced
REMOVE_NEUTRAL = False
