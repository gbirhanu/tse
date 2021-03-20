import tokenizers
"""
This level one model was trained on google colab pro and the folder mentioned here is 
from google driver.
"""
MODEL_SAVE_PATH = '/content/drive/MyDrive/gema/roberta-squad2' # save the model here
TRAINED_MODEL_PATH = '/content/drive/MyDrive/gema/roberta-squad2' # load the model from here
TRAIN0 = '/content/train/train_folds0.csv' # five fold sample for seed0
TRAIN1 = '/content/train/train_folds1.csv' # five fold sample for seed1
TRAIN2 = '/content/train/train_folds2.csv' # five fold sample for seed2
TRAIN3 = '/content/train/train_folds3.csv' # five fold sample for seed3
TRAIN4 = '/content/train/train_folds4.csv' # five fold sample for seed4
TEST = '/content/train/test.csv' # test data 
SAMPLE_SUBMISSION = '/content/train/sample_submission.csv'  #submission format


# pretrained Model check point save in my google drive 
PRETRAINED_MODEL = '/content/drive/MyDrive/gema/roberta-base-squad2'

# Model paramameters
SEED0 = 25
SEED1 = 42
SEED2 = 2252021
SEED3 = 3162021 
SEED4 = 3058
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 106  # it is 106 becouse of the space in the tweet any number less than 106 did not work for me
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab='/content/drive/MyDrive/gema/roberta-base/vocab.json',
    merges='/content/drive/MyDrive/gema/roberta-base/merges.txt',
    lowercase=True,
    add_prefix_space=True)
HIDDEN_SIZE = 768   #hidden size from roberta model
N_LAST_HIDDEN = 12  # head of roberta
HIGH_DROPOUT = 0.5  # multisample dropout was used in the model for regulirazation 
SOFT_ALPHA = 0.6
WARMUP_RATIO = 0.25 # warming up learning rate by this ratio 
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
