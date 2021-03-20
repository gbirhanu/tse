from keras.preprocessing.text import Tokenizer
import tokenizers
DATA_PATH = '../input/tweet-sentiment-extraction/'
PKL_PATH = './char_level/'
LEVEL_ONE_MODEL_PATH = '../input/level-one-model'
PRETRAINED_MODEL = '../input/roberta-base-squad/roberta_base_squad'

CP_PATH = './'
TRAIN0 = './train_folds0.csv'
TRAIN1 = './train_folds1.csv'
TRAIN2 = './train_folds2.csv'
TRAIN3 = './train_folds3.csv'
TRAIN4 = './train_folds4.csv'
TEST   = '../input/tweet-sentiment-extraction/test.csv'
# Fold seeds
SEED0 = 25
SEED1 = 42
SEED2 = 2252021
SEED3 = 3162021 
SEED4 = 3058
# 1st level models
MODELS = [('roberta0-','r0') , ('roberta1-','r1'),('roberta2-','r2'),('roberta3-','r3'),('roberta4-','r4')]
selected_model = "rnn"
HIDDEN_SIZE = 768
N_LAST_HIDDEN = 12
HIGH_DROPOUT = 0.5
MAX_LEN_MODEL_ONE = 106  #
TOKENIZER_MODEL_ONE = tokenizers.ByteLevelBPETokenizer(
    vocab='../input/roberta-base-squad/roberta_base_squad/vocab.json',
    merges='../input/roberta-base-squad/roberta_base_squad/merges.txt',
    lowercase=True,
    add_prefix_space=True)
SOFT_ALPHA = 0.6
# second level Model parameters
MODEL_SEED = 2021
N_FOLDS = 5
EPOCHS = 20
LR = 5e-3
WAMUP_PROP = 0.0
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 512
VALID_BATCH_SIZE_ONE = 32
TOKENIZER = Tokenizer(num_words=None, char_level=True,
                        oov_token='UNK', lower=True)
MAX_LEN = 150
SENT_EMBED_DIM = 16
CHAR_EMBED_DIM = 8
FT_LSTM_DIM = 16
LSTM_DIM = 64
USE_MSD = True

# Loss 
loss_config = {'smoothing': True,
                'eps': 0.1}

# Postprocessing This does not help much so I did not remove neutral texts
REMOVE_NEUTRAL = False