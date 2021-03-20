import create_fold
from first_level_model import helper
from first_level_model import config 
from first_level_model import run_train
import numpy as np 
seed = [config.SEED0, config.SEED1, config.SEED2, config.SEED3, config.SEED4]
train_file = [config.TRAIN0, config.TRAIN1,config.TRAIN2, config.TRAIN3, config.TRAIN4]


def run_diffrent_seed(seed,file_to_train,f_number):
    helper.seed_everything(seed)
    fold_scores = []
 
    for i in range(config.N_FOLDS):
        fold_score = run_train(i,file_to_train,f_number)
        fold_scores.append(fold_score)
    print('\nScores without SWA:')
    for i in range(config.N_FOLDS):
        print(f'Fold={f_number}{i}, Jaccard = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')


if __name__ == '__main__':
    for i in range(5):
        create_fold.create_fold(seed[i], i)
    for i in range(5):
       run_diffrent_seed(seed[i],train_file[i],i)