from os import mkdir
from second_level_model import config
from second_level_model import run_oof_pred
import pickle
import os
"""
create directory to save char level oof prediction
"""
path = "char_level"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
char_pred_oof_start = []
char_pred_oof_end = []
train_file = [config.TRAIN0,config.TRAIN1,config.TRAIN2,config.TRAIN3,config.TRAIN4]
for i in range(5):
    for j in range(5):
        oof_start, oof_end = run_oof_pred(i,j,train_file[i])
        char_pred_oof_start.extend(oof_start)
        char_pred_oof_end.extend(oof_end)
    with open(f'./char_level/roberta{i}-char_pred_oof_start.pkl', 'wb') as handle:
        pickle.dump(char_pred_oof_start, handle)
    char_pred_oof_start.clear()
    with open(f'./char_level/roberta{i}-char_pred_oof_end.pkl', 'wb') as handle:
        pickle.dump(char_pred_oof_end, handle)
    char_pred_oof_end.clear()
