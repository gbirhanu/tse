import pandas as pd
import numpy as np
from second_level_model import config,helper,dataset

def run_train():
    """
    This function runs the second level model, since the model are run using 
    five diffrent seed we have to reorder them. in order to re-order we have to 
    first arrange the way they are trained in level one.
    """
    df0 = pd.read_csv(config.TRAIN0)
    df1 = pd.read_csv(config.TRAIN1)
    df2 = pd.read_csv(config.TRAIN2)
    df3 = pd.read_csv(config.TRAIN3)
    df4 = pd.read_csv(config.TRAIN4)

    df_train0 = pd.DataFrame(index=df0.index, columns=df0.columns)
    df_train1 = pd.DataFrame(index=df0.index, columns=df0.columns)
    df_train2 = pd.DataFrame(index=df0.index, columns=df0.columns)
    df_train3 = pd.DataFrame(index=df0.index, columns=df0.columns)
    df_train4 = pd.DataFrame(index=df0.index, columns=df0.columns)

    for i in range(5):
        df_train0.append(df0[df0.kfold == i].reset_index(drop=True))
    
    for i in range(5):
        df_train1.append(df1[df1.kfold == 0].reset_index(drop=True))
  
    for i in range(5):
        df_train2.append(df2[df2.kfold == 0].reset_index(drop=True))
    for i in range(5):
        df_train3.append(df3[df3.kfold == 0].reset_index(drop=True))
    for i in range(5):
        df_train4.append(df4[df4.kfold == 0].reset_index(drop=True))
    
    order_r0 = list(df_train0['textID'].values)
    order_r1 = list(df_train1['textID'].values)
    order_r2 = list(df_train2['textID'].values)
    order_r3 = list(df_train3['textID'].values)
    order_r4 = list(df_train4['textID'].values)
    
    df_test = pd.read_csv(config.DATA_PATH + 'test.csv').fillna('')
    df_test['selected_text'] = ''
    orders = {'r0': order_r0,
              'r1': order_r1,
              'r2': order_r2,
              'r3': order_r3,
              'r4': order_r4}
    
    (char_pred_oof_start, char_pred_oof_end,
     char_pred_test_start, char_pred_test_end) = helper.get_char_preds(orders, len(df_train0), len(df_test))
    tokenizer = config.TOKENIZER
    tokenizer.fit_on_texts(df_train0['text'].values)

    len_voc = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(df_train0['text'].values)
    X_test = tokenizer.texts_to_sequences(df_test['text'].values)
    preds = {'test_start': np.array(char_pred_test_start),
             'test_end': np.array(char_pred_test_end),
             'oof_start': np.array(char_pred_oof_start),
             'oof_end': np.array(char_pred_oof_end)}

    pred_oof, pred_tests = helper.k_fold(df_train0, df_test,
                                         np.array(X_train), np.array(X_test),
                                         preds, len_voc,
                                         k=config.N_FOLDS, model_seed=config.MODEL_SEED,
                                         fold_seed=config.SEED0,
                                         verbose=1, save=True, cp=False)

    train_dataset = dataset.TweetCharDataset(df_train0, X_train,
                                             preds['test_start'],
                                             preds['test_end'],
                                             max_len=config.MAX_LEN,
                                             train=True,
                                             n_models=len(config.MODELS))

    selected_texts_oof = helper.string_from_preds_char_level(
        train_dataset, pred_oof,
        test=False, remove_neutral=config.REMOVE_NEUTRAL)

    scores = [helper.jaccard(pred, truth) for (pred, truth) in zip(
        selected_texts_oof, df_train0['selected_text'])]
    score = np.mean(scores)
    print(f'Local CV score is {score:.4f}')
if __name__ == '__main__':
    run_train()


