import numpy as np
import pandas as pd 
import torch 
from second_level_model import config
import transformers 
from first_level_model import model,helper
from tqdm import tqdm 
def run_oof_pred(seed,fold, files_train):
    dfx = pd.read_csv(files_train)
    device = torch.device('cuda')
    model_config = transformers.RobertaConfig.from_pretrained(
        config.PRETRAINED_MODEL)
    model_config.output_hidden_states = True
    model = model.TweetModel(conf=model_config)
    model = model.to(device)
    # loading models
    model = model.TweetModel(conf=model_config)
    model.to(device)
    model.load_state_dict(torch.load(
        f'{config.LEVEL_ONE_MODEL_PATH}/model_{seed}{fold}.bin'),
        strict=False)
    model.eval()
    fold_models = model
    char_pred_oof_start = []
    char_pred_oof_end = []
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    valid_dataset = model.TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE_ONE,
        num_workers=4,
        shuffle=False)
    with torch.no_grad():
        tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            orig_tweet = d['orig_tweet']
            offsets = d['offsets']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs_start_folds = []
            outputs_end_folds = []
            outputs_start, outputs_end = \
                fold_models(ids=ids,
                            mask=mask,
                            token_type_ids=token_type_ids)
            outputs_start_folds.append(outputs_start)
            outputs_end_folds.append(outputs_end)



            outputs_start = torch.softmax(outputs_start, dim=-1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=-1).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                char_pred_oof_start.append(
                    helper.token_level_to_char_level(tweet, offsets[px], outputs_start[px]))
                char_pred_oof_end.append(
                    helper.token_level_to_char_level(tweet, offsets[px], outputs_end[px]))
            
    return  char_pred_oof_start, char_pred_oof_end