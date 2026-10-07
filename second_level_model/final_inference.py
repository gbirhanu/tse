"""Writes submission.csv from the level-one test predictions and the level-two models."""
from first_level_model.lo_inference import read_test
from second_level_model import config, helper, train_routine


def run_inference(args, paths, device):
    df_test = read_test(paths, args.test_limit)
    df_test['selected_text'] = ''
    seeds = args.seeds

    test_start, test_end = helper.load_char_preds(paths, seeds, 'test', list(df_test.textID))

    # same character ids as in training (saved by the train2 step)
    tokenizer = helper.CharTokenizer.load(paths.char_vocab_json)
    X_test = helper.to_object_array(tokenizer.texts_to_sequences(df_test['text'].values))
    preds = {'test_start': helper.to_object_array(test_start),
             'test_end': helper.to_object_array(test_end)}

    pred_tests, test_dataset = train_routine.infer(df_test, X_test, preds, len(tokenizer),
                                                   len(seeds), paths, device,
                                                   k=config.N_FOLDS)
    selected_texts = helper.string_from_preds_char_level(
        test_dataset, pred_tests, test=True, remove_neutral=config.REMOVE_NEUTRAL)

    sub_df = df_test[['textID']].copy()
    sub_df['selected_text'] = selected_texts
    sub_df.to_csv(paths.submission_csv, index=False)
    print(f'Wrote {paths.submission_csv} ({len(sub_df)} rows)')
    return sub_df
