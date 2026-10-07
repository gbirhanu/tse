"""
Builds a tiny, randomly initialised RoBERTa folder for a quick CPU test of the
pipeline when the real deepset/roberta-base-squad2 is not available.
The scores it gives mean nothing; it only shows that every step runs.

    python tools/make_tiny_pretrained.py --train-csv data/train.csv --out work_smoke/tiny-roberta
"""
import argparse
import os

import pandas as pd
import tokenizers
import transformers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--vocab-size', type=int, default=8000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    texts = pd.read_csv(args.train_csv).text.dropna().astype(str).tolist()
    # repeat the sentiment words so each becomes a single token, as in roberta-base
    texts += ['positive negative neutral'] * 2000
    tok = tokenizers.ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
    tok.train_from_iterator(texts, vocab_size=args.vocab_size, min_frequency=2,
                            special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    tok.save_model(args.out)

    conf = transformers.RobertaConfig(
        vocab_size=tok.get_vocab_size(), hidden_size=64, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128, max_position_embeddings=514,
        type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    transformers.RobertaModel(conf).save_pretrained(args.out)
    print(f'Tiny RoBERTa saved to {args.out}')


if __name__ == '__main__':
    main()
