import torch
import transformers

from first_level_model import config


class TweetModel(torch.nn.Module):
    """
    RoBERTa (SQuAD2 checkpoint) with a start/end head.
    The head averages and max-pools the last N hidden layers and uses
    multi-sample dropout. Attribute names are the same as in 2021, so old
    model_XY.bin files still load.
    """

    def __init__(self, pretrained_dir, load_pretrained_weights=True,
                 n_last_hidden=config.N_LAST_HIDDEN,
                 high_dropout=config.HIGH_DROPOUT):
        super().__init__()
        conf = transformers.RobertaConfig.from_pretrained(pretrained_dir)
        conf.output_hidden_states = True
        if load_pretrained_weights:
            self.roberta = transformers.RobertaModel.from_pretrained(
                pretrained_dir, config=conf, add_pooling_layer=False)
        else:   # weights come later from a saved fold model
            self.roberta = transformers.RobertaModel(conf, add_pooling_layer=False)
        self.n_last_hidden = min(n_last_hidden, conf.num_hidden_layers + 1)
        self.high_dropout = torch.nn.Dropout(high_dropout)
        self.classifier = torch.nn.Linear(conf.hidden_size * 2, 2)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.roberta(ids, attention_mask=mask,
                               token_type_ids=token_type_ids,
                               output_hidden_states=True)
        hidden = outputs.hidden_states
        # (N_LAST_HIDDEN, batch_size, num_tokens, hidden_size)
        out = torch.stack(tuple(hidden[-i - 1] for i in range(self.n_last_hidden)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def load_fold_model(pretrained_dir, weights_path, device):
    """Builds the model from config.json only and loads a saved fold."""
    model = TweetModel(pretrained_dir, load_pretrained_weights=False)
    state = torch.load(weights_path, map_location='cpu')
    # older checkpoints may hold the unused pooler or position_ids buffers,
    # so extra keys are fine, but every weight the model needs must be there
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if 'position_ids' not in k]
    if missing:
        raise RuntimeError(f'{weights_path} is missing weights: {missing[:5]}')
    model.to(device)
    model.eval()
    return model
