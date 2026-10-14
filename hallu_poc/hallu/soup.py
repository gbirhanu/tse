"""Weight averaging ("model soup", Wortsman et al. 2022) for Part E step 1.

Several fine-tuned copies of the same starting model often sit close enough
together that the plain average of their weights works as one model. If it
does, we keep most of an ensemble's gain at the cost of a single model.
"""
import torch

from hallu import encoder

SOUP_VERSION = 'uniform_v1'


def average_models(member_dirs, log=print):
    """Loads each saved model in turn and returns (tokenizer, averaged model) on CPU."""
    if len(member_dirs) < 2:
        raise ValueError('Averaging needs at least two models')
    total, keep_tok, keep_model = None, None, None
    for directory in member_dirs:
        log(f'Adding the weights of {directory}.')
        tok, model = encoder.load(directory, 'cpu')
        state = model.state_dict()
        if total is None:
            keep_tok, keep_model = tok, model
            total = {k: v.detach().float().clone() if v.is_floating_point() else v.clone()
                     for k, v in state.items()}
            continue
        if set(state) != set(total) or any(state[k].shape != total[k].shape for k in state):
            raise ValueError(f'{directory} has a different model shape; it cannot be averaged')
        for k, v in state.items():
            if v.is_floating_point():
                total[k] += v.detach().float()
            elif not torch.equal(v, total[k]):
                raise ValueError(f'{directory}: non-weight tensor {k} differs between members')
        del model, state
    reference = keep_model.state_dict()
    keep_model.load_state_dict({k: (v / len(member_dirs)).to(reference[k].dtype)
                                if v.is_floating_point() else v for k, v in total.items()})
    return keep_tok, keep_model
