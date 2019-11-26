import torch
from torch import Tensor, LongTensor
from fastai.metrics import auc_roc_score, fbeta


def auc_roc_score_multi(input, targ):
    """area under curve for multi category list (multiple bce losses)."""
    n = input.shape[1]
    scores = [auc_roc_score(input[:, i], targ[:, i]) for i in range(n)]
    return torch.tensor(scores).mean()

def fbeta_binary(y_pred, y_true, **args):
    return fbeta(y_pred[:, None], y_true[:, None], **args)
