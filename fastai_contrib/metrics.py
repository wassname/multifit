import torch
from torch import Tensor, LongTensor
from fastai.metrics import auc_roc_score, fbeta

def fbeta_cls_n(y_pred, y_true, class_n=1, **args):
    """F1 score of class 1, to be used with 2 classes."""
    y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
    return fbeta(y_pred, y_true[:, None], sigmoid=False, **args)

def auc_roc_score_cls_n(y_pred, y_true, class_n=1, **args):
    """auc_roc_score score of class 1, to be used with 2 classes."""
    y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
    return auc_roc_score(y_pred[:, class_n], y_true==class_n, **args)

def auc_roc_score(input: Tensor, targ: Tensor):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.0).sum(-1)


def roc_curve(input: Tensor, targ: Tensor):
    "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
    # wassname: fix this by making LongTensor([0]=>device)
    targ = targ == 1
    desc_score_indices = torch.flip(input.argsort(-1), [-1])
    input = input[desc_score_indices]
    targ = targ[desc_score_indices]
    d = input[1:] - input[:-1]
    distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]
    threshold_idxs = torch.cat(
        (distinct_value_indices, LongTensor([len(targ) - 1]).to(targ.device))
    )
    tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    if tps[0] != 0 or fps[0] != 0:
        zer = torch.zeros(1, dtype=fps.dtype, device=fps.device)
        fps = torch.cat((zer, fps))
        tps = torch.cat((zer, tps))
    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
    return fpr, tpr


def f1_score(*args, **kwargs):
    return fbeta_cls_n(beta=1, thresh=0.5, *args, **kwargs)

def accuracy_binary(input, targs):
    input = torch.sigmoid(input) > 0.5
    return (input == targs).float().mean()

def mean_output(input, targs):
    return input.mean()
