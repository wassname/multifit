import torch
from torch import Tensor, LongTensor
from fastai.metrics import auc_roc_score, fbeta


def auc_roc_score_multi(input, targ):
    """area under curve for multi category list (multiple bce losses)."""
    n = input.shape[1]
    targ = targ * 1
    if targ.shape != input.shape:
        targ = targ.expand(input.T.shape).T
    scores = [auc_roc_score(input[:, i], targ[:, i]) for i in range(n)]
    return torch.tensor(scores).mean()


def fbeta_binary(y_pred, y_true, **args):
    return fbeta(y_pred[:, None], y_true[:, None], **args)


def auc_roc_score(input: Tensor, targ: Tensor):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input.squeeze(), targ.squeeze())
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


def accuracy_binary(input, targs):
    input = torch.sigmoid(input) > 0.5
    return (input == targs).float().mean()


def dice_binary(input, targs, iou=False, eps=1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    input = torch.sigmoid(input) > 0.5
    intersect = (input * targs).sum(dim=1).float()
    union = (input + targs).sum(dim=1).float()
    if not iou:
        l = 2.0 * intersect / union
    else:
        l = intersect / (union - intersect + eps)
    l[union == 0.0] = 1.0
    return l.mean()
