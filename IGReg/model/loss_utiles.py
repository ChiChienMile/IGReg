from torch import nn
import torch
import numpy as np


def sum_tensor(inp, axes, keepdim=False):
    """
    Sum tensor along multiple axes (optionally keeping dims).
    """
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    Compute TP/FP/FN/TN from soft predictions against label maps or one-hot maps.

    Args:
        net_output (Tensor): Soft predictions [B, C, ...].
        gt (Tensor): Label map [B, 1, ...] or [B, ...], or one-hot [B, C, ...].
        axes (tuple/list): Axes to sum over (spatial dims).
        mask (Tensor or None): Optional mask [B, 1, ...] to ignore regions.
        square (bool): Square the terms before summation (rarely needed).

    Returns:
        tp, fp, fn, tn (Tensor): Per-class counts aggregated over 'axes'.
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        # Broadcast mask to all channels
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    Args:
        apply_nonlin (callable or None): Optional nonlinearity (e.g., softmax) applied to predictions.
        batch_dice (bool): If True, aggregate Dice over batch; else per-sample then average.
        do_bg (bool): If False, exclude background channel from Dice computation.
        smooth (float): Smoothing constant to avoid zero division.
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        axes = [0] + list(range(2, len(shp_x))) if self.batch_dice else list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nom = 2 * tp + self.smooth
        den = 2 * tp + fp + fn + self.smooth
        dc = nom / (den + 1e-8)

        if not self.do_bg:
            dc = dc[1:] if self.batch_dice else dc[:, 1:]
        return 1 - dc.mean()

