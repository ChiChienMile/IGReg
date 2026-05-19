import torch
import numpy as np
from medpy import metric as metric_medpy
from torch.nn import functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_onehot_fall(Prediction):
    seg_class=2
    Prediction_ = torch.argmax(Prediction, dim=1, keepdim=True)
    output_ = F.one_hot(Prediction_.long(), seg_class)
    output = torch.Tensor.permute(output_, [0, 5, 2, 3, 4, 1])
    output = output.view(output.size(0), output.size(1), output.size(2) * output.size(3) * output.size(4))
    return output


def calculate_evaluation(outseg_list, target1, target2, target3):
    output1 = get_onehot_fall(outseg_list[0])
    output2 = get_onehot_fall(outseg_list[1])
    output3 = get_onehot_fall(outseg_list[2])

    target1 = target1.view(target1.size(0), target1.size(1), target1.size(2) * target1.size(3) * target1.size(4))
    target2 = target2.view(target2.size(0), target2.size(1), target2.size(2) * target2.size(3) * target2.size(4))
    target3 = target3.view(target3.size(0), target3.size(1), target3.size(2) * target3.size(3) * target3.size(4))

    output_ = ((output1[:, 1, :]).data.cpu().numpy()).astype('int')
    target_ = (target1[:, 1, :].data.cpu().numpy()).astype('int')
    dice_1 = metric_medpy.dc(output_, target_)

    output_ = ((output2[:, 1, :]).data.cpu().numpy()).astype('int')
    target_ = (target2[:, 1, :].data.cpu().numpy()).astype('int')
    dice_2 = metric_medpy.dc(output_, target_)

    output_ = ((output3[:, 1, :]).data.cpu().numpy()).astype('int')
    target_ = (target3[:, 1, :].data.cpu().numpy()).astype('int')
    dice_3 = metric_medpy.dc(output_, target_)

    dice_mean = (dice_1 + dice_2 + dice_3)/3.0
    return dice_1, dice_2, dice_3, dice_mean


def calculate_evaluation_single(outseg_list, target1):
    output1 = get_onehot_fall(outseg_list)
    target1 = target1.view(target1.size(0), target1.size(1), target1.size(2) * target1.size(3) * target1.size(4))

    output_ = ((output1[:, 1, :]).data.cpu().numpy()).astype('int')
    target_ = (target1[:, 1, :].data.cpu().numpy()).astype('int')
    dice_1 = metric_medpy.dc(output_, target_)
    return dice_1

def calculate_evaluation_single_pseudo(outseg_list, target1):
    output1 = get_onehot_fall(outseg_list)
    target1 = get_onehot_fall(target1)

    output_ = ((output1[:, 1, :]).data.cpu().numpy()).astype('int')
    target_ = (target1[:, 1, :].data.cpu().numpy()).astype('int')
    dice_1 = metric_medpy.dc(output_, target_)
    return dice_1

def get_dice(output, target, index):
    output_ = ((output[:, index, :]).data.cpu().numpy()).astype('int')
    target_ = (target[:, index, :].data.cpu().numpy()).astype('int')
    dice = metric_medpy.dc(output_, target_)
    return dice


def calculate_evaluation_SOTA(Prediction, target, seg_class=4):
    Prediction_ = torch.argmax(Prediction, dim=1, keepdim=True)
    output_ = F.one_hot(Prediction_.long(), seg_class)
    output = torch.Tensor.permute(output_, [0, 5, 2, 3, 4, 1])

    output = output.view(output.size(0), output.size(1), output.size(2) * output.size(3) * output.size(4))
    target = target.view(target.size(0), target.size(1), target.size(2) * target.size(3) * target.size(4))

    dice_1 = get_dice(output, target, 1)
    dice_2 = get_dice(output, target, 2)
    dice_3 = get_dice(output, target, 3)
    dice_mean = (dice_1 + dice_2 + dice_3) / 3.0
    return dice_1, dice_2, dice_3, dice_mean
