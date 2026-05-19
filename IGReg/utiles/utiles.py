import os
import torch
import numpy as np
import matplotlib
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def save_model(model, config):
    F1 = config['Test_F1'] * 100
    name = "{}_F1_{:.2f}_step_{}.pkl".format(config['name'],
                                              F1,
                                              config['global_step'])
    model_path = os.path.join(config['save_dir'], name)
    torch.save(model, model_path)
    print("Saved model to {}".format(model_path))


def save_model_dict(model, config):
    F1 = config['Test_F1'] * 100
    name = "{}_F1_{:.2f}_step_{}.pkl".format(config['name'],
                                              F1,
                                              config['global_step'])
    model_path = os.path.join(config['save_dir'], name)
    torch.save(model.state_dict(), model_path)
    print("Saved model to {}".format(model_path))


def get_learning_rate(optimizer):
    if len(optimizer.param_groups) > 0:
        return optimizer.param_groups[0]['lr']
    else:
        raise ValueError('No trainable parameters.')



def to_device(tensor, gpu=False):
    return tensor.cuda(non_blocking=True) if gpu else tensor.cpu()


def plot_(name, save_path, type ='acc'):
    textsize = 15
    marker = 5
    matplotlib.use('Agg')

    loss_train = np.load(save_path + '/' + name + '_' + type + '_train.npy')
    loss_valid = np.load(save_path + '/' + name + '_' + type + '_valid.npy')

    plt.figure(dpi=300)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, len(loss_train), 1), loss_train, 'b-')
    ax1.plot(loss_valid, 'r--')
    if type =='acc':
        ax1.set_ylabel('Accuracy')
    else:
        ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    if type =='acc':
        lgd = plt.legend(['Train Acc', 'Test Acc'], markerscale=marker,
                         prop={'size': textsize, 'weight': 'normal'})
    else:
        lgd = plt.legend(['Train loss', 'Test loss'], markerscale=marker,
                         prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(save_path + '/' + name + '_' + type + '.png',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')

def clf_metrics(predictions, targets, average='macro'):
    matrix = confusion_matrix(targets, predictions)
    matrix = matrix.astype('int')

    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    acc = accuracy_score(targets, predictions)
    return acc, f1, precision, recall, matrix

def load_nii_affine(filename):
    if not os.path.exists(filename):
        return np.array([1])
    nii = nib.load(filename)
    data = nii.get_data()
    affine = nii.affine
    nii.uncache()
    return data, affine

def save_nii(arr, path, affine):
    nib.Nifti1Image(arr, affine).to_filename(path)


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