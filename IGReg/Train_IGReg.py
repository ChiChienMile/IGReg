# -*- coding: utf-8 -*-
"""
Training script for IGReg (Implicit Gradient Regularization) on multi-task glioma prediction.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # set visible GPUs before importing torch

import warnings
import time
import random

import torch
import monai
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

# Optimizer
from lion_pytorch import Lion

# -----------------------------
# Project-local dependencies
# -----------------------------

# Model
from model.IGReg import IGReg

# Losses
from model.loss_utiles import SoftDiceLoss

# Configs & utils
import config
from utiles.utiles import clf_metrics, to_device, AverageMeter

# Classification datasets
from dataLoder.Multi_1p19q_3D import Dataset_TrainBal_Ratio as Dataset_self_cls_1
from dataLoder.Multi_1p19q_3D import Dataset_Test as Dataset_self_cls_test_1

from dataLoder.Multi_IDH_3D import Dataset_TrainBal_Ratio as Dataset_self_cls_2
from dataLoder.Multi_IDH_3D import Dataset_Test as Dataset_self_cls_test_2

from dataLoder.Multi_HL_3D import Dataset_TrainBal_Ratio as Dataset_self_cls_3
from dataLoder.Multi_HL_3D import Dataset_Test as Dataset_self_cls_test_3

from evaluation.metrics_self import calculate_evaluation_single

# Segmentation dataset
from dataLoder.MICCAI_2021_3D import Dataset_Train as Dataset_Train_Seg
from dataLoder.MICCAI_2021_3D import Dataset_Test as Dataset_Test_Seg


# ============================================================
# I/O helpers
# ============================================================

def save_model(model, config):
    """
    Save full model object with a descriptive filename that includes key metrics.

    Args:
        model (torch.nn.Module): Trained model object to be saved via `torch.save(model, path)`.
        config (dict): Contains fields:
            - 'name', 'save_dir', 'global_step'
            - 'Test_F1', 'dice1'
            - 'F1_Test_HL', 'F1_Test_IDH', 'F1_Test_1p19q'
    """
    F1 = config['Test_F1'] * 100
    Dice = config['dice1'] * 100

    F1_HL = config['F1_Test_HL'] * 100
    F1_IDH = config['F1_Test_IDH'] * 100
    F1_1p19q = config['F1_Test_1p19q'] * 100

    name = "{}_F1_{:.2f}_Dice_{:.2f}" \
           "_F1_HL_{:.2f}_F1_IDH_{:.2f}_F1_1p19q_{:.2f}" \
           "_step_{}.pkl".format(config['name'], F1, Dice, F1_HL, F1_IDH, F1_1p19q, config['global_step'])
    model_path = os.path.join(config['save_dir'], name)
    torch.save(model, model_path)
    print("Saved model to {}".format(model_path))


# ============================================================
# Validation helpers (classification & segmentation)
# ============================================================

def validate(data_loader, model, valid_flag, ssl_index=0, cls_name='IDH'):
    """
    Evaluate classification head for a given task index (`ssl_index`).

    Args:
        data_loader (DataLoader): classification dataloader.
        model (IGReg): model instance with `.predictcls` method.
        valid_flag (str): 'Test' or other string used for printing.
        ssl_index (int): which classification branch to evaluate (0: 1p19q, 1: IDH, 2: HL).
        cls_name (str): human-readable class name for logging.

    Returns:
        tuple: (loss_avg, Accuracy, F1, Precision, Recall)
    """
    model.eval()
    loss_Cross = CrossEntropyLoss(reduction='mean')
    target, predictions = [], []
    loss_record = AverageMeter()

    for data in data_loader:
        img_tensor, cls_label = data

        img_tensor = to_device(img_tensor, gpu=config.use_cuda)
        cls_label = to_device(cls_label, gpu=config.use_cuda)

        with torch.no_grad():
            out = model.predictcls(img_tensor, ssl_index)
            loss = loss_Cross(out, cls_label)
            loss_record.update(loss.item(), 1)

            probs = torch.nn.functional.softmax(out)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            labelscpu = cls_label.cpu().detach().numpy()
            predictions.extend(preds)
            target.extend(labelscpu)

    loss_all = loss_record.avg
    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(target, dtype=np.int32)
    Accuracy, F1, Precision, Recall, matrix = clf_metrics(
        predictions=predictions,
        targets=gts,
        average="macro"
    )
    print(matrix)
    if valid_flag == 'Test':
        print('Test set {:} | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | Recall {:.4f} | '
              .format(cls_name, Accuracy, F1, Precision, Recall))
    else:
        print('Train set {:}| Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | Recall {:.4f} | '
              .format(cls_name, Accuracy, F1, Precision, Recall))
    model.train()

    return loss_all, Accuracy, F1, Precision, Recall


def validseg(data_loader, model):
    """
    Evaluate segmentation branch using SoftDiceLoss and per-class Dice (WT if two classes).

    Args:
        data_loader (DataLoader): segmentation dataloader.
        model (IGReg): model with `.predictseg`.

    Returns:
        tuple: (loss_avg, dice1_avg)
    """
    model.eval()
    loss_DC = SoftDiceLoss(smooth=1e-5, do_bg=False)
    losses_seg_record = AverageMeter()
    dice_1_record = AverageMeter()

    for data in data_loader:
        img, img_mask = data

        img = to_device(img, gpu=config.use_cuda)
        seg_mask = to_device(img_mask, gpu=config.use_cuda)

        with torch.no_grad():
            outSeg = model.predictseg(img)
            seg_mask1 = get_onehot_fall(seg_mask[:, 0, :])
            # The original code considered multiple classes; kept commented to preserve behavior.
            # seg_mask2 = get_onehot_fall(seg_mask[:, 1, :])
            # seg_mask3 = get_onehot_fall(seg_mask[:, 2, :])

            loss = loss_DC(outSeg, seg_mask1)
            dice1 = calculate_evaluation_single(outSeg, seg_mask1)

            losses_seg_record.update(loss.item(), 1)
            dice_1_record.update(dice1, 1)

    loss_all = losses_seg_record.avg
    dice1_all = dice_1_record.avg
    model.train()
    print('Test set | loss_all {:.4f} | dice_1 {:.4f}'.format(loss_all, dice1_all))
    return loss_all, dice1_all


def valid_metrics_by_cls(model, data_loader_Test, ssl_index, cls_name):
    """
    Wrapper for classification metrics (kept as in original).
    """
    loss_Test, Accuracy_Test, F1_Test, Precision_Test, Recall_Test = validate(
        data_loader=data_loader_Test,
        model=model,
        valid_flag='Test',
        ssl_index=ssl_index,
        cls_name=cls_name
    )
    metrics_test = [loss_Test, Accuracy_Test, F1_Test, Precision_Test, Recall_Test]
    return metrics_test


def valid_metrics_by_cls_single(model, data_loader_Test, ssl_index, cls_name):
    """
    Duplicate of `valid_metrics_by_cls` kept to preserve the public API.
    """
    loss_Test, Accuracy_Test, F1_Test, Precision_Test, Recall_Test = validate(
        data_loader=data_loader_Test,
        model=model,
        valid_flag='Test',
        ssl_index=ssl_index,
        cls_name=cls_name
    )
    metrics_test = [loss_Test, Accuracy_Test, F1_Test, Precision_Test, Recall_Test]
    return metrics_test


# ============================================================
# Dataloading helpers
# ============================================================

def get_cls_loder(Dataset_cls, Dataset_cls_test, val_transforms, loder_id, basic_ratio):
    """
    Build classification train/val dataloaders with the given Dataset classes.

    Args:
        Dataset_cls: training dataset class (balanced by ratio).
        Dataset_cls_test: test dataset class.
        val_transforms: monai transform pipeline (here used as ToTensor).
        loder_id (int): for logging purpose.
        basic_ratio (list or tuple): ratio spec e.g. [1,1,1].

    Returns:
        tuple: (train_loader_cls, val_loader_cls)
    """
    Cross = 1
    train_dataset_cls = Dataset_cls(
        basic_len=48, basic_ratio=basic_ratio,
        read_type='train',
        Cross=Cross,
        transform=val_transforms
    )
    train_loader_cls = DataLoader(
        train_dataset_cls,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print("Classification | Number of train_dataset_cls {} examples {}".format(loder_id, len(train_dataset_cls)))

    val_dataset_cls = Dataset_cls_test(
        read_type='test',
        Cross=Cross,
        transform=val_transforms
    )
    val_loader_cls = DataLoader(
        val_dataset_cls,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print("Classification | Number of val_loader_cls {} examples {}".format(loder_id, len(val_dataset_cls)))

    return train_loader_cls, val_loader_cls


def get_onehot_fall(Prediction):
    """
    Convert label volume [B, W, H, D] into one-hot tensor [B, C, W, H, D].

    Args:
        Prediction (torch.Tensor): long tensor of shape [B, W, H, D].

    Returns:
        torch.Tensor: one-hot with shape [B, C, W, H, D], here C=2.
    """
    seg_class = 2
    output_ = F.one_hot(Prediction.long(), seg_class)
    output = torch.Tensor.permute(output_, [0, 4, 1, 2, 3])
    return output


def get_cls_parameters(predictions, target):
    """
    Compute F1 given predictions and targets using `clf_metrics`.

    Args:
        predictions (list/ndarray)
        target (list/ndarray)

    Returns:
        float: F1 score (macro)
    """
    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(target, dtype=np.int32)
    Accuracy, F1, Precision, Recall, matrix = clf_metrics(
        predictions=predictions,
        targets=gts,
        average="macro"
    )
    return F1


# ============================================================
# Training loop
# ============================================================

def train(model, init_seg, ratio_list, Cross_Seg=1):
    """
    Main training loop combining segmentation and three classification tasks.

    Pipeline per epoch:
      1) Copy pretrained backbone params to G1/G2/G3.
      2) Stage-1: update task-specific heads (G1/G2/G3) for 100 iters.
      3) Stage-2: update main IGReg_DPA backbone with uncertainty-weighted objective for 100 iters.
      4) Run validation for all tasks + segmentation; log to CSV; save the best checkpoint by (Dice + mean F1).

    Args:
        model (IGReg): model instance.
        init_seg (int): batch size used for initializing segmentation clusters (first-time init).
        ratio_list (list[int]): ratios for the three classification datasets, e.g., [1,1,1].
        Cross_Seg (int): fold index for segmentation dataset.
    """
    val_transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),
    ])

    # ---------------- Segmentation dataloaders ----------------
    train_dataset_seg = Dataset_Train_Seg(Cross=Cross_Seg, transform=val_transforms)
    train_loader_seg = DataLoader(
        train_dataset_seg,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print("Segmentation | Number of training examples {}".format(len(train_dataset_seg)))

    test_dataset_seg = Dataset_Test_Seg(Cross=Cross_Seg, transform=val_transforms)
    test_loader_seg = DataLoader(
        test_dataset_seg,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print("Segmentation | Number of testing examples {}".format(len(test_dataset_seg)))

    # Loader used ONLY for initialization (cluster init)
    init_seg_dataset = Dataset_Train_Seg(Cross=Cross_Seg, transform=val_transforms)
    init_loader_seg = DataLoader(
        init_seg_dataset,
        batch_size=init_seg,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print("init_loader_seg | Number of training examples {}".format(len(init_seg_dataset)))

    # ---------------- Classification dataloaders ----------------
    train_loader_cls_1p19q, val_loader_cls_1p19q = get_cls_loder(
        Dataset_self_cls_1, Dataset_self_cls_test_1, val_transforms, loder_id=1, basic_ratio=ratio_list[0]
    )
    train_loader_cls_IDH, val_loader_cls_IDH = get_cls_loder(
        Dataset_self_cls_2, Dataset_self_cls_test_2, val_transforms, loder_id=2, basic_ratio=ratio_list[1]
    )
    train_loader_cls_HL, val_loader_cls_HL = get_cls_loder(
        Dataset_self_cls_3, Dataset_self_cls_test_3, val_transforms, loder_id=3, basic_ratio=ratio_list[2]
    )

    # ---------------- Logger (pandas DataFrame) ----------------
    log = pd.DataFrame(
        index=[],
        columns=[
            'epoch',
            'Train_loss',
            'Test_loss_1', 'Test_loss_2', 'Test_loss_3',
            'Test_F1_1', 'Test_F1_2', 'Test_F1_3',
            'Test_seg_loss', 'Test_Dice',
            'Mean_TeCls'
        ]
    )

    cls_name_final = ['1p19q', 'IDH', 'HL']

    if config.use_cuda:
        model.cuda()

    # ---------------- Losses ----------------
    ce_loss = CrossEntropyLoss(reduction='mean')
    dice_loss = SoftDiceLoss(smooth=1e-5, do_bg=False)

    # ---------------- Optimizers ----------------
    params_main = list(model.IGReg_DPA.parameters()) + [
        model.log_sigma1, model.log_sigma2, model.log_sigma3,
        model.log_sigma4, model.log_sigma5, model.log_sigma6
    ]
    optim_main = Lion(
        (p for p in params_main if p.requires_grad),
        lr=config.lr_self,
        weight_decay=config.weight_decay
    )
    optim_g1 = Lion(filter(lambda p: p.requires_grad, model.G1.parameters()),
                    lr=config.lr_self, weight_decay=config.weight_decay)
    optim_g2 = Lion(filter(lambda p: p.requires_grad, model.G2.parameters()),
                    lr=config.lr_self, weight_decay=config.weight_decay)
    optim_g3 = Lion(filter(lambda p: p.requires_grad, model.G3.parameters()),
                    lr=config.lr_self, weight_decay=config.weight_decay)

    # ---------------- Training state ----------------
    global_step = 0
    best_score_save = -1
    path_save = config.data_dir + '/results/' + str(save_name) + '/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    model.train()
    iter_index_cls_1 = 0
    iter_index_cls_2 = 0
    iter_index_cls_3 = 0
    iter_index_seg = 0

    iter_labeled_cls_1 = iter(train_loader_cls_1p19q)
    iter_labeled_cls_2 = iter(train_loader_cls_IDH)
    iter_labeled_cls_3 = iter(train_loader_cls_HL)
    iter_labeled_seg = iter(train_loader_seg)

    # ============================================================
    # Epoch loop
    # ============================================================
    for epoch in range(config.epochs):
        Train_loss_record = AverageMeter()

        # Reset best-score window after a certain epoch threshold (kept as in original)
        if epoch == config.save_epoch:
            best_score_save = -1

        # ------------------------------------------------------------
        # Stage 0: Copy pretrained backbone into each branch G1/G2/G3
        # ------------------------------------------------------------
        model.copy_param(model.G1, model.IGReg_DPA.BranchMain)
        model.copy_param(model.G2, model.IGReg_DPA.BranchMain)
        model.copy_param(model.G3, model.IGReg_DPA.BranchMain)

        # ------------------------------------------------------------
        # Stage 1: Update G1/G2/G3
        # ------------------------------------------------------------
        # Stage 2: Update main IGReg_DPA backbone
        # ------------------------------------------------------------
        for batch_idx in range(100):
            iter_index_cls_1 += 1
            iter_index_cls_2 += 1
            iter_index_cls_3 += 1
            iter_index_seg += 1

            # Recycle iterators if reaching the end
            if iter_index_cls_1 >= len(train_loader_cls_1p19q):
                iter_labeled_cls_1 = iter(train_loader_cls_1p19q)
                iter_index_cls_1 = 0
            if iter_index_cls_2 >= len(train_loader_cls_IDH):
                iter_labeled_cls_2 = iter(train_loader_cls_IDH)
                iter_index_cls_2 = 0
            if iter_index_cls_3 >= len(train_loader_cls_HL):
                iter_labeled_cls_3 = iter(train_loader_cls_HL)
                iter_index_cls_3 = 0
            if iter_index_seg >= len(train_loader_seg):
                iter_labeled_seg = iter(train_loader_seg)
                iter_index_seg = 0

            # ----- Segmentation batch -----
            data_seg = next(iter_labeled_seg)
            imgs_seg, seg_mask = data_seg
            imseg = to_device(imgs_seg, gpu=config.use_cuda)
            seg_mask = to_device(seg_mask, gpu=config.use_cuda)
            seg_onehot = get_onehot_fall(seg_mask[:, 0, :])

            # ----- Classification batches -----
            data_CLS_1p19q = next(iter_labeled_cls_1)
            imgs_1p19q, labels_1p19q = data_CLS_1p19q

            data_CLS_IDH = next(iter_labeled_cls_2)
            imgs_IDH, labels_IDH = data_CLS_IDH

            data_CLS_HL = next(iter_labeled_cls_3)
            imgs_HL, labels_HL = data_CLS_HL

            input_imgs = torch.cat((imgs_1p19q, imgs_IDH, imgs_HL), 0)
            input_imgs = input_imgs.reshape(
                input_imgs.size(0) * input_imgs.size(1),
                input_imgs.size(2), input_imgs.size(3),
                input_imgs.size(4), input_imgs.size(5)
            )
            imcls = to_device(input_imgs, gpu=config.use_cuda)

            label_list = []
            labels_1p19q = labels_1p19q.reshape(labels_1p19q.size(0) * labels_1p19q.size(1))
            labels_IDH = labels_IDH.reshape(labels_IDH.size(0) * labels_IDH.size(1))
            labels_HL = labels_HL.reshape(labels_HL.size(0) * labels_HL.size(1))

            label_cls1_to_list = to_device(labels_1p19q, gpu=config.use_cuda)
            label_cls2_to_list = to_device(labels_IDH, gpu=config.use_cuda)
            label_cls3_to_list = to_device(labels_HL, gpu=config.use_cuda)

            label_list.append(label_cls1_to_list)
            label_list.append(label_cls2_to_list)
            label_list.append(label_cls3_to_list)

            label_list_cpu = []
            label_list_cpu.append(label_cls1_to_list.cpu())
            label_list_cpu.append(label_cls2_to_list.cpu())
            label_list_cpu.append(label_cls3_to_list.cpu())

            rand_idx = random.randint(0, 5)

            # ----- Optimize three branches -----
            optim_g1.zero_grad()
            optim_g2.zero_grad()
            optim_g3.zero_grad()
            model(
                imcls, imseg, label_list, seg_onehot, ce_loss, dice_loss, rand_idx,
                optim_g1, optim_g2, optim_g3, step=1
            )
            optim_g1.step()
            optim_g2.step()
            optim_g3.step()
            # --------- Step 2: train main IGReg_DPA with uncertainty-weighted objective --------------
            optim_main.zero_grad()
            loss = model(
                imcls, imseg, label_list, seg_onehot, ce_loss, dice_loss, rand_idx,
                None, None, None, step=2
            )
            loss.backward()
            optim_main.step()
            Train_loss_record.update(loss.item(), 1)

            global_step += 1

        print('Global_Step {} | Train Epoch: {}'.format(global_step, epoch))

        train_loss_avg = Train_loss_record.avg

        # ---------------- Validation ----------------
        # Classification metrics (loss, Accuracy, F1, Precision, Recall)
        metrics_test_cls_1 = valid_metrics_by_cls(model, val_loader_cls_1p19q, ssl_index=0, cls_name=cls_name_final[0])
        metrics_test_cls_2 = valid_metrics_by_cls(model, val_loader_cls_IDH,   ssl_index=1, cls_name=cls_name_final[1])
        metrics_test_cls_3 = valid_metrics_by_cls(model, val_loader_cls_HL,    ssl_index=2, cls_name=cls_name_final[2])

        test_seg_loss, diceDis = validseg(test_loader_seg, model)

        loss_Test_1p19q = metrics_test_cls_1[0]
        loss_Test_IDH   = metrics_test_cls_2[0]
        loss_Test_HL    = metrics_test_cls_3[0]

        F1_Test_1p19q = metrics_test_cls_1[2]
        F1_Test_IDH   = metrics_test_cls_2[2]
        F1_Test_HL    = metrics_test_cls_3[2]

        Mean_TeCls = (F1_Test_1p19q + F1_Test_IDH + F1_Test_HL) / 3

        # ---------------- Logging to CSV ----------------
        tmp = pd.Series([
            epoch,
            train_loss_avg,
            loss_Test_1p19q, loss_Test_IDH, loss_Test_HL,
            F1_Test_1p19q, F1_Test_IDH, F1_Test_HL,
            test_seg_loss, diceDis,
            Mean_TeCls
        ], index=[
            'epoch',
            'Train_loss',
            'Test_loss_1', 'Test_loss_2', 'Test_loss_3',
            'Test_F1_1', 'Test_F1_2', 'Test_F1_3',
            'Test_seg_loss', 'Test_Dice',
            'Mean_TeCls'
        ])


        log = log.append(tmp, ignore_index=True)
        log.to_csv(path_save + 'log.csv', index=False)

        # ---------------- Checkpointing ----------------
        F1 = (F1_Test_HL + F1_Test_IDH + F1_Test_1p19q) / 3
        compare = diceDis + F1
        if epoch >= config.save_epoch:
            if compare > best_score_save:
                save_config = {
                    'name': save_name,
                    'save_dir': path_save,
                    'global_step': global_step,
                    'Test_F1': F1,
                    'F1_Test_HL': F1_Test_HL,
                    'F1_Test_IDH': F1_Test_IDH,
                    'F1_Test_1p19q': F1_Test_1p19q,
                    'dice1': diceDis,
                }
                save_model(model=model, config=save_config)
                best_score_save = compare

        if compare > best_score_save:
            best_score_save = compare

        # ---------------- One-time segmentation cluster init ----------------
        if model.SegDCFun.MAUCloss.init == False:
            with torch.no_grad():
                start_time = time.time()
                print("Initializing........")
                iter_init_loader_seg = iter(init_loader_seg)
                data_init_seg = next(iter_init_loader_seg)
                imgs_init_seg, seg_init_mask = data_init_seg
                imgs_init_seg = to_device(imgs_init_seg, gpu=config.use_cuda)
                seg_init_mask = to_device(seg_init_mask, gpu=config.use_cuda)
                seg_init_mask = get_onehot_fall(seg_init_mask[:, 0, :])
                model.IGReg_DPA.init_cluster_seg(imgs_init_seg, seg_init_mask)
                print("Initializing end")
                end_time = time.time()
                print(f"Initialization time: {end_time - start_time:.2f} seconds")


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    """
    Reproducible setup + experiment selection.
    """
    Cross_Seg = 1
    warnings.filterwarnings("ignore")

    # ---------------- Determinism / Seeds ----------------
    seed = config.random_seed
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # for hash randomization
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPUs

    torch.backends.cudnn.benchmark = False  # deterministic == True requires benchmark False
    torch.backends.cudnn.deterministic = True

    print("Available GPUs:", torch.cuda.device_count())

    # ---------------- Experiment setup ----------------
    init_seg = 10  # number of samples for segmentation cluster initialization

    ratio_list_all = [
        [1, 1, 1],
        [1, 1, 5],
        [1, 1, 10],
        [1, 5, 1],
        [1, 10, 1],
        [5, 1, 1],
        [10, 1, 1]
    ]

    ratio_index = 6  # choose scenario index
    model = IGReg(n_classes=2, num_channels=24, dim_proj=64, nc_pos=6, nc_neg=6, m=0.5)

    save_name = model.name + '_' + str(ratio_list_all[ratio_index])
    print(save_name)

    # ---------------- Run training ----------------
    train(model, init_seg, ratio_list_all[ratio_index], Cross_Seg=Cross_Seg)
