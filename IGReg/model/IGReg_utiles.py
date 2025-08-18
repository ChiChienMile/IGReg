from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
#  run from IGReg.model.IGReg.py
# from kmeans_pytorch import kmeans
# from Backbone.IGReg import Backbone_IGReg

#  run from IGReg.Train_IGReg.py
from model.kmeans_pytorch import kmeans
from model.Backbone.IGReg import Backbone_IGReg

class Clusterfunction(nn.Module):
    """
    Maintains learnable cluster prototypes for the NEGATIVE (background) and POSITIVE (tumor)
    super-classes. Supports two usages:

      1) Unsupervised align:
         - For each embedding, the nearest prototype is treated as the positive.
         - All remaining prototypes are treated as negatives.

      2) Margin and het loss:
         - Given coarse labels (background vs tumor), explicitly pull features toward the correct
           super-class prototype(s) and push away from the opposite super-class.
         - Applies a margin to the ground-truth logit before CE.

    Args:
        in_features (int): Dimensionality of input feature embeddings.
        nc_pos (int): Number of POSITIVE (tumor) prototypes.
        nc_neg (int): Number of NEGATIVE (background) prototypes.
        m (float): Margin value used in supervised (directed) loss.
        T (float): Temperature for scaling logits before softmax.
        margin (bool): Whether to apply margin.
    """
    def __init__(self, in_features, nc_pos, nc_neg, m, T=0.01, margin=True):
        super(Clusterfunction, self).__init__()

        self.m = m
        self.T = T
        self.nc_pos = nc_pos
        self.nc_neg = nc_neg
        self.margin = margin

        # Prototypes layout: [negatives..., positives...] for convenience when slicing
        self.cluster = nn.Parameter(torch.Tensor(self.nc_neg + self.nc_pos, in_features))
        self.init = False
        nn.init.constant_(self.cluster, 1.0)  # temporary constant init; replaced by kmeans/means later

    @torch.no_grad()
    def _init_cluster(self, bg_feats, tumor_feats):
        """
        Initialize prototypes from feature pools (background & tumor).

        If the number of clusters per super-class is 1, use the mean feature.
        Otherwise, run k-means on CPU and then move centroids to CUDA.

        Args:
            bg_feats (Tensor): Background feature set, shape [N_b, C].
            tumor_feats (Tensor): Tumor feature set, shape [N_t, C].
        """
        with torch.no_grad():
            if not self.init:
                if self.nc_neg == 1:
                    # Use simple means when only one centroid is required per super-class
                    bg_cpu = bg_feats.to('cpu')
                    tumor_cpu = tumor_feats.to('cpu')
                    k_bg = torch.mean(bg_cpu, dim=0).unsqueeze(0)
                    k_tumor = torch.mean(tumor_cpu, dim=0).unsqueeze(0)
                    k_vector = torch.cat((k_bg, k_tumor), dim=0).cuda()
                else:
                    # K-means centroids for each super-class independently
                    _, k_bg = kmeans(X=bg_feats, num_clusters=self.nc_neg,
                                     device='cpu', tqdm_flag=False, distance='euclidean')
                    _, k_tumor = kmeans(X=tumor_feats, num_clusters=self.nc_pos,
                                        device='cpu', tqdm_flag=False, distance='euclidean')
                    k_vector = torch.cat((k_bg, k_tumor), dim=0).cuda()
                self.cluster.data = k_vector
                self.init = True

    def _get_cluster_pos_neg(self, feats):
        """
        For each embedding, find the nearest prototype (positive) and the remaining (negatives).

        Args:
            feats (Tensor): Input embeddings, shape [B, C].

        Returns:
            pos (Tensor): Nearest prototype for each sample, shape [B, C].
            neg (Tensor): Remaining prototypes, shape [B, K-1, C].
        """
        feats = F.normalize(feats, p=2, dim=1)
        cluster = F.normalize(self.cluster, p=2, dim=1)

        # Cosine similarity to each prototype
        cosine = feats @ cluster.T  # [B, K]
        _, sorted_idx = torch.topk(cosine, k=cluster.size(0), dim=1, largest=True, sorted=True)
        sorted_proto = cluster[sorted_idx.view(-1)].view(feats.size(0), cluster.size(0), -1)

        pos = sorted_proto[:, 0, :]     # top-1
        neg = sorted_proto[:, 1:, :]    # the rest
        return pos, neg

    def forward(self, embeddings, ce_loss):
        """
        Steps:
          - Normalize embeddings and prototypes.
          - For each sample, compute logits against its nearest prototype (positive)
            and all remaining prototypes (negatives).
        Args:
            embeddings (Tensor): Shape [N, C].
            ce_loss (callable): Cross-entropy loss function.
        Returns:
            loss_align (Tensor): Scalar loss.
        """
        pos, neg = self._get_cluster_pos_neg(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Similarities for positive and negative sets
        l_pos = torch.einsum('nc,nc->n', [embeddings, pos]).unsqueeze(-1)  # [N, 1]
        l_neg = torch.cat(
            [torch.einsum('nc,nc->n', [embeddings, neg[:, i, :]]).unsqueeze(-1)
             for i in range(neg.size(1))],
            dim=1
        )  # [N, K-1]

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # [N, K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # index 0 is pos
        loss_align = ce_loss(logits, labels)
        return loss_align

    def get_logits_iter(self, logits, labels, ce_loss):
        """
        Directed (supervised) margin loss computed across all (neg, pos) prototype pairs.

        For each (neg_i, pos_j) pair:
          - Form a 2-way logit vector [neg_i, pos_j]
          - Optionally subtract a margin from the ground-truth logit

        Args:
            logits (Tensor): Embedding vs. all prototypes, shape [N, K].
            labels (Tensor): Ground-truth labels in {0,1}, shape [N].
                             0 => background, 1 => tumor
            ce_loss (callable): Cross-entropy loss function.

        Returns:
            loss_margin (Tensor): Mean loss over all neg/pos prototype pairs.
            collected_preds (list[Tensor]): Each is softmax probs of shape [N, 2].
        """
        total_margin_loss = 0.0
        collected_preds = []

        for i_neg in range(self.nc_neg):
            for j_pos in range(self.nc_pos):
                # Build 2-way logits: [neg_i, pos_j]
                neg_log = logits[:, i_neg].unsqueeze(1)
                pos_log = logits[:, self.nc_neg + j_pos].unsqueeze(1)
                pair_logits = torch.cat((neg_log, pos_log), dim=1)  # [N, 2]

                if self.margin:
                    # Apply margin to the ground-truth index before temperature scaling
                    pair_margin = torch.zeros_like(pair_logits)
                    pair_margin.scatter_(1, labels.view(-1, 1), self.m)
                    pair_logits = (pair_logits - pair_margin) / self.T
                    loss_margin = ce_loss(pair_logits, labels)
                    preds = F.softmax(pair_logits, dim=1)
                else:
                    loss_margin = ce_loss(pair_logits, labels)
                    preds = F.softmax(pair_logits, dim=1)

                total_margin_loss = total_margin_loss + loss_margin
                collected_preds.append(preds)

        avg_margin_loss = total_margin_loss / (self.nc_neg * self.nc_pos)
        return avg_margin_loss, collected_preds

    def forward_directed_clustering(self, embeddings, labels, ce_loss):
        """
        Args:
            embeddings (Tensor): Shape [N, C].
            labels (Tensor): Shape [N], values in {0,1} (background, tumor).
            ce_loss (callable): Cross-entropy loss.
        Returns:
            loss_margin (Tensor): Averaged margin loss over all prototype pairs.
            preds_list (list[Tensor]): Optional list of [N,2] per-pair soft predictions.
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        cluster = F.normalize(self.cluster, p=2, dim=1)
        logits = embeddings @ cluster.T  # [N, K]
        loss_margin, preds_list = self.get_logits_iter(logits, labels, ce_loss)
        return loss_margin, preds_list


class PropCluster(nn.Module):
    """
    Wrapper applying Clusterfunction to dense 3D feature maps (segmentation features).

    Provides:
      - Unsupervised prototype contrast over randomly downsampled voxels (memory-friendly).
      - Directed unsupervised clustering (DUC): uses weak segmentation labels (background vs tumor)
        to enforce directional separation with a margin; also returns an averaged Dice loss
        across all pairwise soft predictions to align with the segmentation target.
    """
    def __init__(self, nc_pos, nc_neg, m, dim_proj=64):
        super(PropCluster, self).__init__()
        self.Clusterfunction = Clusterfunction(in_features=dim_proj, nc_pos=nc_pos, nc_neg=nc_neg, m=m)

    def get_cluster_loss(self, feat_5d, ce_loss):
        """
        Unsupervised contrastive loss on a random subset of voxels.

        Args:
            feat_5d (Tensor): Feature map [B, C, W, H, D].
            ce_loss (callable): Cross-entropy loss.

        Returns:
            loss_cluster (Tensor): Scalar contrastive loss.
        """
        B, C, W, H, D = feat_5d.size()
        # Flatten spatial dims to a voxel list: [B*WHD, C]
        vox = feat_5d.reshape(B, C, W * H * D).permute(0, 2, 1).reshape(-1, C)

        # Randomly sample 1/64 voxels to reduce computation
        idx = torch.randperm(vox.size(0))[: vox.size(0) // 64]
        vox = vox[idx]

        loss_align = self.Clusterfunction(vox, ce_loss)
        return loss_align

    def get_margin_het_loss(self, feat_5d_labeled, seg_onehot, ce_loss, dice_loss):
        """
        Directed unsupervised clustering with segmentation guidance.

        Args:
            feat_5d_labeled (Tensor): Labeled feature map [B, C, W, H, D].
            seg_onehot (Tensor): One-hot segmentation [B, 2, W, H, D].
                                 channel-0: background, channel-1: tumor
            ce_loss (callable): CE loss for the directed objective.
            dice_loss (callable): Dice loss to measure agreement with segmentation.

        Returns:
            avg_margin_loss (Tensor): Directed margin loss (averaged over prototype pairs).
            loss_seg_avg (Tensor or float): Averaged Dice loss over pairwise soft predictions.
        """
        B, C, W, H, D = feat_5d_labeled.size()
        vox = feat_5d_labeled.reshape(B, C, W * H * D).permute(0, 2, 1).reshape(-1, C)

        # Convert one-hot to hard labels: [B*WHD]
        hard_labels = torch.argmax(seg_onehot, dim=1).reshape(B * W * H * D)

        avg_margin_loss, pair_pred_list = self.Clusterfunction.forward_directed_clustering(vox, hard_labels, ce_loss)

        # For each (neg,pos) pair, get predicted [bg,tumor] probs per voxel and compute Dice
        loss_seg_total = 0.0
        for pair_pred in pair_pred_list:
            pair_pred_5d = pair_pred.reshape(B, W, H, D, 2).permute(0, 4, 1, 2, 3)  # [B,2,W,H,D]
            loss_seg_total += dice_loss(pair_pred_5d, seg_onehot)

        loss_seg_avg = loss_seg_total / len(pair_pred_list) if len(pair_pred_list) > 0 else 0.0
        return avg_margin_loss, loss_seg_avg

    def forward(self, feat_5d_pair, seg_onehot, ce_loss, dice_loss):
        """
        Combine unsupervised contrast (on a pair batch) with segmentation-guided directed loss.

        Args:
            feat_5d_pair (Tensor): Two volumes stacked along batch dim [2, C, W, H, D] [unlabel+label, C, W, H, D].
            seg_onehot (Tensor): One-hot segmentation [B, 2, W, H, D].
            ce_loss, dice_loss: Loss functions.

        Returns:
            loss_margin (Tensor): Directed margin loss on labeled features.
            loss_het (Tensor/float): Averaged Dice loss across pair predictions.
            loss_align (Tensor): Unsupervised contrastive alignment loss on the pair.
        """
        loss_align = self.get_cluster_loss(feat_5d_pair, ce_loss)
        # feat_5d_pair[1, :] have seg label
        loss_margin, loss_het = self.get_margin_het_loss(feat_5d_pair[1, :].unsqueeze(0), seg_onehot, ce_loss, dice_loss)
        return loss_margin, loss_het, loss_align

    def _init_cluster_seg(self, feat_5d_labeled, seg_onehot):
        """
        Initialize prototypes from labeled segmentation features (background vs tumor).

        Steps:
          - Flatten labeled features into voxel list.
          - Use one-hot to split voxels into positive (label>0) and negative (label==0).
          - Downsample negatives to match the number of positives for balanced k-means.
        """
        if self.Clusterfunction.init is False:
            hard = torch.argmax(seg_onehot, dim=1)
            B, C, W, H, D = feat_5d_labeled.size()
            vox = feat_5d_labeled.reshape(B, C, W * H * D).permute(0, 2, 1).reshape(-1, C)
            hard = hard.reshape(B * W * H * D)

            pos_mask = hard > 0
            neg_mask = hard == 0
            pos_vox = vox[pos_mask]
            neg_vox = vox[neg_mask]

            # Balance counts by sampling negatives
            idx = torch.randperm(neg_vox.size(0))[: pos_vox.size(0)]
            neg_vox = neg_vox[idx]
            self.Clusterfunction._init_cluster(neg_vox, pos_vox)


class IGReg_DPA(nn.Module):
    """
    Responsibilities:
      - Extract multi-task features via Backbone_IGReg.
      - Project segmentation features and feed into clustering regularizers (unsupervised + directed).
      - Return task losses (classification/seg) and clustering regularizers.
    Args:
        n_classes (int): Number of classes per classification task (binary -> 2).
        num_channels (int): Feature channel width in the backbone.
        dim_proj (int): Channel dim used for 1x1 projection before clustering.
        nc_pos/nc_neg (int): Number of prototypes per super-class (tumor/background).
        m (float): Margin for directed clustering.
    """
    def __init__(self, n_classes=2, num_channels=24, dim_proj=64, nc_pos=3, nc_neg=3, m=0.5):
        super(IGReg_DPA, self).__init__()
        self.name = 'IGReg_DPA' + '_np_' + str(nc_pos) + '_nn_' + str(nc_neg) + '_m_' + str(m)

        self.BranchMain = Backbone_IGReg.Backbone(n_classes=n_classes, num_channels=num_channels)
        self.PropCluster = PropCluster(dim_proj=dim_proj, nc_pos=nc_pos, nc_neg=nc_neg, m=m)
        self.proj = nn.Conv3d(num_channels, dim_proj, kernel_size=1)

    def forward(self, cls_inputs, seg_inputs, label_list, seg_onehot, ce_loss, dice_loss, rand_idx):
        """
        Args:
            cls_inputs (Tensor): Classification images (stacked sub-batches), shape [B_cls, C, W, H, D].
            seg_inputs (Tensor): Segmentation branch images, shape [B_seg, C, W, H, D].
            label_list (Tensor): Task labels list/stack (shape like [3, 2] for three tasks, two samples each).
            seg_onehot (Tensor): One-hot segmentation [B_seg, 2, W, H, D].
            ce_loss, dice_loss: Loss functions.
            rand_idx (int): Index to select a sample from seg_proj to pair with the last one for contrast.

        Returns:
            loss_cls, loss_seg (Tensor): From the backbone.
            loss_margin, loss_het, loss_align (Tensor/float): Clustering regularizers.
            seg_feats, cls_feats (Tensor): Raw features from the backbone for potential match losses.
        """
        loss_cls, loss_seg, seg_feats, cls_feats = self.BranchMain(
            cls_inputs, seg_inputs, label_list, seg_onehot, ce_loss, dice_loss
        )

        # 1x1 projection to dim_proj for clustering stability
        seg_proj = self.proj(seg_feats)

        B = cls_inputs.size(0)
        # Build a two-sample mini-batch for clustering regularization
        seg_pair = torch.cat((seg_proj[rand_idx, :].unsqueeze(0), seg_proj[B, :].unsqueeze(0)), dim=0)
        seg_pair = F.interpolate(seg_pair, size=[160, 192, 160], mode='trilinear', align_corners=True)

        loss_margin, loss_het, loss_align = self.PropCluster(seg_pair, seg_onehot, ce_loss, dice_loss)
        return loss_cls, loss_seg, loss_margin, loss_het, loss_align, seg_feats, cls_feats

    def predictcls(self, x, cls_index):
        """Forward a sub-batch to the classification head of the requested task."""
        with torch.no_grad():
            class_logits = self.BranchMain.predictcls(x, cls_index)
            return class_logits


    def init_cluster_seg(self, seg_imgs, seg_onehot):
        """
        Initialize clustering prototypes from segmentation images and masks.

        Steps:
          - Get segmentation features from the backbone.
          - Project to 'dim_proj' channels and upsample to a canonical spatial size.
          - Use PropCluster._init_cluster_seg to compute initial centroids.
        """
        with torch.no_grad():
            if self.PropCluster.Clusterfunction.init is False:
                seg_feats = self.BranchMain.get_features(seg_imgs)
                proj = self.proj(seg_feats)
                proj = F.interpolate(proj, size=[160, 192, 160], mode='trilinear', align_corners=True)
                _ = self.PropCluster._init_cluster_seg(proj, seg_onehot)
                return _
