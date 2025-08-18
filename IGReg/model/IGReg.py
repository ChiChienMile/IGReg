from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

#  run from IGReg.model.IGReg.py
# from IGReg_utiles import IGReg_DPA
# from loss_utiles import SoftDiceLoss
# from Backbone.IGReg import Backbone_IGReg

#  run from IGReg.Train_IGReg.py
from model.IGReg_utiles import IGReg_DPA
from model.loss_utiles import SoftDiceLoss
from model.Backbone.IGReg import Backbone_IGReg

class IGReg(nn.Module):
    """
    Two-stage training scheme:

      Step = 1:
        - Freeze IGReg_DPA (main branch).
        - Train three auxiliary classifiers (G1/G2/G3), each centered on a task (1p19q / IDH / LHG).
        - Use gradient projection to reduce cross-task conflicts among their losses.

      Step > 1:
        - Freeze G1/G2/G3.
        - Train IGReg_DPA with an uncertainty-weighted sum of:
            * classification loss
            * segmentation loss
            * directed clustering margin loss
            * Dice loss of pairwise predictions (heterogeneity)
            * unsupervised contrastive alignment loss
            * feature-matching regularizer (only for samples gated to update)

    It also provides 'copy_param' to initialize auxiliary classifiers from the backbone.
    """
    def __init__(self, n_classes=2, num_channels=24, dim_proj=64, nc_pos=3, nc_neg=3, m=0.5, distance='l2'):
        super(IGReg, self).__init__()
        self.name = ('IGReg' + '_np_' + str(nc_pos) + '_nn_' + str(nc_neg) + '_m_' + str(m))

        self.IGReg_DPA = IGReg_DPA(n_classes=n_classes, num_channels=num_channels,
                                   dim_proj=dim_proj, nc_pos=nc_pos, nc_neg=nc_neg, m=m)
        self.G1 = Backbone_IGReg.AuxClassifer(num_channels=num_channels, n_classes=n_classes)  # 1p19q-centered
        self.G2 = Backbone_IGReg.AuxClassifer(num_channels=num_channels, n_classes=n_classes)  # IDH-centered
        self.G3 = Backbone_IGReg.AuxClassifer(num_channels=num_channels, n_classes=n_classes)  # LHG-centered

        # Learnable log-variances for multi-loss uncertainty weighting
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma4 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma5 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma6 = nn.Parameter(torch.tensor(0.0))

        # Distance function selector for the feature-matching regularizer
        self.distance = distance

    def _set_requires_grad(self, nets, requires_grad=False):
        """Enable/disable gradients for a (list of) module(s)."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for p in net.parameters():
                    p.requires_grad = requires_grad

    def copy_param(self, model_dst, model_src):
        """
        Partially copy backbone parameters into an auxiliary classifier (G1/G2/G3).
        The modules listed here must exist in both 'model_src' and 'model_dst'.
        """
        model_dst.bivablock1.load_state_dict(model_src.bivablock1.state_dict())
        model_dst.fusion0.load_state_dict(model_src.fusion0.state_dict())
        model_dst.ASA0.load_state_dict(model_src.ASA0.state_dict())
        model_dst.ASA1.load_state_dict(model_src.ASA1.state_dict())
        model_dst.ASA2.load_state_dict(model_src.ASA2.state_dict())
        model_dst.ASA3.load_state_dict(model_src.ASA3.state_dict())

        model_dst.pool0.load_state_dict(model_src.pool0.state_dict())
        model_dst.attention2.load_state_dict(model_src.attention2.state_dict())
        model_dst.conv2.load_state_dict(model_src.conv2.state_dict())
        model_dst.pool1.load_state_dict(model_src.pool1.state_dict())
        model_dst.attention3.load_state_dict(model_src.attention3.state_dict())
        model_dst.conv3.load_state_dict(model_src.conv3.state_dict())

    def forward(self, cls_inputs, seg_inputs, label_list, seg_onehot, ce_loss, dice_loss,
                rand_idx, optim_g1, optim_g2, optim_g3, step):
        """
        Dispatch to two modes depending on 'step'.

        Args:
            cls_inputs (Tensor): Classification inputs [B_cls, C, W, H, D].
            seg_inputs (Tensor): Segmentation inputs [B_seg, C, W, H, D].
            label_list (Tensor): Labels per task (e.g., shape [3, 2] for three tasks).
            seg_onehot (Tensor): One-hot seg [B_seg, 2, W, H, D].
            ce_loss, dice_loss (callable): Loss functions.
            rand_idx (int): Random index to pick one seg projection for pair construction.
            optim_g1/g2/g3: Optimizers for auxiliary classifiers (used only when step==1).
            step (int): 1 -> train auxiliary classifiers; else -> train IGReg_DPA.

        Returns:
            When step > 1: loss_guide (Tensor), the final scalar to backprop.
        """
        if step == 1:
            # ============== Train auxiliary classifiers only ==============
            self._set_requires_grad(self.IGReg_DPA, False)
            self._set_requires_grad(self.G1, True)
            self._set_requires_grad(self.G2, True)
            self._set_requires_grad(self.G3, True)

            with torch.no_grad():
                # Extract multi-scale features once for all three auxiliary heads
                x_cat = torch.cat((cls_inputs, seg_inputs), dim=0)
                layer1, layer2, layer3, layer4 = self.IGReg_DPA.BranchMain.get_m_features(x_cat)

            # G1 (1p19q-centered): order losses as [task0, task1, task2]
            loss_t0, loss_t1, loss_t2 = self.G1(layer1, layer2, layer3, layer4, label_list, ce_loss)
            guided_grads = self._task_guide_gradients_unconflict([loss_t0, loss_t1, loss_t2], optim_g1)
            optim_g1.zero_grad()
            self._apply_gradients(optim_g1, guided_grads)

            # G2 (IDH-centered): adjust ordering so that the IDH loss has precedence
            loss_t0, loss_t1, loss_t2 = self.G2(layer1, layer2, layer3, layer4, label_list, ce_loss)
            guided_grads = self._task_guide_gradients_unconflict([loss_t1, loss_t0, loss_t2], optim_g2)
            optim_g2.zero_grad()
            self._apply_gradients(optim_g2, guided_grads)

            # G3 (LHG-centered): similarly prioritize the LHG loss
            loss_t0, loss_t1, loss_t2 = self.G3(layer1, layer2, layer3, layer4, label_list, ce_loss)
            guided_grads = self._task_guide_gradients_unconflict([loss_t2, loss_t0, loss_t1], optim_g3)
            optim_g3.zero_grad()
            self._apply_gradients(optim_g3, guided_grads)

        else:
            # ============== Train IGReg_DPA only (uncertainty-weighted objective) ==============
            self._set_requires_grad(self.G1, False)
            self._set_requires_grad(self.G2, False)
            self._set_requires_grad(self.G3, False)
            self._set_requires_grad(self.IGReg_DPA, True)

            with torch.no_grad():
                # Extract multi-scale features once for probability gating & aux feature snapshots
                x_cat = torch.cat((cls_inputs, seg_inputs), dim=0)
                layer1, layer2, layer3, layer4 = self.IGReg_DPA.BranchMain.get_m_features(x_cat)

                # Auxiliary classifier features used for feature-matching when gating triggers
                feats_g1 = self.G1.get_features(layer1, layer2, layer3, layer4)
                feats_g2 = self.G2.get_features(layer1, layer2, layer3, layer4)
                feats_g3 = self.G3.get_features(layer1, layer2, layer3, layer4)

                # Build update states for each sub-batch across tasks.
                # State definition:
                #   2 -> move current features toward the corrected aux features
                #   0 -> no update
                state_list = []
                for i in range(2):
                    state = self.get_prob(cls_inputs, layer1, layer2, layer3, layer4, label_list, 0, i)
                    state_list.append(state)
                for i in range(2, 4):
                    state = self.get_prob(cls_inputs, layer1, layer2, layer3, layer4, label_list, 1, i - 2)
                    state_list.append(state)
                for i in range(4, 6):
                    state = self.get_prob(cls_inputs, layer1, layer2, layer3, layer4, label_list, 2, i - 4)
                    state_list.append(state)

            # Main branch forward to obtain losses and features
            loss_cls, loss_seg, loss_margin, loss_het, loss_align, seg_feats, cls_feats = self.IGReg_DPA(
                cls_inputs, seg_inputs, label_list, seg_onehot, ce_loss, dice_loss, rand_idx
            )

            # Feature-matching regularizer (only for those samples with state==2)
            mse_list, count = [], 0
            for idx, state in enumerate(state_list):
                if idx < 2:  # 1p19q sub-batches
                    if state == 2:
                        mse_list.append(self._feature_distance(feats_g1[idx, :], cls_feats[idx, :]))
                        count += 1
                elif idx < 4:  # IDH sub-batches
                    if state == 2:
                        mse_list.append(self._feature_distance(feats_g2[idx, :], cls_feats[idx, :]))
                        count += 1
                else:  # LHG sub-batches
                    if state == 2:
                        mse_list.append(self._feature_distance(feats_g3[idx, :], cls_feats[idx, :]))
                        count += 1
            loss_reg = torch.mean(torch.stack(mse_list)) if count > 0 else torch.tensor(0.0, device=cls_feats.device)

            # Uncertainty-weighted total objective (Kendall & Gal style)
            loss_guide = (
                torch.exp(-self.log_sigma1) * loss_cls          + self.log_sigma1 +
                torch.exp(-self.log_sigma2) * loss_seg          + self.log_sigma2 +
                torch.exp(-self.log_sigma3) * loss_margin       + self.log_sigma3 +
                torch.exp(-self.log_sigma4) * loss_het          + self.log_sigma4 +
                torch.exp(-self.log_sigma5) * loss_align        + self.log_sigma5 +
                torch.exp(-self.log_sigma6) * loss_reg          + self.log_sigma6
            )
            return loss_guide

    def get_prob(self, cls_inputs, layer1, layer2, layer3, layer4, label_list, cls_index, i_sample):
        """
        Decide whether to move current features toward auxiliary corrected ones.

        Compare:
          - prob_g: predicted probability from the gradient-corrected auxiliary classifier (G1/G2/G3)
          - prob_uncorr: probability from the current (uncorrected) main branch

        Rule:
          if (prob_uncorr < prob_g) and (prob_g > 0.5):
              return 2  # update toward corrected features
          else:
              return 0  # no update
        """
        with torch.no_grad():
            if cls_index == 0:
                logits_g = self.G1.predictcls(layer1, layer2, layer3, layer4, cls_index)[i_sample, :]
            elif cls_index == 1:
                logits_g = self.G2.predictcls(layer1, layer2, layer3, layer4, cls_index)[i_sample, :]
            else:
                logits_g = self.G3.predictcls(layer1, layer2, layer3, layer4, cls_index)[i_sample, :]

            prob_g = F.softmax(logits_g, dim=0)[label_list[cls_index][i_sample]].item()

            logits_uncorr = self.predictcls(cls_inputs, cls_index)[0]
            prob_uncorr = F.softmax(logits_uncorr, dim=0)[label_list[cls_index][i_sample]].item()

            if prob_uncorr < prob_g and prob_g > 0.5:
                return 2
            else:
                return 0

    def _feature_distance(self, vec1, vec2):
        """
        Compute distance between two feature vectors according to `self.distance`.

        Supported options:
          - 'mse'     : mean squared error (squared L2 mean)
          - 'cos'/'cosine': 1 - cosine similarity
          - 'l2'/'euclidean': Euclidean norm ||vec1 - vec2||_2
          - callable  : a custom function distance(vec1, vec2) -> scalar tensor

        Returns:
          Tensor scalar with the chosen distance.
        """
        if isinstance(self.distance, str):
            name = self.distance.lower()
            if name == 'mse':
                return F.mse_loss(vec1, vec2)
            elif name in ('cos', 'cosine'):
                return self.cosine(vec1, vec2)
            elif name in ('l2', 'euclidean'):
                return torch.norm(vec1 - vec2, p=2)
            else:
                raise ValueError(f"Unsupported distance: {self.distance}")
        elif callable(self.distance):
            return self.distance(vec1, vec2)
        else:
            raise ValueError("distance must be 'mse' | 'cosine' | 'l2' | callable(vec1, vec2)->scalar")

    def _apply_gradients(self, optim, merged_grads):
        """
        Write merged gradient tensors back into optimizer param.grad buffers.

        If a parameter has an existing grad, we add; otherwise we assign a clone.
        """
        idx = 0
        for group in optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if merged_grads[idx] is not None:
                        p.grad += merged_grads[idx]
                else:
                    if merged_grads[idx] is not None:
                        p.grad = merged_grads[idx].clone()
                idx += 1

    def _get_grad(self, optim):
        """
        Collect current gradients (or zeros if None) and remember shapes.

        Returns:
            grads (list[Tensor]): Per-parameter gradient tensors (or zeros).
            shapes (list[torch.Size]): Shapes for unflattening later.
        """
        grads, shapes = [], []
        for group in optim.param_groups:
            for p in group['params']:
                shapes.append(p.shape)
                if p.grad is None:
                    grads.append(torch.zeros_like(p, device=p.device))
                else:
                    grads.append(p.grad.clone())
        return grads, shapes

    def _grad_proj(self, grad_flat, base_grad_flat):
        """
        Project grad_flat onto base_grad_flat; compute cosine similarity of the projection.

        projection = ((g·b)/(b·b)) * b
        cos = (projection·b) / (||projection|| * ||b||)

        Returns:
            projection (Tensor): The projected vector.
            cos_sim (float): Cosine similarity between projection and base vector in [-1,1].
        """
        if torch.norm(base_grad_flat) == 0:
            return torch.zeros_like(base_grad_flat), 0.0

        scale = torch.dot(grad_flat, base_grad_flat) / torch.dot(base_grad_flat, base_grad_flat)
        projection = scale * base_grad_flat

        if torch.norm(projection) == 0:
            return projection, 0.0

        cos_sim = torch.dot(projection, base_grad_flat) / (torch.norm(projection) * torch.norm(base_grad_flat))
        return projection, cos_sim

    def _flatten_grad(self, grads):
        """Flatten and concatenate a list of gradient tensors into a single vector."""
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, flat, shapes):
        """Restore a flattened gradient vector back to a list of tensors with given shapes."""
        unflat, idx = [], 0
        for shp in shapes:
            length = int(np.prod(shp))
            unflat.append(flat[idx:idx + length].view(shp))
            idx += len(flat[idx:idx + length])
        return unflat

    def _task_guide_gradients_unconflict(self, loss_list, optim):
        """
        Compute per-loss gradients, project later ones to reduce conflict w.r.t. the first,
        then sum them and return as a list of per-parameter gradient tensors.

        Procedure:
          1) For each loss in 'loss_list', backprop to collect gradient vector (flattened).
          2) For i>=1, project grad_i onto grad_0; if cosine < 0, remove the conflicting component.
          3) Sum all (possibly adjusted) grads, unflatten back to param-shaped tensors.
        """
        num_tasks = len(loss_list)
        flat_grads, shapes_record = [], []

        for loss in loss_list:
            optim.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grads, shapes = self._get_grad(optim)
            flat_grads.append(self._flatten_grad(grads))
            shapes_record.append(shapes)

        # Project conflicts against the first gradient
        for i in range(1, num_tasks):
            proj, cos = self._grad_proj(flat_grads[i], flat_grads[0])
            if cos < 0:
                flat_grads[i] -= proj

        merged = torch.stack(flat_grads).sum(dim=0)
        merged = self._unflatten_grad(merged, shapes_record[0])
        return merged

    def predictcls(self, x, cls_index):
        """Prediction helper exposing the main branch classification heads."""
        with torch.no_grad():
            return self.IGReg_DPA.predictcls(x, cls_index)

    def predictseg(self, x):
        """Prediction segmentation."""
        with torch.no_grad():
            return self.IGReg_DPA.BranchMain.predictseg(x)



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    import random
    from torch.nn import CrossEntropyLoss
    import time
    from lion_pytorch import Lion

    lr_self = 1e-4
    weight_decay = 2e-5

    # Sub-batches: 0-1 for 1p19q, 2-3 for IDH, 4-5 for LHG
    imcls_1p19q = torch.randn(2, 2, 160, 192, 160).cuda()
    imcls_IDH   = torch.randn(2, 2, 160, 192, 160).cuda()
    imcls_LHG   = torch.randn(2, 2, 160, 192, 160).cuda()
    imcls = torch.cat((imcls_1p19q, imcls_IDH, imcls_LHG), dim=0).cuda()

    imseg = torch.randn(1, 2, 160, 192, 160).cuda()

    # Build one-hot seg mask: (B=1, C=2, W,H,D)
    hard_seg = torch.randint(0, 2, size=(160, 192, 160))
    seg_onehot = torch.zeros(2, 160, 192, 160)
    seg_onehot.scatter_(0, hard_seg.unsqueeze(0), 1)
    seg_onehot = seg_onehot.unsqueeze(0).expand(1, -1, -1, -1, -1).cuda() # (1,2,160,192,160)

    # Task labels: shape (3, 2)
    labels_1p19q = torch.randint(0, 2, size=(2,)).cuda()
    labels_IDH   = torch.randint(0, 2, size=(2,)).cuda()
    labels_LHG   = torch.randint(0, 2, size=(2,)).cuda()
    label_list = torch.stack([labels_1p19q, labels_IDH, labels_LHG], dim=0)

    model = IGReg(n_classes=2, num_channels=24, dim_proj=64, nc_pos=6, nc_neg=6, m=0.5).cuda()

    ce_loss  = CrossEntropyLoss(reduction='mean').cuda()
    dice_loss = SoftDiceLoss(smooth=1e-5, do_bg=False).cuda()
    rand_idx = random.randint(0, imcls.size(0))

    # Optimizers
    optim_main = Lion(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config.lr_self, weight_decay=config.weight_decay)
    optim_g1 = Lion(filter(lambda p: p.requires_grad, model.G1.parameters()),
                                lr=lr_self, weight_decay=weight_decay)
    optim_g2 = Lion(filter(lambda p: p.requires_grad, model.G2.parameters()),
                                lr=lr_self, weight_decay=weight_decay)
    optim_g3 = Lion(filter(lambda p: p.requires_grad, model.G3.parameters()),
                                lr=lr_self, weight_decay=weight_decay)

    # Copy pretrained backbone parts into G1/G2/G3
    model.copy_param(model.G1, model.IGReg_DPA.BranchMain)
    model.copy_param(model.G2, model.IGReg_DPA.BranchMain)
    model.copy_param(model.G3, model.IGReg_DPA.BranchMain)

    # --------- Step 1: train auxiliary classifiers with gradient conflict mitigation ---------
    optim_g1.zero_grad()
    optim_g2.zero_grad()
    optim_g3.zero_grad()
    model(imcls, imseg, label_list, seg_onehot, ce_loss, dice_loss, rand_idx,
          optim_g1, optim_g2, optim_g3, step=1)
    optim_g1.step()
    optim_g2.step()
    optim_g3.step()

    # --------- Step 2: train main IGReg_DPA with uncertainty-weighted objective --------------
    optim_main.zero_grad()
    loss = model(imcls, imseg, label_list, seg_onehot, ce_loss, dice_loss, rand_idx,
                 None, None, None, step=2)
    loss.backward()
    optim_main.step()

    # Inference on sub-batches
    logits_1p19q = model.predictcls(imcls_1p19q, 0)
    logits_IDH   = model.predictcls(imcls_IDH,   1)
    logits_LHG   = model.predictcls(imcls_LHG,   2)
    print(logits_1p19q.size(), logits_IDH.size(), logits_LHG.size())

    # After the first epoch, initialize clustering prototypes from segmentation aux data
    if model.IGReg_DPA.PropCluster.Clusterfunction.init is False:
        with torch.no_grad():
            t0 = time.time()
            print("Initializing........")
            model.IGReg_DPA.init_cluster_seg(imseg, seg_onehot)
            print("Initializing end")
            print(f"Initialization time: {time.time() - t0:.2f} seconds")