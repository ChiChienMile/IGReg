from torch import nn
import torch
import torch.nn.functional as F

#  run from IGReg.model.IGReg.py
# from Backbone.IGReg.BiVAblock import BiVA
# from Backbone.IGReg.SENet import SENet3D
# from Backbone.IGReg.SENet import SENetBottleneck

#  run from IGReg.Train_IGReg.py
from model.Backbone.IGReg.BiVAblock import BiVA
from model.Backbone.IGReg.SENet import SENet3D
from model.Backbone.IGReg.SENet import SENetBottleneck

class BackBone3D(nn.Module):
    def __init__(self):
        super(BackBone3D, self).__init__()
        net = SENet3D(SENetBottleneck, [3, 4, 6, 3], num_classes=2)
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        # and if we use the resnet3d-101, change the block list with [3, 4, 23, 3]
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        # the layer0 contains the first convolution, bn and relu
        self.layer1 = nn.Sequential(*net[3:5])
        # the layer1 contains the first pooling and the first 3 bottle blocks
        self.layer2 = net[5]
        # the layer2 contains the second 4 bottle blocks
        self.layer3 = net[6]
        # the layer3 contains the media bottle blocks
        # with 6 in 50-layers and 23 in 101-layers
        self.layer4 = net[7]
        # the layer4 contains the final 3 bottle blocks
        # according the backbone the next is avg-pooling and dense with num classes uints
        # but we don't use the final two layers in backbone networks

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


def add_conv3D(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('group_norm', nn.GroupNorm(8, out_ch))
    stage.add_module('leaky', nn.PReLU())
    return stage


class ASA3D(nn.Module):
    def __init__(self, num_channels, level, vis=False):
        super(ASA3D, self).__init__()
        self.level = level
        compress_c = 16
        self.attention = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )
        self.pool = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.weight_level_0 = add_conv3D(num_channels, compress_c, 1, 1)
        self.weight_level_1 = add_conv3D(num_channels, compress_c, 1, 1)

        self.weight_levels = nn.Conv3d(compress_c*2, 2, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, inputs0, inputs1):
        if self.level == 0:
            level_f = inputs1
        elif self.level == 1:
            level_f = self.pool(inputs1)
        elif self.level == 2:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)
        elif self.level == 3:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)

        level_0_weight_v = self.weight_level_0(inputs0)
        level_1_weight_v = self.weight_level_1(level_f)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        adaptive_attention = self.attention(inputs0) * inputs0 * levels_weight[:, 0:1, :, :, :] + \
                            self.attention(level_f) * level_f * levels_weight[:, 1:, :, :, :]

        out = self.refine(torch.cat((inputs0, adaptive_attention*level_f), 1))
        if self.vis:
            return out, levels_weight, adaptive_attention.sum(dim=1)
        else:
            return out

class Backbone(nn.Module):
    def __init__(self, n_classes, num_channels, vis=False):
        super(Backbone, self).__init__()
        self.name = 'Backbone'
        self.backbone = BackBone3D()

        self.bivablock1 = BiVA(num_channels=num_channels, first_time=True)

        self.ASA0 = ASA3D(num_channels=num_channels, level=0, vis=vis)
        self.ASA1 = ASA3D(num_channels=num_channels, level=1, vis=vis)
        self.ASA2 = ASA3D(num_channels=num_channels, level=2, vis=vis)
        self.ASA3 = ASA3D(num_channels=num_channels, level=3, vis=vis)

        self.fusion0 = nn.Sequential(
            nn.Conv3d(num_channels * 4, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )

        self.fusion1 = nn.Sequential(
            nn.Conv3d(num_channels * 4, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )

        ### segmentaion branch
        self.attention0 = nn.Sequential(
            nn.Conv3d(num_channels * 3, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv0 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )

        self.attention1 = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )
        self.conv1_up = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )
        self.predict_fuse0 = nn.Conv3d(num_channels, 2, kernel_size=1)
        self.predict_fuse1 = nn.Conv3d(num_channels, 2, kernel_size=1)
        self.predict = nn.Conv3d(num_channels, 2, kernel_size=1)

        ### classification branch

        self.pool0 = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.attention2 = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )
        self.pool1 = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.attention3 = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1), nn.GroupNorm(num_channels//2, num_channels), nn.PReLU()
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.CLSmodel = nn.ModuleList([])
        for i in range(3):
            self.CLSmodel.append(nn.Linear(num_channels*2, n_classes))

    # 二分类 需返回真实分类标签计算loss
    def class_glioma(self, features_all, label_list):
        size_img = label_list[0].size(0)
        out_cls_labels = []
        out_cls_list = []
        for ssl_index in range(0, 3):
            features_cls = features_all[size_img * ssl_index:size_img + size_img * ssl_index, :]
            # 取glioma分类的标签
            features_cls_labels = label_list[ssl_index]
            out_cls = self.CLSmodel[ssl_index](features_cls)
            out_cls_list.append(out_cls)
            out_cls_labels.append(features_cls_labels)
        return out_cls_list, out_cls_labels

    def f_theta_1(self, input):
        layer0 = self.backbone.layer0(input)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        Scale1V, Scale2V, Scale3V, Scale4V = self.bivablock1(layer1, layer2, layer3, layer4)

        F20_upsample = F.upsample(Scale2V, size=layer1.size()[2:], mode='trilinear')
        F30_upsample = F.upsample(Scale3V, size=layer1.size()[2:], mode='trilinear')
        F40_upsample = F.upsample(Scale4V, size=layer1.size()[2:], mode='trilinear')

        fuse0 = self.fusion0(torch.cat((F40_upsample, F30_upsample, F20_upsample, Scale1V), 1))
        #
        Scale1A = self.ASA0(Scale1V, fuse0)
        Scale2A = self.ASA1(Scale2V, fuse0)
        Scale3A = self.ASA2(Scale3V, fuse0)
        Scale4A = self.ASA3(Scale4V, fuse0)

        F2_upsample = F.upsample(Scale2A, size=layer1.size()[2:], mode='trilinear')
        F3_upsample = F.upsample(Scale3A, size=layer1.size()[2:], mode='trilinear')
        F4_upsample = F.upsample(Scale4A, size=layer1.size()[2:], mode='trilinear')

        fuse1 = self.fusion1(torch.cat((F4_upsample, F3_upsample, F2_upsample, Scale1A), 1))

        return Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1

    # 提取分类特征用于多任务分类
    def f_theta_2(self, Scale1A, Scale2A, Scale3A, Scale4A, label_list, loss_func):
        ### classificication branch
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(out_F20) * Scale2A)
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(out_F30) * Scale3A)

        out_F40 = torch.cat((out_F31, Scale4A), 1)

        class_predict1 = self.pool(out_F40)
        class_predict1 = class_predict1.view(class_predict1.size(0), -1)

        Ft = class_predict1

        out_cls_list, out_cls_labels = self.class_glioma(class_predict1, label_list)
        loss_1 = loss_func(out_cls_list[0], out_cls_labels[0])
        loss_2 = loss_func(out_cls_list[1], out_cls_labels[1])
        loss_3 = loss_func(out_cls_list[2], out_cls_labels[2])
        loss_cls = loss_1 + loss_2 + loss_3
        return loss_cls, Ft

    # 提取分割特征用于原型聚类，以缓解多任务分类的梯度噪声问题,其中，fuse0以及fuse1符合Backbone的原始机制
    def g_fai(self, Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1, index_seg, x, loss_DC, seg_mask1):
        ### segmentation branch
        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.upsample(out_F3_0, size=Scale2A.size()[2:], mode='trilinear')

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(out_F2_0) * Scale2A)
        out_F2_2 = F.upsample(out_F2_1, size=Scale1A.size()[2:], mode='trilinear')

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(out_F1_0) * Scale1A)

        out_F1_up = F.interpolate(out_F1_1, scale_factor=2, mode='trilinear', align_corners=False)
        out_F1_up = self.conv1_up(out_F1_up)
        out_F1_2 = F.upsample(out_F1_up[index_seg].unsqueeze(0), size=x.size()[2:], mode='trilinear')

        fuse0 = F.upsample(fuse0[index_seg].unsqueeze(0), size=x.size()[2:], mode='trilinear')
        fuse1 = F.upsample(fuse1[index_seg].unsqueeze(0), size=x.size()[2:], mode='trilinear')

        seg_fuse0 = F.softmax(self.predict_fuse0(fuse0), 1)
        seg_fuse1 = F.softmax(self.predict_fuse1(fuse1), 1)
        seg_predict = F.softmax(self.predict(out_F1_2), 1)

        loss_seg = loss_DC(seg_fuse0, seg_mask1) + loss_DC(seg_fuse1, seg_mask1) + loss_DC(seg_predict, seg_mask1)
        # 返回Mt用于原型聚类
        Mt = out_F1_1
        return loss_seg, Mt

    def forward(self, input_imgs, imgs_seg, label_list, seg_mask1, loss_func, loss_DC):
        x = torch.cat((input_imgs, imgs_seg), 0)
        index_seg = input_imgs.size(0)

        Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1 = self.f_theta_1(x)

        loss_cls, Ft = self.f_theta_2(Scale1A, Scale2A, Scale3A, Scale4A, label_list, loss_func)
        loss_seg, Mt = self.g_fai(Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1, index_seg, x, loss_DC, seg_mask1)

        return (loss_cls, loss_seg, Mt, Ft)

    def get_features(self, input):
        Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1 = self.f_theta_1(input)

        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.upsample(out_F3_0, size=Scale2A.size()[2:], mode='trilinear')

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(out_F2_0) * Scale2A)
        out_F2_2 = F.upsample(out_F2_1, size=Scale1A.size()[2:], mode='trilinear')

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(out_F1_0) * Scale1A)
        Mt = out_F1_1
        return Mt

    def get_m_features(self, input):
        layer0 = self.backbone.layer0(input)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def predictcls(self, input, cls_index):
        Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1 = self.f_theta_1(input)
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(out_F20) * Scale2A)
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(out_F30) * Scale3A)

        out_F40 = torch.cat((out_F31, Scale4A), 1)

        class_predict1 = self.pool(out_F40)
        class_predict1 = class_predict1.view(class_predict1.size(0), -1)

        out_cls = self.CLSmodel[cls_index](class_predict1)
        return out_cls

    def predictseg(self, input):
        Scale1A, Scale2A, Scale3A, Scale4A, fuse0, fuse1 = self.f_theta_1(input)

        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.upsample(out_F3_0, size=Scale2A.size()[2:], mode='trilinear')

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(out_F2_0) * Scale2A)
        out_F2_2 = F.upsample(out_F2_1, size=Scale1A.size()[2:], mode='trilinear')

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(out_F1_0) * Scale1A)

        out_F1_up = F.interpolate(out_F1_1, scale_factor=2, mode='trilinear', align_corners=False)
        out_F1_up = self.conv1_up(out_F1_up)
        out_F1_2 = F.upsample(out_F1_up, size=input.size()[2:], mode='trilinear')
        seg_predict = F.softmax(self.predict(out_F1_2), 1)
        return seg_predict

# 辅助分类器
class AuxClassifer(nn.Module):
    def __init__(self, n_classes, num_channels, vis=False):
        super(AuxClassifer, self).__init__()
        self.name = 'AuxClassifer'
        self.bivablock1 = BiVA(num_channels=num_channels, first_time=True)
        self.fusion0 = nn.Sequential(
            nn.Conv3d(num_channels * 4, num_channels, kernel_size=1), nn.GroupNorm(num_channels // 2, num_channels),
            nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels // 2, num_channels), nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels // 2, num_channels), nn.PReLU()
        )
        self.ASA0 = ASA3D(num_channels=num_channels, level=0, vis=vis)
        self.ASA1 = ASA3D(num_channels=num_channels, level=1, vis=vis)
        self.ASA2 = ASA3D(num_channels=num_channels, level=2, vis=vis)
        self.ASA3 = ASA3D(num_channels=num_channels, level=3, vis=vis)
        ### classification branch
        self.pool0 = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.attention2 = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels // 2, num_channels),
            nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels // 2, num_channels), nn.PReLU()
        )
        self.pool1 = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.attention3 = nn.Sequential(
            nn.Conv3d(num_channels * 2, num_channels, kernel_size=1), nn.GroupNorm(num_channels // 2, num_channels),
            nn.PReLU(),
            nn.Conv3d(num_channels, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels // 2, num_channels), nn.PReLU()
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.CLSmodel = nn.ModuleList([])
        for i in range(3):
            self.CLSmodel.append(nn.Linear(num_channels * 2, n_classes))

    def forward(self, layer1, layer2, layer3, layer4, label_list, loss_func):
        features = self.get_features(layer1, layer2, layer3, layer4)

        logist_1p19q_final = self.CLSmodel[0](features)
        logist_IDH_final = self.CLSmodel[1](features)
        logist_LHG_final = self.CLSmodel[2](features)

        out_cls_list_final = []
        out_cls_list_final.append(logist_1p19q_final[0:2, :])
        out_cls_list_final.append(logist_IDH_final[2:4, :])
        out_cls_list_final.append(logist_LHG_final[4:6, :])

        loss_1 = loss_func(out_cls_list_final[0], label_list[0])
        loss_2 = loss_func(out_cls_list_final[1], label_list[1])
        loss_3 = loss_func(out_cls_list_final[2], label_list[2])

        return loss_1, loss_2, loss_3

    def predictcls(self, layer1, layer2, layer3, layer4, cls_index):
        with torch.no_grad():
            features = self.get_features(layer1, layer2, layer3, layer4)
            class_predict = self.CLSmodel[cls_index](features)
            return class_predict

    def get_features(self, layer1, layer2, layer3, layer4):
        Scale1V, Scale2V, Scale3V, Scale4V = self.bivablock1(layer1, layer2, layer3, layer4)
        F20_upsample = F.upsample(Scale2V, size=layer1.size()[2:], mode='trilinear')
        F30_upsample = F.upsample(Scale3V, size=layer1.size()[2:], mode='trilinear')
        F40_upsample = F.upsample(Scale4V, size=layer1.size()[2:], mode='trilinear')

        fuse0 = self.fusion0(torch.cat((F40_upsample, F30_upsample, F20_upsample, Scale1V), 1))
        #
        Scale1A = self.ASA0(Scale1V, fuse0)
        Scale2A = self.ASA1(Scale2V, fuse0)
        Scale3A = self.ASA2(Scale3V, fuse0)
        Scale4A = self.ASA3(Scale4V, fuse0)

        ### classificication branch
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(out_F20) * Scale2A)
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(out_F30) * Scale3A)

        out_F40 = torch.cat((out_F31, Scale4A), 1)

        class_predict1 = self.pool(out_F40)
        features = class_predict1.view(class_predict1.size(0), -1)
        return features