from skimage.feature import graycomatrix, graycoprops
import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
from thop import profile

Epsilon = 1e-5


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.2)


class Dense2Block(nn.Module):
    def __init__(self, channels, output_channels):
        super(Dense2Block, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(channels, channels)
        self.conv2 = ConvBnLeakyRelu2d(2 * channels, channels)
        self.bn = nn.BatchNorm2d(3 * channels)
        self.conv_final = nn.Conv2d(3 * channels, output_channels, kernel_size=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        o1 = self.conv1(x)  # c
        i2 = torch.cat((x, o1), dim=1)  # 2c
        o2 = self.conv2(i2)  # c
        output = torch.cat((x, o1, o2), dim=1)  # 3c
        output = self.bn(output)
        output = self.conv_final(output)
        output = self.leaky_relu(output)
        return output


def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf values!")


class Fusion_mask(torch.nn.Module):
    def __init__(self):
        super(Fusion_mask, self).__init__()

    def Salient_Background(self, img_ir, img_vi, img_mask):
        img_ir_ahead = torch.mul(img_mask, img_ir)
        img_ir_back = torch.mul(1 - img_mask, img_ir)

        img_vi_ahead = torch.mul(img_mask, img_vi)
        img_vi_back = torch.mul(1 - img_mask, img_vi)

        return img_ir_ahead, img_ir_back, img_vi_ahead, img_vi_back

    def SOS_fusion(self, en_vi, en_ir, mask_list, p_type='avg'):
        SCA = Spatial_Channel_Attention()
        fusion_function = SCA.attention_fusion_weight
        en_vi_a, en_vi_b, en_ir_a, en_ir_b = [], [], [], []
        # en_vi_a[0], en_vi_b[0], en_ir_a[0], en_ir_b[0] = en_vi[0] * mask_list[0], en_vi[0] * (1 - mask_list[0]), en_ir[0] * mask_list[0], en_ir[0] * (1 - mask_list[0])
        # en_vi_a[1], en_vi_b[1], en_ir_a[1], en_ir_b[1] = en_vi[1] * mask_list[1], en_vi[1] * (1 - mask_list[1]), en_ir[1] * mask_list[1], en_ir[1] * (1 - mask_list[1])
        # en_vi_a[2], en_vi_b[2], en_ir_a[2], en_ir_b[2] = en_vi[2] * mask_list[2], en_vi[2] * (1 - mask_list[2]), en_ir[2] * mask_list[2], en_ir[2] * (1 - mask_list[2])
        # en_vi_a[3], en_vi_b[3], en_ir_a[3], en_ir_b[3] = en_vi[3] * mask_list[3], en_vi[3] * (1 - mask_list[3]), en_ir[3] * mask_list[3], en_ir[3] * (1 - mask_list[3])
        en_vi_a = [en_vi[0] * mask_list[0], en_vi[1] * mask_list[1], en_vi[2] * mask_list[2], en_vi[3] * mask_list[3]]
        en_vi_b = [en_vi[0] * (1 - mask_list[0]), en_vi[1] * (1 - mask_list[1]), en_vi[2] * (1 - mask_list[2]), en_vi[3] * (1 - mask_list[3])]
        en_ir_a = [en_ir[0] * mask_list[0], en_ir[1] * mask_list[1], en_ir[2] * mask_list[2], en_ir[3] * mask_list[3]]
        en_ir_b = [en_ir[0] * (1 - mask_list[0]), en_ir[1] * (1 - mask_list[1]), en_ir[2] * (1 - mask_list[2]), en_ir[3] * (1 - mask_list[3])]
        SCA_1 = fusion_function(en_ir_b[0], en_vi_b[0])
        SCA_2 = fusion_function(en_ir_b[1], en_vi_b[1])
        SCA_3 = fusion_function(en_ir_b[2], en_vi_b[2])
        SCA_4 = fusion_function(en_ir_b[3], en_vi_b[3])

        SCA_weight_1 = SCA_1 / (SCA_1 + en_vi_b[0] + 1e-5)
        SCA_weight_2 = SCA_2 / (SCA_2 + en_vi_b[1] + 1e-5)
        SCA_weight_3 = SCA_3 / (SCA_3 + en_vi_b[2] + 1e-5)
        SCA_weight_4 = SCA_4 / (SCA_4 + en_vi_b[3] + 1e-5)

        s_miu = 0.6

        # fx_0 = (addition of significance) + 0.6 * (softmax for vi_b and SCA) + 0.4 * vi_b
        f1_0 = (s_miu * en_ir_a[0] + (1 - s_miu) * en_vi_a[0]
                ) + 0.6 * (SCA_weight_1 * SCA_1 + (1 - SCA_weight_1) * en_vi_b[0]) + 0.8 * en_vi_b[0]
        f2_0 = (s_miu * en_ir_a[1] + (1 - s_miu) * en_vi_a[1]
                ) + 0.6 * (SCA_weight_2 * SCA_2 + (1 - SCA_weight_2) * en_vi_b[1]) + 0.8 * en_vi_b[1]
        f3_0 = (s_miu * en_ir_a[2] + (1 - s_miu) * en_vi_a[2]
                ) + 0.6 * (SCA_weight_3 * SCA_3 + (1 - SCA_weight_3) * en_vi_b[2]) + 0.8 * en_vi_b[2]
        f4_0 = (s_miu * en_ir_a[3] + (1 - s_miu) * en_vi_a[3]
                ) + 0.6 * (SCA_weight_4 * SCA_4 + (1 - SCA_weight_4) * en_vi_b[3]) + 0.8 * en_vi_b[3]
        return [f1_0, f2_0, f3_0, f4_0]


    def Fusion_foreback(self, vi_fore, vi_back, ir_fore, ir_back, p_type='avg'):
        f1_0 = self.obj_back_fusion(vi_fore[0], vi_back[0], ir_fore[0], ir_back[0])
        f2_0 = self.obj_back_fusion(vi_fore[1], vi_back[1], ir_fore[1], ir_back[1])
        f3_0 = self.obj_back_fusion(vi_fore[2], vi_back[2], ir_fore[2], ir_back[2])
        f4_0 = self.obj_back_fusion(vi_fore[3], vi_back[3], ir_fore[3], ir_back[3])
        return [f1_0, f2_0, f3_0, f4_0]

class Spatial_Channel_Attention(torch.nn.Module):
    def __init__(self):
        super(Spatial_Channel_Attention, self).__init__()
        self.epsilon = 1e-5
        self.global_pooling_type = 'average_global_pooling' # channel_attention
        self.spatial_type = 'mean' # spatial_attention

    # attention fusion strategy
    def attention_fusion_weight(self, tensor1, tensor2):
        f_channel = self.channel_fusion(tensor1, tensor2)
        f_spatial = self.spatial_fusion(tensor1, tensor2)

        tensor_fusion = (f_channel + f_spatial) / 2

        return tensor_fusion

    def channel_fusion(self, tensor1, tensor2):
        shape = tensor1.size()
        # calculate channel attention
        global_p1 = self.channel_attention(tensor1)
        global_p2 = self.channel_attention(tensor2)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + self.epsilon)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + self.epsilon)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])  # 全局池化后重新扩展维度与特征图相乘
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

        return tensor_f

    def spatial_fusion(self, tensor1, tensor2):
        shape = tensor1.size()
        # calculate spatial attention
        spatial1 = self.spatial_attention(tensor1)
        spatial2 = self.spatial_attention(tensor2)

        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + self.epsilon)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + self.epsilon)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def channel_attention(self, tensor):
        # global pooling
        shape = tensor.size()

        if self.global_pooling_type == 'average_global_pooling':
            pooling_function = F.avg_pool2d
        elif self.global_pooling_type == 'max_global_pooling':
            pooling_function = F.max_pool2d
        global_p = pooling_function(tensor, kernel_size=shape[2:])
        return global_p

    def spatial_attention(self, tensor):
        spatial = []
        if self.spatial_type == 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif self.spatial_type == 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        return spatial

class Saliency_weight_module(nn.Module):
    def __init__(self):
        super(Saliency_weight_module, self).__init__()
        self.conv_fore1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.Avg = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Max = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convpool_fore = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv_fore2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.Dense_fore = Dense2Block(channels=4, output_channels=16)
        self.conv_fore_final = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.conv_back1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.convpool_back = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv_back2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=2, dilation=2, stride=1)
        self.conv_back_final = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ir, mask, en_vis_fore, en_vis_back, en_ir_fore, en_ir_back):
        check_nan_inf(ir, "ir")
        check_nan_inf(mask, "mask")
        check_nan_inf(en_ir_fore[0], "en_ir_fore")
        check_nan_inf(en_ir_back[1], "en_ir_back")
        check_nan_inf(en_vis_fore[2], "en_vis_fore")
        check_nan_inf(en_vis_back[3], "en_vis_back")
        B, C, H, W = ir.shape
        ir_fore = ir * mask
        ir_back = ir * (1 - mask)
        branch_fore1 = self.conv_fore1(ir_fore)
        average1 = self.Avg(branch_fore1)
        maxpool1 = self.Max(branch_fore1)
        branch_fore1 = torch.cat([average1, maxpool1], dim=1)
        branch_fore1 = self.convpool_fore(branch_fore1)
        branch_fore1 = F.interpolate(branch_fore1, size=(H, W), mode='bilinear', align_corners=False)
        branch_fore2 = self.conv_fore2(ir_fore)
        branch_fore2 = self.Dense_fore(branch_fore2)
        output1 = branch_fore1 + branch_fore2
        output1 = self.conv_fore_final(output1)
        ir_fore_weight = torch.clamp(self.sigmoid(output1), 0, 1)
        branch_back1 = self.conv_back1(ir_back)
        average2 = self.Avg(branch_back1)
        maxpool2 = self.Max(branch_back1)
        branch_back1 = torch.cat([average2, maxpool2], dim=1)
        branch_back1 = self.convpool_back(branch_back1)
        branch_back1 = F.interpolate(branch_back1, size=(H, W), mode='bilinear', align_corners=False)
        branch_back2 = self.conv_back2(ir_back)
        output2 = branch_back1 + branch_back2
        output2 = self.conv_back_final(output2)
        ir_back_weight = torch.clamp(self.sigmoid(output2), 0, 1)
        ir_fore_weight = ir_fore_weight * mask
        ir_back_weight = ir_back_weight * (1 - mask)
        ir_fore_weight = ir_fore_weight.sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-15)
        ir_back_weight = ir_back_weight.sum(dim=[2, 3]) / ((1 - mask).sum(dim=[2, 3]) + 1e-15)
        ir_fore_weight = ir_fore_weight.view(B, C, 1, 1)
        ir_back_weight = ir_back_weight.view(B, C, 1, 1)
        vi_fore_weight = 1 - ir_fore_weight
        vi_back_weight = 1 - ir_back_weight
        f1_0 = ir_fore_weight * en_ir_fore[0] + vi_fore_weight * en_vis_fore[0] + 0.6 * (
                    ir_back_weight * en_ir_back[0] + vi_back_weight * en_vis_back[0]) + 0.4 * en_vis_back[0]
        f2_0 = ir_fore_weight * en_ir_fore[1] + vi_fore_weight * en_vis_fore[1] + 0.6 * (
                    ir_back_weight * en_ir_back[1] + vi_back_weight * en_vis_back[1]) + 0.4 * en_vis_back[1]
        f3_0 = ir_fore_weight * en_ir_fore[2] + vi_fore_weight * en_vis_fore[2] + 0.6 * (
                    ir_back_weight * en_ir_back[2] + vi_back_weight * en_vis_back[2]) + 0.4 * en_vis_back[2]
        f4_0 = ir_fore_weight * en_ir_fore[3] + vi_fore_weight * en_vis_fore[3] + 0.6 * (
                    ir_back_weight * en_ir_back[3] + vi_back_weight * en_vis_back[3]) + 0.4 * en_vis_back[3]

        return [f1_0, f2_0, f3_0, f4_0]


class Map_fusion_fore(nn.Module):
    def __init__(self, ch):
        super(Map_fusion_fore, self).__init__()
        self.conv_fore1 = nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, stride=1)
        self.Avg = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Max = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convpool_fore = nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, stride=1)
        self.conv_fore2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=1)
        self.Dense_fore = Dense2Block(ch, ch // 2)
        self.conv_fore_final1 = nn.Conv2d(2 * ch // 2, ch // 4, kernel_size=3, padding=1, stride=1)
        self.conv_fore_final2 = nn.Conv2d(ch // 4, 1, kernel_size=1, padding=0, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1):
        B, C, H, W = input1.shape
        branch_fore1 = self.conv_fore1(input1)
        average1 = self.Avg(branch_fore1)
        maxpool1 = self.Max(branch_fore1)
        branch_fore1 = torch.cat([average1, maxpool1], dim=1)
        branch_fore1 = self.convpool_fore(branch_fore1)
        branch_fore1 = F.interpolate(branch_fore1, size=(H, W), mode='bilinear', align_corners=False)
        branch_fore2 = self.conv_fore2(input1)
        branch_fore2 = self.Dense_fore(branch_fore2)
        output1 = torch.cat((branch_fore1, branch_fore2), dim=1)
        output1 = self.conv_fore_final1(output1)
        output1 = self.conv_fore_final2(output1)
        weight1 = self.sigmoid(output1)
        return weight1


class Map_fusion_back(nn.Module):
    def __init__(self, ch):
        super(Map_fusion_back, self).__init__()
        self.Avg = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Max = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_back1 = nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, stride=1)
        self.convpool_back = nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, stride=1)
        self.conv_back2 = nn.Conv2d(ch, ch // 2, kernel_size=3, padding=2, dilation=2, stride=1)
        self.conv_back_final1 = nn.Conv2d(2 * ch // 2, ch // 4, kernel_size=3, padding=1, stride=1)
        self.conv_back_final2 = nn.Conv2d(ch // 4, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1):
        B, C, H, W = input1.shape
        branch_back1 = self.conv_back1(input1)
        average2 = self.Avg(branch_back1)
        maxpool2 = self.Max(branch_back1)
        branch_back1 = torch.cat([average2, maxpool2], dim=1)
        branch_back1 = self.convpool_back(branch_back1)
        branch_back1 = F.interpolate(branch_back1, size=(H, W), mode='bilinear', align_corners=False)
        branch_back2 = self.conv_back2(input1)
        output2 = torch.cat((branch_back1, branch_back2), dim=1)
        output2 = self.conv_back_final1(output2)
        output2 = self.conv_back_final2(output2)
        weight2 = self.sigmoid(output2)
        return weight2


class Map_fusion(nn.Module):
    def __init__(self, ch_num):
        super(Map_fusion, self).__init__()
        self.fore1 = Map_fusion_fore(ch_num[0])
        self.back1 = Map_fusion_back(ch_num[0])

        self.fore2 = Map_fusion_fore(ch_num[1])
        self.back2 = Map_fusion_back(ch_num[1])

        self.fore3 = Map_fusion_fore(ch_num[2])
        self.back3 = Map_fusion_back(ch_num[2])

        self.fore4 = Map_fusion_fore(ch_num[3])
        self.back4 = Map_fusion_back(ch_num[3])

    def forward(self, en_vis_fore, en_vis_back, en_ir_fore, en_ir_back):
        ir_fore_weight1 = self.fore1(en_ir_fore[0])
        vis_back_weight1 = self.back1(en_vis_back[0])

        ir_fore_weight2 = self.fore2(en_ir_fore[1])
        vis_back_weight2 = self.back2(en_vis_back[1])

        ir_fore_weight3 = self.fore3(en_ir_fore[2])
        vis_back_weight3 = self.back3(en_vis_back[2])

        ir_fore_weight4 = self.fore4(en_ir_fore[3])
        vis_back_weight4 = self.back4(en_vis_back[3])

        feature1 = 0.4 * (ir_fore_weight1 * en_ir_fore[0] + (1 - ir_fore_weight1) * en_vis_fore[0]) + 0.6 * en_ir_fore[
            0]
        + 0.4 * (vis_back_weight1 * en_vis_back[0] + (1 - vis_back_weight1) * en_ir_back[0]) + 0.6 * en_vis_back[0]
        feature2 = 0.4 * (ir_fore_weight2 * en_ir_fore[1] + (1 - ir_fore_weight2) * en_vis_fore[1]) + 0.6 * en_ir_fore[
            1]
        + 0.4 * (vis_back_weight2 * en_vis_back[1] + (1 - vis_back_weight2) * en_ir_back[1]) + 0.6 * en_vis_back[1]
        feature3 = 0.4 * (ir_fore_weight3 * en_ir_fore[2] + (1 - ir_fore_weight3) * en_vis_fore[2]) + 0.6 * en_ir_fore[
            2]
        + 0.4 * (vis_back_weight3 * en_vis_back[2] + (1 - vis_back_weight3) * en_ir_back[2]) + 0.6 * en_vis_back[2]
        feature4 = 0.4 * (ir_fore_weight4 * en_ir_fore[3] + (1 - ir_fore_weight4) * en_vis_fore[3]) + 0.6 * en_ir_fore[
            3]
        + 0.4 * (vis_back_weight4 * en_vis_back[3] + (1 - vis_back_weight4) * en_ir_back[3]) + 0.6 * en_vis_back[3]

        return [feature1, feature2, feature3, feature4]


# 梯度纹理权重模块
class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobel_x = self.convx(x)
        sobel_y = self.convy(x)
        image_gradient = torch.abs(sobel_x) + torch.abs(sobel_y)
        # image_gradient = torch.sqrt(sobel_x**2 + sobel_y**2)
        return image_gradient


def l2_norm(tensor):
    return torch.norm(tensor, p=2)








class ConvBnRe(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(ConvBnRe, self).__init__()

        self.ConvModule = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ConvModule(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x) * x_source + x_source


# ---  需要学习的权重融合模块来自A semantic-driven coupled network for infrared and visible image fusion  ---#
class CDFusion(nn.Module):
    def __init__(self, ch):
        super(CDFusion, self).__init__()
        self.Main_Q = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Main_K = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Main_V = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)

        self.Aux_Q = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Aux_V = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)

        self.mix_att_output = ConvBnRe(ch * 3, ch, kernel_size=3, stride=1, padding=1)
        self.output = ConvBnRe(ch * 2, ch, kernel_size=3, padding=1, stride=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.main_attn = SpatialAttention()
        # self.aux_attn = SpatialAttention()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, main, aux):
        batchsize, c, h, w = main.shape

        main_down = F.interpolate(main, size=(32, 32), scale_factor=None, mode='bilinear', align_corners=None)
        aux_down = F.interpolate(aux, size=(32, 32), scale_factor=None, mode='bilinear', align_corners=None)
        # print(batchsize, c, h, w)
        # print(main_down.shape)
        # print(aux_down.shape)

        main_q = self.Main_Q(main_down)
        main_k = self.Main_K(main_down)
        main_v = self.Main_V(main_down)

        aux_q = self.Aux_Q(aux_down)
        aux_v = self.Aux_V(aux_down)

        joint_v = main_v + aux_v

        main_q = main_q.view(batchsize, -1, 32 * 32)
        main_k = main_k.view(batchsize, -1, 32 * 32).permute(0, 2, 1)
        main_v = main_v.view(batchsize, -1, 32 * 32)

        aux_q = aux_q.view(batchsize, -1, 32 * 32)
        # aux_v = aux_v.view(batchsize, -1, 32 * 32)
        joint_v = joint_v.view(batchsize, -1, 32 * 32)

        main_mask = torch.bmm(main_k, main_q)
        main_mask = self.softmax(main_mask)
        main_rebuild = torch.bmm(joint_v, main_mask.permute(0, 2, 1))
        main_rebuild = main_rebuild.view(batchsize, -1, 32, 32)
        main_rebuild = self.gamma1 * main_rebuild
        main_rebuild = F.interpolate(main_rebuild, size=(h, w),
                                     scale_factor=None, mode='bilinear', align_corners=None) + main

        main_aux_mask = torch.bmm(main_k, aux_q)
        main_aux_mask = self.softmax(main_aux_mask)
        main_aux_rebuild = torch.bmm(main_v, main_aux_mask.permute(0, 2, 1))
        main_aux_rebuild = main_aux_rebuild.view(batchsize, -1, 32, 32)
        main_aux_rebuild = self.gamma2 * main_aux_rebuild
        main_aux_rebuild = F.interpolate(main_aux_rebuild, size=(h, w),
                                         scale_factor=None, mode='bilinear', align_corners=None) + main

        mix_att = self.mix_att_output(torch.cat((main_rebuild, main_aux_rebuild), dim=1))
        main_single_att = self.main_attn(main)
        # aux_single_att = self.aux_attn(aux)
        output = self.output(torch.cat((mix_att, main_single_att), dim=1))
        # output = self.output(mix_att)

        return output


class MAFusion(nn.Module):
    def __init__(self, ch):
        super(MAFusion, self).__init__()

        self.Main_Q = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Main_K = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Main_V = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)

        self.Aux_Q = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Aux_K = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Aux_V = ConvBnRe(ch, ch, kernel_size=3, stride=1, padding=1)

        self.qkv_final = ConvBnRe(ch*3, ch, 3, 1, 1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.main_spatt = SpatialAttention()
        self.aux_spatt = SpatialAttention()

        self.all_final = ConvBnRe(ch*3, ch, 3, 1, 1)

    def forward(self, main, aux):
        b, c, h, w = main.shape

        main_down = F.interpolate(main, size=(32, 32), scale_factor=None, mode='bicubic', align_corners=None)
        aux_down = F.interpolate(aux, size=(32, 32), scale_factor=None, mode='bicubic', align_corners=None)

        main_q = self.Main_Q(main_down)
        main_k = self.Main_K(main_down)
        main_v = self.Main_V(main_down)

        aux_q = self.Aux_Q(aux_down)
        aux_k = self.Aux_K(aux_down)
        aux_v = self.Aux_V(aux_down)

        joint_v = main_v + aux_v

        main_v = main_v.view(b, -1, 32 * 32)
        main_k = main_k.view(b, -1, 32 * 32).permute(0, 2, 1)
        main_q = main_q.view(b, -1, 32 * 32)

        aux_v = aux_v.view(b, -1, 32 * 32)
        aux_k = aux_k.view(b, -1, 32 * 32).permute(0, 2, 1)
        aux_q = aux_q.view(b, -1, 32 * 32)

        joint_v = joint_v.view(b, -1, 32 * 32)

        # main_att
        main_mask = torch.bmm(main_k, main_q)
        main_mask = self.softmax(main_mask)
        main_rubild = torch.bmm(joint_v, main_mask.permute(0, 2, 1))
        main_rubild = main_rubild.view(b, -1, 32, 32)
        main_rubild = self.gamma1 * main_rubild
        # RGB_refine_view = main_rubild
        main_rubild = F.interpolate(main_rubild, size=(h, w), scale_factor=None, mode='bicubic',
                                    align_corners=None) + main

        # aux2main_att
        aux2main_mask = torch.bmm(aux_k, aux_q)
        aux2main_mask = self.softmax(aux2main_mask)
        aux2main_rebuild = torch.bmm(main_v, aux2main_mask.permute(0, 2, 1))
        aux2main_rebuild = aux2main_rebuild.view(b, -1, 32, 32)
        aux2main_rebuild = self.gamma2 * aux2main_rebuild
        # INF_refine_view = INF_refine
        aux2main_rebuild = F.interpolate(aux2main_rebuild, size=(h, w), scale_factor=None, mode='bicubic',
                                         align_corners=None) + main

        # main2aux_att
        main2aux_mask = torch.bmm(aux_k, main_q)
        main2aux_mask = self.softmax(main2aux_mask)
        main2aux_rebuild = torch.bmm(aux_v, main2aux_mask.permute(0, 2, 1))
        main2aux_rebuild = main2aux_rebuild.view(b, -1, 32, 32)
        main2aux_rebuild = self.gamma3 * main2aux_rebuild
        # RGB_INF_refine_view = main2aux_rebuild
        main2aux_rebuild = F.interpolate(main2aux_rebuild, size=(h, w), scale_factor=None, mode='bicubic',
                                         align_corners=None) + aux

        mix_transatt = self.qkv_final(torch.cat((main_rubild, aux2main_rebuild, main2aux_rebuild), dim=1))
        main_spatialatt = self.main_spatt(main)
        aux_spatialatt = self.aux_spatt(aux)

        out = self.all_final(torch.cat([mix_transatt, main_spatialatt, aux_spatialatt], dim=1))

        return out


class Learnable_Fusion(nn.Module):
    def __init__(self, channel_num):
        super(Learnable_Fusion, self).__init__()
        self.le_fusion_fore1 = MAFusion(channel_num[0])
        self.le_fusion_fore2 = MAFusion(channel_num[1])
        self.le_fusion_fore3 = MAFusion(channel_num[2])
        self.le_fusion_fore4 = MAFusion(channel_num[3])

        # 或许fore用MA，back用CD也可以
        self.le_fusion_back1 = MAFusion(channel_num[0])
        self.le_fusion_back2 = MAFusion(channel_num[1])
        self.le_fusion_back3 = MAFusion(channel_num[2])
        self.le_fusion_back4 = MAFusion(channel_num[3])

    def learnable_fusion_mask(self, en_vis, en_ir, mask_list):
        f1_fore = self.le_fusion_fore1(en_ir[0], en_vis[0]) * mask_list[0]
        f1_back = self.le_fusion_back1(en_vis[0], en_ir[0]) * (1 - mask_list[0])
        f2_fore = self.le_fusion_fore2(en_ir[1], en_vis[1]) * mask_list[1]
        f2_back = self.le_fusion_back2(en_vis[1], en_ir[1]) * (1 - mask_list[1])
        f3_fore = self.le_fusion_fore3(en_ir[2], en_vis[2]) * mask_list[2]
        f3_back = self.le_fusion_back3(en_vis[2], en_ir[2]) * (1 - mask_list[2])
        f4_fore = self.le_fusion_fore4(en_ir[3], en_vis[3]) * mask_list[3]
        f4_back = self.le_fusion_back4(en_vis[3], en_ir[3]) * (1 - mask_list[3])
        f1 = f1_fore + f1_back
        f2 = f2_fore + f2_back
        f3 = f3_fore + f3_back
        f4 = f4_fore + f4_back
        return [f1, f2, f3, f4]


class Learnable_Fusion_mask_before(nn.Module):
    def __init__(self, channel_num):
        super(Learnable_Fusion_mask_before, self).__init__()
        self.le_fusion_fore1 = MAFusion(channel_num[0])
        self.le_fusion_fore2 = MAFusion(channel_num[1])
        self.le_fusion_fore3 = MAFusion(channel_num[2])
        self.le_fusion_fore4 = MAFusion(channel_num[3])

        # 或许fore用MA，back用CD也可以
        self.le_fusion_back1 = MAFusion(channel_num[0])
        self.le_fusion_back2 = MAFusion(channel_num[1])
        self.le_fusion_back3 = MAFusion(channel_num[2])
        self.le_fusion_back4 = MAFusion(channel_num[3])

    def forward(self, en_vis, en_ir, mask_list):
        f1_fore = self.le_fusion_fore1(en_ir[0] * mask_list[0], en_vis[0] * mask_list[0])
        f1_back = self.le_fusion_back1(en_vis[0] * (1 - mask_list[0]), en_ir[0] * (1 - mask_list[0]))
        f2_fore = self.le_fusion_fore2(en_ir[1] * mask_list[1], en_vis[1] * mask_list[1])
        f2_back = self.le_fusion_back2(en_vis[1] * (1 - mask_list[1]), en_ir[1] * (1 - mask_list[1]))
        f3_fore = self.le_fusion_fore3(en_ir[2] * mask_list[2], en_vis[2] * mask_list[2])
        f3_back = self.le_fusion_back3(en_vis[2] * (1 - mask_list[2]), en_ir[2] * (1 - mask_list[2]))
        f4_fore = self.le_fusion_fore4(en_ir[3] * mask_list[3], en_vis[3] * mask_list[3])
        f4_back = self.le_fusion_back4(en_vis[3] * (1 - mask_list[3]), en_ir[3] * (1 - mask_list[3]))
        f1 = f1_fore + f1_back
        f2 = f2_fore + f2_back
        f3 = f3_fore + f3_back
        f4 = f4_fore + f4_back
        return [f1, f2, f3, f4]


class Learnable_Fusion_mask_en(nn.Module):
    def __init__(self, channel_num):
        super(Learnable_Fusion_mask_en, self).__init__()
        self.le_fusion_fore1 = MAFusion(channel_num[0])
        self.le_fusion_fore2 = MAFusion(channel_num[1])
        self.le_fusion_fore3 = MAFusion(channel_num[2])
        self.le_fusion_fore4 = MAFusion(channel_num[3])

        # 或许fore用MA，back用CD也可以
        self.le_fusion_back1 = MAFusion(channel_num[0])
        self.le_fusion_back2 = MAFusion(channel_num[1])
        self.le_fusion_back3 = MAFusion(channel_num[2])
        self.le_fusion_back4 = MAFusion(channel_num[3])

    def forward(self, en_vis_fore, en_vis_back, en_ir_fore, en_ir_back):
        f1_fore = self.le_fusion_fore1(en_ir_fore[0], en_vis_fore[0])
        f1_back = self.le_fusion_back1(en_vis_back[0], en_ir_back[0])
        f2_fore = self.le_fusion_fore2(en_ir_fore[1], en_vis_fore[1])
        f2_back = self.le_fusion_back2(en_vis_back[1], en_ir_back[1])
        f3_fore = self.le_fusion_fore3(en_ir_fore[2], en_vis_fore[2])
        f3_back = self.le_fusion_back3(en_vis_back[2], en_ir_back[2])
        f4_fore = self.le_fusion_fore4(en_ir_fore[3], en_vis_fore[3])
        f4_back = self.le_fusion_back4(en_vis_back[3], en_ir_back[3])
        f1 = f1_fore + f1_back
        f2 = f2_fore + f2_back
        f3 = f3_fore + f3_back
        f4 = f4_fore + f4_back
        return [f1, f2, f3, f4]


class Learnable_Fusion_mask_matrix(nn.Module):
    def __init__(self, channel_num):
        super(Learnable_Fusion_mask_matrix, self).__init__()
        self.le_fusion_fore1 = MAFusion(channel_num[0])
        self.le_fusion_fore2 = MAFusion(channel_num[1])
        self.le_fusion_fore3 = MAFusion(channel_num[2])
        self.le_fusion_fore4 = MAFusion(channel_num[3])

        # 或许fore用MA，back用CD也可以
        self.le_fusion_back1 = MAFusion(channel_num[0])
        self.le_fusion_back2 = MAFusion(channel_num[1])
        self.le_fusion_back3 = MAFusion(channel_num[2])
        self.le_fusion_back4 = MAFusion(channel_num[3])

    def learnable_fusion_mask(self, en_vis, en_ir, mask_list):
        f1_fore = 0.6 * self.le_fusion_fore1(en_ir[0] * mask_list[0], en_vis[0] * mask_list[0]) + 0.4 * \
                  self.le_fusion_fore1(en_vis[0] * mask_list[0], en_ir[0] * mask_list[0])
        f1_back = 0.6 * self.le_fusion_back1(en_vis[0] * (1 - mask_list[0]), en_ir[0] * (1 - mask_list[0])) + 0.4 * \
                  self.le_fusion_back1(en_ir[0] * (1 - mask_list[0]), en_vis[0] * (1 - mask_list[0]))
        f2_fore = 0.6 * self.le_fusion_fore2(en_ir[1] * mask_list[1], en_vis[1] * mask_list[1]) + 0.4 * \
                  self.le_fusion_fore2(en_vis[1] * mask_list[1], en_ir[1] * mask_list[1])
        f2_back = 0.6 * self.le_fusion_back2(en_vis[1] * (1 - mask_list[1]), en_ir[1] * (1 - mask_list[1])) + 0.4 * \
                  self.le_fusion_back2(en_ir[1] * (1 - mask_list[1]), en_vis[1] * (1 - mask_list[1]))
        f3_fore = 0.6 * self.le_fusion_fore3(en_ir[2] * mask_list[2], en_vis[2] * mask_list[2]) + 0.4 * \
                  self.le_fusion_fore3(en_vis[2] * mask_list[2], en_ir[2] * mask_list[2])
        f3_back = 0.6 * self.le_fusion_back3(en_vis[2] * (1 - mask_list[2]), en_ir[2] * (1 - mask_list[2])) + 0.4 * \
                  self.le_fusion_back3(en_ir[2] * (1 - mask_list[2]), en_vis[2] * (1 - mask_list[2]))
        f4_fore = 0.6 * self.le_fusion_fore4(en_ir[3] * mask_list[3], en_vis[3] * mask_list[3]) + 0.4 * \
                  self.le_fusion_fore4(en_vis[3] * mask_list[3],en_ir[3] * mask_list[3])
        f4_back = 0.6 * self.le_fusion_back4(en_vis[3] * (1 - mask_list[3]), en_ir[3] * (1 - mask_list[3])) + 0.4 * \
                  self.le_fusion_back4(en_ir[3] * (1 - mask_list[3]), en_vis[3] * (1 - mask_list[3]))
        f1 = f1_fore + f1_back
        f2 = f2_fore + f2_back
        f3 = f3_fore + f3_back
        f4 = f4_fore + f4_back
        return [f1, f2, f3, f4]


if __name__ == '__main__':
    channel_num = [64, 112, 160, 208, 256]
    sizes = [
        [1, 64, 300, 400],
        [1, 112, 150, 200],
        [1, 160, 75, 100],
        [1, 208, 38, 50]
    ]

    # 生成随机张量列表并放到CUDA上
    tensor_list = [torch.randn(*size).cuda().float() for size in sizes]
    net = Learnable_Fusion_mask_before(channel_num).cuda()
    flops, params = profile(net, inputs=(tensor_list, tensor_list, tensor_list))
    print(f"Total FLOPs: {flops / 1e9:.4f} G")
    print(f"Total parameters: {params / 1e6:.4f} M")
    # model = Map_fusion(channel_num=channel_num)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    # import torch
    #
    # weight_path = r'D:\ownproject\encoder\run_fusion\train_03-15_17-05\checkpoints\epoch295-loss1.335.pth'
    # # 加载 .pth 文件
    # checkpoint = torch.load(weight_path, map_location='cpu')  # 确保即使在无GPU环境下也能加载
    #
    # # 如果文件保存的是整个模型（含model键）
    # if 'model' in checkpoint:
    #     state_dict = checkpoint['model']
    # # 如果文件直接是state_dict
    # else:
    #     state_dict = checkpoint
    #
    # # 计算总参数量
    # total_params = sum(p.numel() for p in state_dict.values())
    # print(f"Total parameters: {total_params:,}")
    #
    # # 打印各层参数量（可选）
    # for name, param in state_dict.items():
    #     print(f"{name}: {param.numel():,}")

    # 假设输入张量
# B, C, H, W = 4, 3, 256, 256  # 批次大小、通道数、高度、宽度
# ir_background = torch.randn(B, C, H, W)  # 红外背景特征图
# vi_background = torch.randn(B, C, H, W)  # 可见光背景特征图

# 融合背景特征图
# fused_background = dynamic_weight_fusion(ir_background, vi_background)
# print("Fused background shape:", fused_background.shape)
