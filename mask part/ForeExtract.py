import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import CBAM


class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                             dilation=dilation, groups=channels,bias=False, padding_mode='reflect')
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                             dilation=dilation, groups=channels,bias=False, padding_mode='reflect')
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


# 1×1 conv
class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')

    def forward(self,x):
        return self.conv(x)


# 1×1 conv+BN+relu
class Conv1BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Conv+BN
class ConvBn(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# conv+bn+leaky_relu


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
        # x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.2)

# conv+bn+tanh


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.tanh(x)/2+0.5  # 输出归一化到[0,1]

# conv+LeakyRelu
class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

# conv+relu


class ConvRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        x = self.conv(x)
        x = self.relu(x)
        return x


# conv+relu


class ConvBnRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# dilation+BN+LRelu
class DilationConv(nn.Module):
    # dilation convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2, groups=1):
        super(DilationConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.2)


# 55&77Conv+BN+LRelu
class Conv57BNLR(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, groups=1):
        super(Conv57BNLR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.2)

# 深度可分离卷积

class DepthwiseConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class DepthwiseDense(nn.Module):
    """轻量稠密块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = DepthwiseConv(in_channels, mid)
        self.conv2 = DepthwiseConv(in_channels + mid, out_channels)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(torch.cat([x, o1], dim=1))
        return o2
# dense_block

class Dense2Block(nn.Module):
    def __init__(self, channels, output_channels):
        super(Dense2Block, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        self.bn = nn.BatchNorm2d(3*channels)
        self.conv_final = nn.Conv2d(3*channels, output_channels, kernel_size=3, padding=1, padding_mode='reflect')
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        o1 = self.conv1(x)  # c
        i2 = torch.cat((x, o1), dim=1)  # 2c
        o2 = self.conv2(i2)  # c
        output = torch.cat((x, o1, o2), dim=1)  # 3c
        # output = self.bn(output)
        output = self.conv_final(output)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return output


class Dense3Block(nn.Module):
    def __init__(self, channels, output_channels):
        super(Dense3Block, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        self.conv3 = ConvLeakyRelu2d(3*channels, channels)
        self.bn = nn.BatchNorm2d(4*channels)
        self.conv_final = nn.Conv2d(4 * channels, output_channels, kernel_size=3, padding=1,padding_mode='reflect')

    def forward(self, x):
        o1 = self.conv1(x)  # c
        i2 = torch.cat((x, o1), dim=1)  # 2c
        o2 = self.conv2(i2)
        i3 = torch.cat((x, o1, o2), dim=1)  # 3c
        o3 = self.conv3(i3)
        output = torch.cat((x, o1, o2, o3), dim=1)
        # output = self.bn(output)
        output = self.conv_final(output)
        return output


#  轻量级瓶颈融合
class LightBottleneckFusion(nn.Module):
    """轻量级瓶颈融合"""

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.fusion = nn.Sequential(
            # 1. 通道压缩
            nn.Conv2d(in_channels, mid_channels, 1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),

            # 2. 空间信息交互（轻量3x3）
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),

            # 3. 通道扩展
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            # 最后不加激活，让主网络决定
        )

    def forward(self, x):
        return self.fusion(x)


class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DilatedConv, self).__init__()
        self.daconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1,
                                dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LRelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        output = self.daconv(x)
        output = self.bn(output)
        output = self.LRelu(output)
        return output

class ForeTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ForeTrans, self).__init__()
        self.block1 = ConvRelu2d(in_channels, out_channels)
        self.dense_block2 = Dense2Block(in_channels, out_channels)
        self.dense_block3 = Dense3Block(in_channels, out_channels)
        self.sobel = Sobelxy(in_channels)
        self.conv1bn = Conv1BN(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.conv1x1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.dense_block2(x)
        x3 = self.dense_block3(x)
        x4 = self.sobel(x)
        x4 = self.conv1bn(x4)
        output = torch.cat((x1, x2, x3, x4), dim=1)
        # out = self.bn(output)
        out = self.conv1x1(output)
        return out, self.relu(out)

class DWForeTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWForeTrans, self).__init__()
        self.block1 = DepthwiseConv(in_channels, out_channels)
        self.dense_block2 = DepthwiseDense(in_channels, out_channels)
        self.sobel = Sobelxy(in_channels)
        self.sobelconv1 = ConvBn(in_channels, in_channels)
        self.sobelconv2 = Conv1BN(in_channels, out_channels)
        # self.bn = nn.BatchNorm2d(out_channels * 3)
        self.lbnconv = LightBottleneckFusion(out_channels * 3, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.dense_block2(x)
        x3_weight = self.sobel(x)
        x3 = self.sobelconv1(x)
        x3 = x3.mul(x3_weight)
        x3 = self.sobelconv2(x3)
        output = torch.cat((x1, x2, x3), dim=1)
        # out = self.bn(output)
        out = self.lbnconv(output)
        return out, self.relu(out)

class BackTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BackTrans, self).__init__()
        self.dilation2 = DilatedConv(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilation3 = DilationConv(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv5 = Conv57BNLR(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = Conv57BNLR(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.conv1x1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.dilation2(x)
        out2 = self.dilation3(x)
        out3 = self.conv5(x)
        out4 = self.conv7(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.conv1x1(out)
        return self.relu(out)


class MyNet(nn.Module):
    def __init__(self, in_channels):
        super(MyNet, self).__init__()
        ch_num = [16, 32, 64, 128]
        # 编码器浅层提取
        self.conv_vi = ConvLeakyRelu2d(in_channels, ch_num[0])
        self.conv_ir = ConvLeakyRelu2d(in_channels, ch_num[0])

        # 编码器目标和背景模块
        self.vitarget_encoder = ForeTrans(ch_num[0], ch_num[1])
        self.irtarget_encoder = ForeTrans(ch_num[0], ch_num[1])
        self.viback_encoder = BackTrans(ch_num[0], ch_num[1])
        self.irback_encoder = BackTrans(ch_num[0], ch_num[1])

        #  目标和背景融合模块
        self.target_fusion = CBAM.CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.back_fusion = CBAM.CBAMBlock(channel=64, reduction=4, kernel_size=7)
        # 解码器目标和背景模块
        self.target_decoder = ForeTrans(ch_num[2], ch_num[1])
        self.back_decoder = BackTrans(ch_num[2], ch_num[1])
        self.last_decoder = nn.Sequential(ConvBnRelu2d(ch_num[2], ch_num[1]),
                                          ConvBnRelu2d(ch_num[1], ch_num[0]),
                                          ConvBnTanh2d(ch_num[0], 1, kernel_size=1, padding=0))

    def forward(self, ir, vi, mask):
        # encoder
        ir_shallow = self.conv_ir(ir)
        vi_shallow = self.conv_vi(vi)
        irtarget_f = self.irtarget_encoder(ir_shallow)
        irback_f = self.irback_encoder(ir_shallow)
        vitarget_f = self.vitarget_encoder(vi_shallow)
        viback_f = self.viback_encoder(vi_shallow)
        target_f = torch.cat((irtarget_f, vitarget_f), dim=1)
        back_f = torch.cat((irback_f, viback_f), dim=1)
        masked_target_f = target_f * mask
        masked_back_f = back_f * (1 - mask)
        # fusion
        target = self.target_fusion(masked_target_f)
        back = self.back_fusion(masked_back_f)
        # decode
        target_de = self.target_decoder(target)
        back_de = self.back_decoder(back)
        out_de = torch.cat((target_de, back_de), dim=1)
        out = self.last_decoder(out_de)
        return out

class vgg_like_foretrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vgg_like_foretrans, self).__init__()
        self.conv1 = ConvRelu2d(in_channels, out_channels)
        self.conv2 = ConvLeakyRelu2d(in_channels+out_channels, out_channels)
        # self.bn = nn.BatchNorm2d(3*channels)
        self.conv_final = nn.Conv2d(in_channels+2*out_channels, out_channels, kernel_size=3, padding=1,
                                    padding_mode='reflect')
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        o1 = self.conv1(x)  # c
        i2 = torch.cat((x, o1), dim=1)  # 2c
        o2 = self.conv2(i2)  # c
        output = torch.cat((x, o1, o2), dim=1)  # 3c
        # output = self.bn(output)
        out = self.conv_final(output)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return out, self.relu(out)

class Extract(nn.Module):
    def __init__(self, in_channels):
        super(Extract, self).__init__()
        self.shallow = nn.Sequential(ConvRelu2d(in_channels, 16),
                                     ConvRelu2d(16, 64))
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Feature1 = DWForeTrans(64, 128)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Feature2 = DWForeTrans(128, 256)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Feature3 = DWForeTrans(256, 512)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Feature4 = DWForeTrans(512, 512)

    def forward(self, x):
        tmp_x = []
        feature0 = self.shallow(x)  # 64
        tmp_x.append(feature0)
        max_feature = self.max1(feature0)
        out1, feature1 = self.Feature1(max_feature)  # 128
        tmp_x.append(feature1)
        max_feature = self.max2(feature1)
        out2, feature2 = self.Feature2(max_feature)  # 256
        tmp_x.append(feature2)
        max_feature = self.max3(feature2)
        out3, feature3 = self.Feature3(max_feature)  # 512
        tmp_x.append(feature3)
        max_feature = self.max4(feature3)
        out4, feature4 = self.Feature4(max_feature)  # 22*512
        tmp_x.append(feature4 )
        # print(f"out[0] shape: {tmp_x[0].shape}")
        # print(f"out[1] shape: {tmp_x[1].shape}")
        # print(f"out[2] shape: {tmp_x[2].shape}")
        # print(f"out[3] shape: {tmp_x[3].shape}")
        return tmp_x





if __name__ == "__main__":
    ir_tensor = torch.randn(1, 3, 32, 32)  # 输入：通道数64，尺寸32x32
    vi_tensor = torch.randn(1, 3, 32, 32)
    mask = torch.randn(1, 1, 32, 32)
    mask = mask - mask.min()
    mask = mask / mask.max()
    # model = MyNet(in_channels=1)
    model = Extract(3)
    output = model(ir_tensor)