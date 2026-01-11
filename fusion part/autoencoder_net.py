import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                             dilation=dilation, groups=channels, bias=False, padding_mode='reflect')
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                             dilation=dilation, groups=channels, bias=False, padding_mode='reflect')
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


# 1×1 conv+BN
class Conv1BNLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1BNLRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.2)

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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

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
        x = self.bn(x)
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

    def forward(self, x):
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


# CBAM

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        return x * channel_att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)  # 通道注意力
        x = self.spatial_att(x)  # 空间注意力
        return x




class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DilatedConv, self).__init__()
        self.daconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1,
                                dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LRelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        output = self.daconv(x)
        output = self.bn(output)
        output = self.LRelu(output)
        return output

class Dense2Block(nn.Module):
    def __init__(self, channels, output_channels):
        super(Dense2Block, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(channels, channels)
        self.conv2 = ConvBnLeakyRelu2d(2*channels, channels)
        self.bn = nn.BatchNorm2d(3*channels)
        self.conv_final = nn.Conv2d(3*channels, output_channels, kernel_size=1, padding=0)
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


class MixExtract_Block(torch.nn.Module):
    def __init__(self, in_channels):
        super(MixExtract_Block, self).__init__()
        self.line1 = ConvBnLeakyRelu2d(in_channels, in_channels)
        self.line2 = Dense2Block(in_channels, in_channels)
        self.line3 = DilatedConv(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.line4 = Sobelxy(in_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv_sobel = Conv1BNLRelu(in_channels, in_channels)
        self.conv_final = Conv1BNLRelu(4*in_channels, in_channels)

    def forward(self, x):
        out1 = self.line1(x)
        out2 = self.line2(x)
        out3 = self.line3(x)
        sobel = self.line4(x)
        out4 = self.conv_sobel(self.leaky_relu(sobel))
        out = torch.cat((out1, out2, out3, out4), dim=1)
        output = self.conv_final(out)
        return output


class DeepCA(nn.Module):
    def __init__(self, in_ch):
        super(DeepCA, self).__init__()
        ch_1 = int(in_ch / 4)
        ch_2 = int(in_ch / 16)
        self.Avg = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Max = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.avgconv1 = nn.Conv2d(in_channels=in_ch, out_channels=ch_1, kernel_size=3, padding=1, stride=1)
        self.avgconv2 = nn.Conv2d(in_channels=ch_1, out_channels=ch_2, kernel_size=3, padding=1, stride=1)
        self.avgconv3 = nn.Conv2d(in_channels=ch_2, out_channels=ch_1, kernel_size=3, padding=1, stride=1)
        self.avgconv_final = nn.Conv2d(in_channels=ch_1, out_channels=in_ch, kernel_size=3, padding=1, stride=1)
        self.maxconv1 = nn.Conv2d(in_channels=in_ch, out_channels=ch_1, kernel_size=3, padding=1, stride=1)
        self.maxconv2 = nn.Conv2d(in_channels=ch_1, out_channels=ch_2, kernel_size=3, padding=1, stride=1)
        self.maxconv3 = nn.Conv2d(in_channels=ch_2, out_channels=ch_1, kernel_size=3, padding=1, stride=1)
        self.maxconv_final = nn.Conv2d(in_channels=ch_1, out_channels=in_ch, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input):
        b, c, h, w = input.shape
        avg = self.Avg(input)
        max = self.Max(input)
        avg = self.avgconv1(avg)
        avg = self.avgconv2(avg)
        avg = self.relu(avg)
        avg = self.avgconv3(avg)
        avg = self.avgconv_final(avg)
        avg = F.interpolate(avg, size=(h, w), mode='bilinear', align_corners=False)
        max = self.maxconv1(max)
        max = self.maxconv2(max)
        max = self.relu(max)
        max = self.maxconv3(max)
        max = self.maxconv_final(max)
        max = F.interpolate(max, size=(h, w), mode='bilinear', align_corners=False)
        all_attn = avg + max
        weight = self.sigmoid(all_attn)
        output = weight * input
        return output




class Encoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_block, self).__init__()
        transition_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels, transition_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(transition_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1)
        output = self.conv2(x1)
        output_max = self.maxpool(output)
        return output, output_max


class MixEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixEncoder, self).__init__()
        self.block1 = DeepCA(in_channels)
        # self.block1 = CBAM(in_channels, reduction_ratio=16, kernel_size=7)
        # self.block1 = MixExtract_Block(in_channels)
        self.block2 = Encoder_block(in_channels, out_channels)

    def forward(self, x):
        out = self.block1(x)
        # out = self.block2(out)
        out, out_max = self.block2(out)  # 非单encoder block是需要将输入从x改为out
        return out, out_max



class Decoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_block, self).__init__()
        transition_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, transition_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(transition_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1)
        output = self.conv2(x1)
        return output


class UpsampleReshape(torch.nn.Module):  # 对2个输入宽高比较，用反射填充,将x2上采样到x1大小
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 == 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class Autoencoder(torch.nn.Module):
    def __init__(self, channels_num, in_channels=1, out_channels=1):
        super(Autoencoder, self).__init__()
        ME = MixEncoder
        DE = Decoder_block
        in_num = 16

        self.up_test = UpsampleReshape()
        self.up_train = nn.Upsample(scale_factor=2)
        self.conv0 = ConvLeakyRelu2d(in_channels, in_num, kernel_size=1, padding=0)
        self.Encoder1 = ME(in_num, channels_num[0])
        self.Encoder2 = ME(channels_num[0], channels_num[1])
        self.Encoder3 = ME(channels_num[1], channels_num[2])
        self.Encoder4 = ME(channels_num[2], channels_num[3])
        self.maskpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Decoder1_1 = DE(channels_num[0]+channels_num[1], channels_num[0])
        self.Decoder2_1 = DE(channels_num[1]+channels_num[2], channels_num[1])
        self.Decoder3_1 = DE(channels_num[2]+channels_num[3], channels_num[2])

        self.Decoder1_2 = DE(2*channels_num[0]+channels_num[1], channels_num[0])
        self.Decoder2_2 = DE(2*channels_num[1]+channels_num[2], channels_num[1])
        self.Decoder1_3 = DE(3*channels_num[0]+channels_num[1], channels_num[0])
        self.conv_final = nn.Conv2d(channels_num[0], out_channels, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = torch.nn.Sigmoid()

    def encoder(self, input):
        x = self.conv0(input)
        en1, input2 = self.Encoder1(x)
        en2, input3 = self.Encoder2(input2)
        en3, input4 = self.Encoder3(input3)
        en4, _ = self.Encoder4(input4)
        return [en1, en2, en3, en4]

    def mask_down(self, mask):
        mask1 = mask
        mask2 = self.maskpool(mask)
        mask3 = self.maskpool(mask2)
        mask4 = self.maskpool(mask3)
        return [mask1, mask2, mask3, mask4]

    def decoder_train(self, feature):
        x1_1 = self.Decoder1_1(torch.cat([feature[0], self.up_train(feature[1])], 1))

        x2_1 = self.Decoder2_1(torch.cat([feature[1], self.up_train(feature[2])], 1))
        x1_2 = self.Decoder1_2(torch.cat([feature[0], x1_1, self.up_train(x2_1)], 1))

        x3_1 = self.Decoder3_1(torch.cat([feature[2], self.up_train(feature[3])], 1))
        x2_2 = self.Decoder2_2(torch.cat([feature[1], x2_1, self.up_train(x3_1)], 1))

        x1_3 = self.Decoder1_3(torch.cat([feature[0], x1_1, x1_2, self.up_train(x2_2)], 1))

        output = self.conv_final(x1_3)
        return output

    def decoder_test(self, feature):
        x1_1 = self.Decoder1_1(torch.cat([feature[0], self.up_test(feature[0], feature[1])], 1))

        x2_1 = self.Decoder2_1(torch.cat([feature[1], self.up_test(feature[1], feature[2])], 1))
        x1_2 = self.Decoder1_2(torch.cat([feature[0], x1_1, self.up_test(feature[0], x2_1)], 1))

        x3_1 = self.Decoder3_1(torch.cat([feature[2], self.up_test(feature[2], feature[3])], 1))
        x2_2 = self.Decoder2_2(torch.cat([feature[1], x2_1, self.up_test(feature[1], x3_1)], 1))

        x1_3 = self.Decoder1_3(torch.cat([feature[0], x1_1, x1_2, self.up_test(feature[0], x2_2)], 1))

        output = self.conv_final(x1_3)
        return output


