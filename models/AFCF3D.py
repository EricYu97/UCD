
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import copy

class Decompose_conv(nn.Module):
    def __init__(self, conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False):
        super(Decompose_conv, self).__init__()
        self.time_dim = time_dim
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        if time_dim == 1:
            self.conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_dim, padding=padding,
                                          dilation=dilation, stride=stride)

            weight_2d = conv2d.weight.data
            weight_3d = weight_2d.unsqueeze(2)

            self.conv3d.weight = Parameter(weight_3d)
            self.conv3d.bias = conv2d.bias
        else:
            self.conv3d_spatial = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels,
                                                  kernel_size=(1, kernel_dim[1], kernel_dim[2]),
                                                  padding=(0, padding[1], padding[2]),
                                                  dilation=(1, dilation[1], dilation[2]),
                                                  stride=(1, stride[1], stride[2])
                                                  )
            weight_2d = conv2d.weight.data
            self.conv3d_spatial.weight = Parameter(weight_2d.unsqueeze(2))
            self.conv3d_spatial.bias = conv2d.bias
            self.conv3d_time_1 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_2 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_3 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            torch.nn.init.constant_(self.conv3d_time_1.weight, 0.0)
            torch.nn.init.constant_(self.conv3d_time_3.weight, 0.0)
            torch.nn.init.eye_(self.conv3d_time_2.weight[:, :, 0, 0, 0])
            temp = 1

    def forward(self, x):
        if self.time_dim == 1:
            return self.conv3d(x)
        else:
            x_spatial = self.conv3d_spatial(x)
            T1 = x_spatial[:, :, 0:1, :, :]
            T2 = x_spatial[:, :, 1:2, :, :]
            T1_F1 = self.conv3d_time_2(T1)
            T2_F1 = self.conv3d_time_2(T2)
            T1_F2 = self.conv3d_time_1(T1)
            T2_F2 = self.conv3d_time_3(T2)
            x = torch.cat([T1_F1 + T2_F2, T1_F2 + T2_F1], dim=2)

            return x

def Decompose_norm(batch2d):
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d

def Decompose_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(kernel_dim, padding=padding, dilation=dilation, stride=stride,
                                        ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))

    return pool3d


def inflate_conv(conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False):
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_dim, padding=padding,
                             dilation=dilation, stride=stride)

    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


class ResNet3D(torch.nn.Module):
    def __init__(self, resnet2d):
        super(ResNet3D, self).__init__()
        self.conv1 = Decompose_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = Decompose_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = Decompose_pool(resnet2d.maxpool, time_dim=1, time_padding=0, time_stride=1)

        self.layer1 = Decompose_layer(resnet2d.layer1)
        self.layer2 = Decompose_layer(resnet2d.layer2)
        self.layer3 = Decompose_layer(resnet2d.layer3)
        self.layer4 = Decompose_layer(resnet2d.layer4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x




def Decompose_layer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = Decompose_conv(bottleneck2d.conv1, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn1 = Decompose_norm(bottleneck2d.bn1)

        self.conv2 = Decompose_conv(bottleneck2d.conv2, time_dim=3, time_padding=1,
                                                time_stride=1, center=True)
        self.bn2 = Decompose_norm(bottleneck2d.bn2)

        # self.conv3 = Decompose_conv(bottleneck2d.conv3, time_dim=1, center=True)
        # self.bn3 = Decompose_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = Decompose_downsample(bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = out + residual
        out = self.relu(out)
        return out


def Decompose_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        Decompose_norm(downsample2d[1]))
    return downsample3d

class Netmodel(nn.Module):
    def __init__(self, channel, resnet):
        super(Netmodel, self).__init__()
        self.resnet = ResNet3D(resnet)
        # self.resnet = ResNet18_pure(resnet)
        self.decoder = Unet3PP(channel)

    def forward(self, x1, x2):
        x1=x1.unsqueeze(2)
        x2=x2.unsqueeze(2)
        x = torch.cat([x1, x2], 2)
        size = x.size()[3:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        pred_s = self.decoder(x0, x1, x2, x3, x4)
        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return {"main_predictions":pred_s}


class Unet3PP(nn.Module):
    def __init__(self, channel):
        super(Unet3PP, self).__init__()
        # r50_ch = [64, 256, 512, 1024, 2048]
        r_32ch = [64, 64, 128, 256, 512]
        self.reduction0 = Redection3D(r_32ch[0], channel)
        self.reduction1 = Redection3D(r_32ch[1], channel)
        self.reduction2 = Redection3D(r_32ch[2], channel)
        self.reduction3 = Redection3D(r_32ch[3], channel)
        self.reduction4 = Redection3D(r_32ch[4], channel)

        self.AFCF_fuse = Feature_fusion(channel)

        self.output = Decoder(channel)

    def forward(self, x0, x1, x2, x3, x4):
        x_s0 = self.reduction0(x0)
        x_s1 = self.reduction1(x1)
        x_s2 = self.reduction2(x2)
        x_s3 = self.reduction3(x3)
        x_s4 = self.reduction4(x4)

        x_s0, x_s1, x_s2, x_s3, x_s4 = self.AFCF_fuse(x_s0, x_s1, x_s2, x_s3, x_s4)
        pred_s = self.output(x_s0, x_s1, x_s2, x_s3, x_s4)

        return pred_s


class Redection3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Redection3D, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv3d(in_ch, out_ch, kernel_size=[1, 1, 1]),
            BasicConv3d(out_ch, out_ch, kernel_size=[3, 3, 3], padding=1),
            BasicConv3d(out_ch, out_ch, kernel_size=[3, 3, 3], padding=1)
        )

    def forward(self, x):
        y = self.reduce(x)
        return y


class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, stride=stride),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.conv_bn(x)
        return y


class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        y = self.conv_bn(x)
        return y


class Feature_fusion(nn.Module):
    def __init__(self, channel):
        super(Feature_fusion, self).__init__()
        self.AFCF1 = AFCF1(channel)
        self.AFCF2 = AFCF2(channel)
        self.AFCF3 = AFCF3(channel)

    def forward(self, x0, x1, x2, x3, x4):
        C1 = self.AFCF1(x0, x1)
        C2 = self.AFCF2(x0, x1, x2)
        C3 = self.AFCF2(x1, x2, x3)
        C4 = self.AFCF2(x2, x3, x4)
        C5 = self.AFCF3(x3, x4)

        return C1, C2, C3, C4, C5


class AFCF1(nn.Module):
    def __init__(self, channel):
        super(AFCF1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_down = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_cat = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.SE = CMA_variant(channel * 2, channel * 2, 1)

    def forward(self, x0, x1):
        B, C, T, H, W = x1.size()
        x1_flatten = x1.view(B, C * T, H, W)
        x1_flatten_up = self.upsample(x1_flatten)
        x1_up = x1_flatten_up.view(B, C, T, H * 2, W * 2)
        x1_upconv = self.conv_up(x1_up)

        feat1 = x0 + x1_upconv
        feat1 = self.conv_cat(feat1)

        B, C, T, H, W = feat1.size()
        feat1_flatten = feat1.view(B, C * T, H, W)
        feat1_flatten_SE = self.SE(feat1_flatten)
        feat1_SE = feat1_flatten_SE.view(B, C, T, H, W)

        feat = feat1_SE + x0
        return feat


class AFCF2(nn.Module):
    def __init__(self, channel):
        super(AFCF2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_down = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_cat = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.SE = CMA_variant(channel * 2, channel * 2, 1)

    def forward(self, x0, x1, x2):
        x0_down = self.conv_down(x0)

        B, C, T, H, W = x2.size()
        x2_flatten = x2.view(B, C * T, H, W)
        x2_flatten_up = self.upsample(x2_flatten)
        x2_up = x2_flatten_up.view(B, C, T, H * 2, W * 2)
        x2_upconv = self.conv_up(x2_up)

        feat2 = x0_down + x1 + x2_upconv
        feat2 = self.conv_cat(feat2)
        B, C, T, H, W = feat2.size()
        feat2_flatten = feat2.view(B, C * T, H, W)
        feat2_flatten_SE = self.SE(feat2_flatten)
        feat2_SE = feat2_flatten_SE.view(B, C, T, H, W)

        feat = feat2_SE + x1

        return feat


class AFCF3(nn.Module):
    def __init__(self, channel):
        super(AFCF3, self).__init__()
        self.conv_down = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.conv_cat = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.SE = CMA_variant(channel * 2, channel * 2, 1)

    def forward(self, x3, x4):
        x3_down = self.conv_down(x3)
        feat = x3_down + x4
        feat = self.conv_cat(feat)
        B, C, T, H, W = feat.size()
        feat_flatten = feat.view(B, C * T, H, W)
        feat_flatten_SE = self.SE(feat_flatten)
        feat_SE = feat_flatten_SE.view(B, C, T, H, W)
        feat = feat_SE + x4

        return feat


class CMA_variant(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CMA_variant, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)
        x_w = x_w.permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = identity * x_w * x_h

        return y


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample = BasicConv3d(channel, channel, [3, 3, 3], padding=1)
        self.conv_downsample = BasicConv3d(channel, channel, [1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.conv_cat_3 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat_2 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat_1 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )
        self.conv_cat_0 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [1, 1, 1])
        )

        self.downT3 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=[1, 1, 1]),
            BasicConv3d(channel, channel, [4, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            BasicConv3d(channel, channel, [3, 1, 1])
        )
        self.downT2 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=[1, 1, 1]),
            BasicConv3d(channel, channel, [4, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            BasicConv3d(channel, channel, [3, 1, 1])
        )
        self.downT1 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=[1, 1, 1]),
            BasicConv3d(channel, channel, [4, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            BasicConv3d(channel, channel, [3, 1, 1])
        )
        self.downT0 = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=[1, 1, 1]),
            BasicConv3d(channel, channel, [4, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            BasicConv3d(channel, channel, [3, 1, 1])
        )
        self.downfinal = nn.Sequential(
            BasicConv3d(channel, channel, [3, 3, 3], padding=1),
            BasicConv3d(channel, channel, [4, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            BasicConv3d(channel, channel, [4, 1, 1])
        )
        self.superD1 = BasicConv3d(channel, channel, [2, 1, 1])
        self.superD2 = BasicConv3d(channel, channel, [2, 1, 1])
        self.superD3 = BasicConv3d(channel, channel, [2, 1, 1])
        self.superD4 = BasicConv3d(channel, channel, [2, 1, 1])
        self.superout1 = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )
        self.superout2 = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )
        self.superout3 = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )
        self.superout4 = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )

        self.out = nn.Sequential(
            BasicConv2d(channel, 2 * channel, [1, 1]),
            BasicConv2d(2 * channel, channel, [1, 1]),
            nn.Conv2d(channel, 1, [1, 1])
        )

        self.SE_3 = CMA_variant(10 * channel, 10 * channel, 1)
        self.SE_2 = CMA_variant(10 * channel, 10 * channel, 1)
        self.SE_1 = CMA_variant(10 * channel, 10 * channel, 1)
        self.SE_0 = CMA_variant(10 * channel, 10 * channel, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4):
        x0_down1 = self.conv_downsample(x0)
        x0_down2 = self.conv_downsample(x0_down1)
        x0_down3 = self.conv_downsample(x0_down2)
        x1_down2 = self.conv_downsample(x1)
        x1_down3 = self.conv_downsample(x1_down2)
        x2_down3 = self.conv_downsample(x2)

        # Decoder 3
        B, C4, T4, H4, W4 = x4.size()
        x4_flatten = x4.view(B, C4 * T4, H4, W4)
        x4_flatten_up3 = self.upsample(x4_flatten)
        x4_up = x4_flatten_up3.view(B, C4, T4, H4 * 2, W4 * 2)
        x4_upconv = self.conv_upsample(x4_up)
        residual3 = torch.cat([x4_upconv, x3, x2_down3, x1_down3, x0_down3], dim=2)
        x3_ = self.conv_cat_3(residual3)
        x3_flatten = x3_.view(x3_.shape[0], x3_.shape[1] * x3_.shape[2], x3_.shape[3], x3_.shape[4])
        x3_flatten = self.SE_3(x3_flatten)
        x3_ = x3_flatten.view(x3_.shape[0], x3_.shape[1], x3_.shape[2], x3_.shape[3], x3_.shape[4])
        x3 = residual3 + x3_
        x3 = self.downT3(x3)

        # Decoder 2
        B, C3, T3, H3, W3 = x3.size()
        x3_flatten = x3.view(B, C3 * T3, H3, W3)
        x4_flatten = x4_upconv.view(B, C3 * T3, H3, W3)
        x3_flatten_up = self.upsample(x3_flatten)
        x4_flatten_up = self.upsample(x4_flatten)
        x3_up = x3_flatten_up.view(B, C3, T3, 2 * H3, 2 * W3)
        x4_up = x4_flatten_up.view(B, C3, T3, 2 * H3, 2 * W3)
        x3_upconv = self.conv_upsample(x3_up)
        x4_upconv = self.conv_upsample(x4_up)
        residual2 = torch.cat([x4_upconv, x3_upconv, x2, x1_down2, x0_down2], dim=2)
        x2_ = self.conv_cat_2(residual2)
        x2_flatten = x2_.view(x2_.shape[0], x2_.shape[1] * x2_.shape[2], x2_.shape[3], x2_.shape[4])
        x2_flatten = self.SE_2(x2_flatten)
        x2_ = x2_flatten.view(x2_.shape[0], x2_.shape[1], x2_.shape[2], x2_.shape[3], x2_.shape[4])
        x2 = residual2 + x2_
        x2 = self.downT2(x2)

        # Decoder 1
        B, C2, T2, H2, W2 = x2.size()
        x2_flatten = x2.view(B, C2 * T2, H2, W2)
        x3_flatten = x3_upconv.view(B, C2 * T2, H2, W2)
        x4_flatten = x4_upconv.view(B, C2 * T2, H2, W2)
        x2_flatten_up = self.upsample(x2_flatten)
        x3_flatten_up = self.upsample(x3_flatten)
        x4_flatten_up = self.upsample(x4_flatten)
        x2_up = x2_flatten_up.view(B, C2, T2, 2 * H2, 2 * W2)
        x3_up = x3_flatten_up.view(B, C2, T2, 2 * H2, 2 * W2)
        x4_up = x4_flatten_up.view(B, C2, T2, 2 * H2, 2 * W2)
        x2_upconv = self.conv_upsample(x2_up)
        x3_upconv = self.conv_upsample(x3_up)
        x4_upconv = self.conv_upsample(x4_up)
        residual1 = torch.cat([x4_upconv, x3_upconv, x2_upconv, x1, x0_down1], dim=2)
        x1_ = self.conv_cat_1(residual1)
        x1_flatten = x1_.view(x1_.shape[0], x1_.shape[1] * x1_.shape[2], x1_.shape[3], x1_.shape[4])
        x1_flatten = self.SE_1(x1_flatten)
        x1_ = x1_flatten.view(x1_.shape[0], x1_.shape[1], x1_.shape[2], x1_.shape[3], x1_.shape[4])
        x1 = residual1 + x1_
        x1 = self.downT1(x1)

        # Decoder 0
        B, C1, T1, H1, W1 = x1.size()
        x1_flatten = x1.view(B, C1 * T1, H1, W1)
        x2_flatten = x2_upconv.view(B, C1 * T1, H1, W1)
        x3_flatten = x3_upconv.view(B, C1 * T1, H1, W1)
        x4_flatten = x4_upconv.view(B, C1 * T1, H1, W1)
        x1_flatten_up = self.upsample(x1_flatten)
        x2_flatten_up = self.upsample(x2_flatten)
        x3_flatten_up = self.upsample(x3_flatten)
        x4_flatten_up = self.upsample(x4_flatten)
        x1_up = x1_flatten_up.view(B, C1, T1, 2 * H1, 2 * W1)
        x2_up = x2_flatten_up.view(B, C1, T1, 2 * H1, 2 * W1)
        x3_up = x3_flatten_up.view(B, C1, T1, 2 * H1, 2 * W1)
        x4_up = x4_flatten_up.view(B, C1, T1, 2 * H1, 2 * W1)
        x1_upconv = self.conv_upsample(x1_up)
        x2_upconv = self.conv_upsample(x2_up)
        x3_upconv = self.conv_upsample(x3_up)
        x4_upconv = self.conv_upsample(x4_up)
        residual0 = torch.cat([x0, x4_upconv, x3_upconv, x2_upconv, x1_upconv], dim=2)
        x0_ = self.conv_cat_0(residual0)
        x0_flatten = x0_.view(x0_.shape[0], x0_.shape[1] * x0_.shape[2], x0_.shape[3], x0_.shape[4])
        x0_flatten = self.SE_0(x0_flatten)
        x0_ = x0_flatten.view(x0_.shape[0], x0_.shape[1], x0_.shape[2], x0_.shape[3], x0_.shape[4])
        x0 = residual0 + x0_
        x0 = self.downfinal(x0)
        y = x0.squeeze(2)

        out = self.out(y)
        out = self.sigmoid(out)

        return out
    
if __name__=="__main__":
    resnet = torchvision.models.resnet18(pretrained=True)
    # model = CD3D_Net(32, copy.deepcopy(resnet))
    model = Netmodel(32, copy.deepcopy(resnet))
    # model=Netmodel()
    device=torch.device("cpu")
    model=model.to(device)
    a=torch.rand(8,3,256,256).to(device)
    b=torch.rand(8,3,256,256).to(device)
    c=model(a,b)
    print(c.shape)