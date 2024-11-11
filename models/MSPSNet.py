"""
MSPSNet
please Cited the paper:
Q. Guo, J. Zhang, S. Zhu, C. Zhong, and Y. Zhang.
"Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection", IEEE Geoscience and Remote Sensing, early access, 2022.
"""

import torch
import torch.nn as nn


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Conv_CAM_Layer(nn.Module):

    def __init__(self, in_ch, out_in,use_pam=False):
        super(Conv_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            CAM_Module(32),
            nn.Conv2d(32, out_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_in),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)



class FEC(nn.Module):
    """feature extraction cell"""

    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class FEBlock1(nn.Module):
    """MSPSNet"""
    def __init__(self, in_ch=3, ou_ch=2):
        super(FEBlock1, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 32, 64, 128, 256, 512

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(600, filters[1], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, groups=filters[0] // 2,dilation=1)
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, groups=filters[0] // 2,dilation=2)
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, groups=filters[0] // 2,dilation=3)
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, groups=filters[0] // 2,dilation=4)
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)

        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, groups=filters[1] // 2, dilation=1)
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, groups=filters[1] // 2, dilation=2)
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, groups=filters[1] // 2, dilation=3)
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, groups=filters[1] // 2, dilation=4)
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)

        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, groups=filters[2] // 2, dilation=1)
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, groups=filters[2] // 2, dilation=2)
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, groups=filters[2] // 2, dilation=3)
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, groups=filters[2] // 2, dilation=4)
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)

        self.conv6_4_1 = nn.Conv2d(filters[3]*2, filters[3], padding=1, kernel_size=3, groups=filters[3]//2, dilation=1)
        self.conv6_4_2 = nn.Conv2d(filters[3]*2, filters[3], padding=2, kernel_size=3, groups=filters[3]//2, dilation=2)
        self.conv6_4_3 = nn.Conv2d(filters[3]*2, filters[3], padding=3, kernel_size=3, groups=filters[3]//2, dilation=3)
        self.conv6_4_4 = nn.Conv2d(filters[3]*2, filters[3], padding=4, kernel_size=3, groups=filters[3]//2, dilation=4)
        self.conv4_1 = nn.Conv2d(filters[3]*4, filters[3], kernel_size=1, stride=1)


        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        # The first branch

        x1 = self.conv0(self.conv0_0(x1)) # Output of the first scale
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        # The second branch

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        c4 = (self.cam_attention_4(self.conv4_1(torch.cat([self.conv6_4_1(torch.cat([A_F4, A_F8], 1)), self.conv6_4_2(torch.cat([A_F4, A_F8], 1)), self.conv6_4_3(torch.cat([A_F4, A_F8], 1)), self.conv6_4_4(torch.cat([A_F4, A_F8], 1))], 1))))

        c3 = torch.cat([(self.cam_attention_3(self.conv3_1(torch.cat([self.conv6_3_1(torch.cat([x4, x6], 1)), self.conv6_3_2(torch.cat([x4, x6], 1)), self.conv6_3_3(torch.cat([x4, x6], 1)), self.conv6_3_4(torch.cat([x4, x6], 1))], 1)))), self.Up1(c4)], 1)
        c2 = torch.cat([(self.cam_attention_2(self.conv2_1(torch.cat([self.conv6_2_1(torch.cat([x3, x5], 1)), self.conv6_2_2(torch.cat([x3, x5], 1)), self.conv6_2_3(torch.cat([x3, x5], 1)), self.conv6_2_4(torch.cat([x3, x5], 1))], 1)))), self.Up1(c3)], 1)
        c1 = torch.cat([(self.cam_attention_1(self.conv1_1(torch.cat(
            [self.conv6_1_1(torch.cat([x1, x2], 1)), self.conv6_1_2(torch.cat([x1, x2], 1)),
             self.conv6_1_3(torch.cat([x1, x2], 1)), self.conv6_1_4(torch.cat([x1, x2], 1))], 1)))), self.Up1(c2)], 1)
        c1 = self.conv6(c1)
        out1 = self.conv7(c1)

        # feature_map = out1.squeeze(0)  # [1, 64, 112, 112] -> [64, 112, 112]
        #
        # feature_map_num = feature_map.shape[0]  # 返回通道数
        # for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
        #     single_dim = feature_map[index]  # shape[256, 256]
        #     single_dim = single_dim.cpu().numpy()
        #     if index == 2:
        #         plt.imshow(single_dim, cmap='hot')
        #     # plt.imshow(single_dim, cmap='viridis')
        #         plt.axis('off')
        #         plt.show()

        return {"main_predictions": out1}
    
if __name__=="__main__":
    net=FEBlock1(in_ch=3,ou_ch=2).cuda()
    out=net(torch.rand((2,3,256,256)).cuda(),torch.rand((2,3,256,256)).cuda())
    print(out["main_predictions"].shape)