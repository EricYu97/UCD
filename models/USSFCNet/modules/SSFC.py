import torch
import torch.nn as nn
import numpy as np


class SSFC(torch.nn.Module):
    def __init__(self, in_ch):
        super(SSFC, self).__init__()

        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv

    def forward(self, x):
        _, _, h, w = x.size()

        q = x.mean(dim=[2, 3], keepdim=True)
        # k = self.proj(x)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        # print(sigma)

        return x * att_weight

