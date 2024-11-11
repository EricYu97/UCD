import torch.nn as nn
import torch
from .switchable_norm import SwitchNorm2d


class mixer(nn.Module):
    def __init__(self, dim):
        super(mixer, self).__init__()

        self.depthconv = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.gn1 = SwitchNorm2d(dim)

        self.pointconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gn2 = SwitchNorm2d(dim)

        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x

        x = self.depthconv(x)
        x = self.gn1(x)
        x = self.gelu(x)

        x = x + shortcut
        x = self.pointconv(x)
        x = self.gn2(x)
        x = self.gelu(x)

        return x


class up_sampling(nn.Module):
    def __init__(self, in_ch, out_ch, stride=8):
        super(up_sampling, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            SwitchNorm2d(out_ch),
            nn.GELU(),
        )

        dim = out_ch
        self.patchup = nn.ConvTranspose2d(dim, dim, kernel_size=stride, stride=stride)
        self.bn2 = SwitchNorm2d(dim)

        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.layer1(x)

        x = self.patchup(x)
        x = self.bn2(x)
        output = self.gelu(x)

        return output


class RDPNet(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch=384):
        super(RDPNet, self).__init__()
        depth = 32

        self.patchEmb = nn.Conv2d(in_ch * 2, hid_ch, kernel_size=8, stride=8)
        self.gn1 = SwitchNorm2d(hid_ch)

        self.mixer1 = mixer(hid_ch)
        self.ch1 = up_sampling(hid_ch, depth)
        self.mixer2 = mixer(hid_ch)
        self.ch2 = up_sampling(hid_ch, depth)
        self.mixer3 = mixer(hid_ch)
        self.ch3 = up_sampling(hid_ch, depth)
        self.mixer4 = mixer(hid_ch)
        self.ch4 = up_sampling(hid_ch, depth)
        self.mixer5 = mixer(hid_ch)
        self.ch5 = up_sampling(hid_ch, depth)
        self.mixer6 = mixer(hid_ch)
        self.ch6 = up_sampling(hid_ch, depth)

        self.weight = nn.Parameter(torch.randn(1, depth * 6, 1, 1))

        self.final = nn.Conv2d(depth * 6, out_ch, kernel_size=1)

        self.gelu = nn.GELU()

    def forward(self, a, b):
        x = torch.cat([a, b], 1)
        x = self.patchEmb(x)
        x = self.gn1(x)
        x = self.gelu(x)

        x = self.mixer1(x)
        ch1 = self.ch1(x)
        x = self.mixer2(x)
        ch2 = self.ch2(x)
        x = self.mixer3(x)
        ch3 = self.ch3(x)
        x = self.mixer4(x)
        ch4 = self.ch4(x)
        x = self.mixer5(x)
        ch5 = self.ch5(x)
        x = self.mixer6(x)
        ch6 = self.ch6(x)

        out = torch.cat([ch1, ch2, ch3, ch4, ch5, ch6], 1)
        out = out * self.weight
        out = self.final(out)

        return {"main_predictions": out}
if __name__=="__main__":
      
    net = RDPNet(3,2)
    a=torch.rand(2,3,256,256)
    a1=torch.rand(2,3,256,256)
    prob  = net(a,a1)
    print(prob.shape)