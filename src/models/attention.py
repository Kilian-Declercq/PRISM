import torch.nn as nn

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.dw_d_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.dw_conv(x)
        attn = self.dw_d_conv(attn)
        attn = self.pw_conv(attn)
        return u * attn

class AttBlockLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1_1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.lka = LKA(dim)
        self.conv2_1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.elu = nn.ELU()

    def forward(self, x):
        skip = x.clone()
        x = self.conv1_1x1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.lka(x)
        x = self.conv2_1x1(x)
        x = self.bn2(x)
        return x + skip