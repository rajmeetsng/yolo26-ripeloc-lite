"""
Lightweight Feature Pyramid Network (LFPN)
Replaces standard convolutions with depthwise separable C3k2 blocks.
"""
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: DW-Conv + PW-Conv."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, pad,
                           groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class DW_C3k2(nn.Module):
    """Depthwise separable C3k2 block for LFPN.
    
    Replaces standard Conv in C3k2 with DepthwiseSeparableConv.
    """
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, e=0.25):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = DepthwiseSeparableConv(in_ch, mid)
        self.cv2 = DepthwiseSeparableConv(in_ch, mid)
        self.cv3 = DepthwiseSeparableConv(2 * mid, out_ch, 1)
        self.bottleneck = nn.Sequential(*[
            nn.Sequential(
                DepthwiseSeparableConv(mid, mid, 3),
                DepthwiseSeparableConv(mid, mid, 3),
            ) for _ in range(n)
        ])
    
    def forward(self, x):
        a = self.cv1(x)
        b = self.bottleneck(self.cv2(x))
        return self.cv3(torch.cat([a, b], dim=1))
