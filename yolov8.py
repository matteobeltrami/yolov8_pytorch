import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut: bool, groups=1, kernels: list = (3,3), channel_factor=0.5):
        super().__init__()
        c_ = int(c2 * channel_factor)
        self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=0)
        self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=0, groups=groups)
        self.residual = c1 == c2 and shortcut
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, groups=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_Block(c1, 2 * self.c, 1,)
        self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
        self.bottleneck = [Bottleneck(self.c, self.c, shortcut, groups, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]
        
    def forward(self, x):
        #y = list(self.cv1(x).chunk(2, 1))
        y = list(torch.chunk(self.cv1(x), 2, dim=1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        z = y[0]
        #for i in y[1:]: z = z.cat(i, dim=1)
        for i in y[1:]: z = torch.cat((z, i), dim=1)
        return self.cv2(z)
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_Block(c1, c_, 1, 1, padding=0)
        self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=0)
        #self.maxpool = lambda x : x.pad2d((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=k, stride=1)
        self.maxpool = lambda x: F.max_pool2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2)), kernel_size=k, stride=1)

    def forward(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        #return self.cv2(x.cat(x2, x3, x4, dim=1))
        return self.cv2(torch.cat((x, x2, x3, x4), dim=1))
    
# tests

# Conv_Block
input=torch.randn(20, 16, 50, 100)
print(input.shape)
t = Conv_Block(16, 33, 3, stride=2)
output = t(input)
print(output.shape)

# Bottleneck
input=torch.randn(20, 16, 50, 100)
print(input.shape)
b = Bottleneck(16, 33, False)
output = b(input)
print(output.shape)

# SPPF
input=torch.randn(20, 16, 50, 100)
print(input.shape)
s = SPPF(16, 33)
output = s(input)
print(output.shape)

# C2f
input=torch.randn(20, 16, 50, 100)
print(input.shape)
c = C2f(16, 33, 0) # check for n!=0
output = c(input)
print(output.shape)