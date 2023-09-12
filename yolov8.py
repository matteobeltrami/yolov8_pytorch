import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample:
    def __init__(self, scale_factor, mode="nearest"):
        assert mode == "nearest"
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) > 2 and len(x.shape) <= 5, "Input tensor must have 3 to 5 dimensions"
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return upsampled

class Conv_Block(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut: bool, groups=1, kernels: list = (3,3), channel_factor=0.5):
        super().__init__()
        c_ = int(c2 * channel_factor)
        self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=0)
        self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=0, groups=groups)
        self.residual = c1 == c2 and shortcut
    
    def forward(self, x):
        if self.residual:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, groups=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_Block(c1, 2 * self.c, kernel_size=1)
        self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
        self.bottleneck = [Bottleneck(self.c, self.c, shortcut, groups, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]
        
    def forward(self, x):
        x = self.cv1(x)
        y = list(torch.chunk(x, chunks=2, dim=1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        z = y[0]
        
        # PROBLEMA quando fa il concat ottiene dimensioni diverse tra output del primo conv e output dei bottleneck 
        for i in y[1:]: 
            #print(len(i[0][0]))  debug
            z = torch.cat((z, i), dim=1)
        return self.cv2(z)
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_Block(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv_Block(c_ * 4, c2, kernel_size=1, stride=1, padding=0)
        self.maxpool = lambda x: F.max_pool2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2)), kernel_size=k, stride=1)

    def forward(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        y = torch.cat((x, x2, x3, x4), dim=1)
        return self.cv2(y)
    
""" class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        x = torch.arange(c1)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        self.c1 = c1
        
    def forward(self, x):
        b, c, h, w = x.shape
        y = x.reshape(b, 4, self.c1, a).transpose(2, 1)
        y = F.softmax(y, dim=1)
        y = self.conv(y)
        y = y.rehape(b, 4, a)
        return y """
    
# TEST
input=torch.randn(20, 16, 50, 100)
print("Input: ", input.shape, "\n")

# Upsample
u = Upsample(2)
output = u(input)
print("Upsample: ", output.shape)

# Conv_Block
t = Conv_Block(16, 33, 3, stride=2)
output = t(input)
print("Conv_Block: ", output.shape)

# Bottleneck
b = Bottleneck(16, 33, False)
output = b(input)
print("BottleNeck: ", output.shape)

# SPPF
s = SPPF(16, 33)
output = s(input)
print("SPPF: ", output.shape)

# C2f
c = C2f(16, 33, 0) # check for n!=0
output = c(input)
print("C2F: ", output.shape)