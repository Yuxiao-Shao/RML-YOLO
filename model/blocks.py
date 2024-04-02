import torch.nn.functional as F
from utils.basicblocks import *


class Bottleneck_att(nn.Module):

    def __init__(self, c1, c2, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cra = CRA_neck(c_)
        self.sra = SRA(c_)

    def forward(self, x):
        return self.cv2(self.sra(self.cv1(x)) + self.cra(self.cv1(x)))


class RMLBlock(nn.Module):
    pass


class SRA(nn.Module):
    pass


class CRA(nn.Module):
    pass


class CRA_neck(nn.Module):

    def __init__(self, c1):
        super().__init__()
        self.c1 = c1
        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        w = F.softmax(self.conv1(self.advavg(x)), dim=1)
        w = self.conv(w)
        x = w * x
        return x


class C2f(nn.Module):

    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_att(self.c, self.c, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Concat_BiFPN(nn.Module):

    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)