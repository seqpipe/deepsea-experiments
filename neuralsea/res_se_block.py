import torch.nn as nn

from neuralsea.se_block import SEBlock
from neuralsea.utils import conv1d


class ResSEBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int):
        super(ResSEBlock, self).__init__()

        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                conv1d(inplanes, planes, k=1, s=stride),
                nn.BatchNorm1d(planes),
            )

        self.conv1 = conv1d(inplanes, planes, k=5, s=stride)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = conv1d(planes, planes, k=5, s=1)
        self.bn2 = nn.BatchNorm1d(planes)

        self.se_block = SEBlock(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se_block(out)

        out += identity
        out = self.relu(out)

        return out
