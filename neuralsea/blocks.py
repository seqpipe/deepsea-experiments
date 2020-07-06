import torch.nn as nn

from neuralsea._se_block import _SEBlock
from neuralsea._utils import _conv1d

__all__ = [
    'BasicSEBlock',
    'ResSEBlock',
]

# ====================================================================================================


class BasicSEBlock(nn.Module):
    ''' Base Convolutional Squeeze-and-Excitation block

     - One convolutional layer with kernel size 5
         Reduces the input size `stride` times
     - Batch Normalization
     - Squeeze-and-Excitation
     - ReLU

    '''
    def __init__(self, inplanes: int, planes: int, stride: int):
        super(BasicSEBlock, self).__init__()

        self.conv = _conv1d(inplanes, planes, k=5, s=stride)
        self.bn = nn.BatchNorm1d(planes)
        self.se_block = _SEBlock(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.se_block(x)
        x = self.relu(x)

        return x


# ====================================================================================================


class ResSEBlock(nn.Module):
    ''' Residual Squeeze-and-Excitation block

     - Two convolutional layers with kernel size 5
         Reduces the input size `stride` times
     - Batch Normalization
     - Squeeze-and-Excitation
     - ReLU
     - Skip-connection (residual)
     - Downsampler if needed

    '''
    def __init__(self, inplanes: int, planes: int, stride: int):
        super(ResSEBlock, self).__init__()

        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                _conv1d(inplanes, planes, k=1, s=stride),
                nn.BatchNorm1d(planes),
            )

        self.conv1 = _conv1d(inplanes, planes, k=5, s=stride)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = _conv1d(planes, planes, k=5, s=1)
        self.bn2 = nn.BatchNorm1d(planes)

        self.se_block = _SEBlock(planes)
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


# ====================================================================================================
