import torch.nn as nn


class _SEBlock(nn.Module):
    ''' Squeeze-and-Excitation (SE) Block

    SE block to perform feature recalibration - a mechanism that allows
    the network to perform feature recalibration, through which it can
    learn to use global information to selectively emphasise informative
    features and suppress less useful ones

    '''
    def __init__(self, planes: int, reduction_ratio=16):
        super(_SEBlock, self).__init__()

        # Squeezer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Exciter
        self.bottleneck = nn.Sequential(
            nn.Linear(planes, planes // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
        )

        # Normalizer
        self.normalize = nn.Sequential(
            nn.Linear(planes // reduction_ratio, planes, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Squeeze: Global Information Embedding
        squeezed = self.avgpool(x).squeeze()  # Shape: (batch_size, planes)

        # Excitation: Adaptive Feature Recalibration
        # Linear (Bottleneck) -> ReLU
        excitation = self.bottleneck(squeezed)
        # Shape: (batch_size, planes // reduction_ratio)

        # Linear -> Sigmoid
        scale = self.normalize(excitation).unsqueeze(-1)
        # Shape: (batch_size, planes, 1)

        return x * scale
