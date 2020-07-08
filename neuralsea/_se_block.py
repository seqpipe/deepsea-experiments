import torch.nn as nn


class _SEBlock(nn.Module):
    ''' Squeeze-and-Excitation (SE) Block

    SE block to perform feature recalibration - a mechanism that allows
    the network to perform feature recalibration, through which it can
    learn to use global information to selectively emphasise informative
    features and suppress less useful ones

    Reference: https://arxiv.org/abs/1709.01507

    '''
    def __init__(self, channels: int, reduction_ratio=16):
        super(_SEBlock, self).__init__()

        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Squeezer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Exciter
        self.bottleneck = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
        )

        # Normalizer
        self.normalize = nn.Sequential(
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == self.channels

        # Squeeze: Global Information Embedding
        # output shape: (batch_size, channels)
        squeezed = self.avgpool(x).squeeze()

        # Excitation: Adaptive Feature Recalibration
        # Linear (Bottleneck) -> ReLU
        # output shape: (batch_size, channels // reduction_ratio)
        excited = self.bottleneck(squeezed)

        # Normalization: Normalized Scale for Recalibration
        # Linear (Decoder) -> Sigmoid
        # output shape: (batch_size, channels, 1)
        normalized_scale = self.normalize(excited).unsqueeze(-1)

        return x * normalized_scale
