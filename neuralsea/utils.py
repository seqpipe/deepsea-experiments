import torch.nn as nn

__all__ = [
    'conv1d',
]


def conv1d(inplanes: int, planes: int, k: int, s: int) -> nn.Conv1d:
    ''' 1-dimensional reduction convolution with kernel size k '''

    return nn.Conv1d(inplanes,
                     planes,
                     kernel_size=k,
                     stride=s,
                     padding=k // 2,
                     bias=False)
