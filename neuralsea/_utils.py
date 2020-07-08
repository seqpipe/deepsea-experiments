import torch.nn as nn

__all__ = [
    '_conv1d',
]


def _conv1d(inplanes: int, planes: int, k: int, s: int) -> nn.Conv1d:
    ''' 1-dimensional `s` times reduction convolution with kernel size k '''

    return nn.Conv1d(inplanes,
                     planes,
                     kernel_size=k,
                     stride=s,
                     padding=k // 2,
                     bias=False)
