import torch.nn as nn


class _TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first=False):
        super(_TimeDistributed, self).__init__()

        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.shape) <= 2:
            return self.module(x)

        # Squash batch_size and timesteps into a single axis
        # output_shape: (batch_size * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.shape[-1])

        y = self.module(x_reshape)

        # Reshape y
        if self.batch_first:
            # output shape: (batch_size, timesteps, output_size)
            y = y.contiguous().view(x.shape[0], -1, y.shape[-1])
        else:
            # output shape: (timesteps, batch_size, output_size)
            y = y.view(-1, x.shape[1], y.shape[-1])

        return y
