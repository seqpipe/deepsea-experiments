import torch.nn as nn


class NeuralSEA(nn.Module):
    def __init__(self,
                 block: nn.Module,
                 num_blocks: int,
                 chromatin_features=919):

        super(NeuralSEA, self).__init__()

        # Parameters
        self.block = block
        self.num_blocks = num_blocks
        self.chromatin_features = chromatin_features

        # Motifs detector
        self.conv1 = nn.Conv1d(4, 320, kernel_size=25)
        self.bn1 = nn.BatchNorm1d(320)
        self.relu = nn.ReLU(inplace=True)

        # reduction pooling
        self.maxpool = nn.MaxPool1d(4, stride=4)

        # SEA Blocks 1
        self.sea_blocks1 = self._make_layer(self.num_blocks,
                                            320,
                                            320,
                                            stride=1)

        # SEA Blocks 2
        self.sea_blocks2 = self._make_layer(self.num_blocks,
                                            320,
                                            480,
                                            stride=2)

        # SEA Blocks 3
        self.sea_blocks3 = self._make_layer(self.num_blocks,
                                            480,
                                            640,
                                            stride=2)

        # Final Predictor - fully connected
        self.dropout = nn.Dropout(0.5)
        self.predictor = nn.Sequential(
            nn.Linear(640 * 61, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, chromatin_features),
        )

    def _make_layer(self,
                    num_blocks: int,
                    inplanes: int,
                    planes: int,
                    stride=1) -> nn.Sequential:

        layers = []
        layers.append(self.block(inplanes, planes, stride=stride))

        for _ in range(1, num_blocks):
            layers.append(self.block(planes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x's shape: (batch_size, 4, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x's shape: (batch_size, 320, 976)
        x = self.maxpool(x)

        # x's shape: (batch_size, 320, 244)
        x = self.sea_blocks1(x)

        # x's shape: (batch_size, 320, 244)
        x = self.sea_blocks2(x)

        # x's shape: (batch_size, 480, 122)
        x = self.sea_blocks3(x)

        # x's shape: (batch_size, 640, 61)
        x = x.view(x.shape[0], -1)

        # x's shape: (batch_size, 640 * 61)
        x = self.dropout(x)
        x = self.predictor(x)

        return x
