''' Implementation of NeuralSEA
'''

import torch.nn as nn

from neuralsea._se_block import _SEBlock


class NeuralSEA(nn.Module):
    ''' NeuralSEA neural network

    A Neural SEquence Analyzer

    '''
    def __init__(self, num_motifs=320, num_labels=919):
        super(NeuralSEA, self).__init__()

        self.num_motifs = num_motifs
        self.num_labels = num_labels

        # Whole Motifs Scanner
        self.motifs_scanner = nn.Sequential(
            nn.Conv1d(4,
                      self.num_motifs,
                      kernel_size=19,
                      stride=1,
                      padding=19 // 2,
                      bias=False),
            nn.BatchNorm1d(self.num_motifs),
            nn.ReLU(inplace=True),
        )

        # 25x max pooling reduction - forcing whole motifs learning
        self.maxpool = nn.MaxPool1d(25, stride=25)

        # SE-MSR
        # Motifs Significance Recalibration through Squeeze-and-Excitation
        self.se_msr_dropout = nn.Dropout(0.2)
        self.se_msr = _SEBlock(self.num_motifs)

        # Regulatory "Grammar" LSTM
        self.rg_lstm = nn.LSTM(input_size=self.num_motifs,
                               hidden_size=self.num_motifs,
                               num_layers=2,
                               batch_first=True,
                               dropout=0.5,
                               bidirectional=True)

        # Final classifier - fully connected
        self.classifier = nn.Sequential(
            nn.Linear(self.num_motifs * 2 * 40, self.num_motifs * 10),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_motifs * 10, self.num_labels),
        )

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == 4 and x.shape[2] == 1000

        # x's shape: (batch_size, 4, 1000)
        x = self.motifs_scanner(x)

        # x's shape: (batch_size, num_motifs, 1000)
        x = self.maxpool(x)

        # x's shape: (batch_size, num_motifs, 40)
        x = self.se_msr_dropout(x)
        x = self.se_msr(x)

        # x's shape: (batch_size, num_motifs, 40)
        x = x.transpose(1, 2)

        # x's shape: (batch_size, 40 `seq_len `, num_motifs `features`)
        x, _ = self.rg_lstm(x)  # h0 and c0 are initialized to zero

        # x's shape: (batch_size, 40, num_motifs * 2)
        x = x.contiguous().view(x.shape[0], -1)

        # x's shape: (batch_size, num_motifs * 2 * 40)
        x = self.classifier(x)

        return x
