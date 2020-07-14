''' Implementation of NeuralSEA
'''

import torch.nn as nn

from neuralsea._se_block import _SEBlock
from neuralsea._time_distributed import _TimeDistributed


class NeuralSEA(nn.Module):
    ''' NeuralSEA

    A Neural DNA SEquence Analyzer

    '''
    def __init__(self, num_motifs=320, num_labels=919):
        super(NeuralSEA, self).__init__()

        self.num_motifs = num_motifs
        self.num_labels = num_labels

        # Whole Motifs Scanner
        self.motifs_scanner = nn.Sequential(
            nn.Conv1d(4, self.num_motifs, kernel_size=26, bias=False),
            nn.BatchNorm1d(self.num_motifs),
            nn.ReLU(inplace=True),
        )

        # SE-MSR
        # Motifs Significance Recalibration through Squeeze-and-Excitation
        self.se_msr_dropout = nn.Dropout(0.2)
        self.se_msr = _SEBlock(self.num_motifs)

        # 13x max pooling reduction - forcing whole motifs learning
        self.maxpool = nn.MaxPool1d(13, stride=13)

        # Regulatory "Grammar" LSTM
        self.rg_lstm_dropout = nn.Dropout(0.2)
        self.rg_lstm = nn.LSTM(input_size=self.num_motifs,
                               hidden_size=self.num_motifs,
                               bidirectional=True,
                               batch_first=True)

        # Time-distributed Fully-connected NN
        self.tdistributed_fcnn = _TimeDistributed(nn.Sequential(
            nn.Linear(self.num_motifs * 2, self.num_motifs),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_motifs, self.num_motifs // 2),
            nn.ReLU(inplace=True),
        ),
                                                  batch_first=True)

        # Final classifier - Fully-connected NN
        self.classifier_dropout = nn.Dropout(0.2)
        num_hidden_neurons = (
            (self.num_motifs // 2) * 75 + self.num_labels) // 2
        self.classifier = nn.Sequential(
            nn.Linear((self.num_motifs // 2) * 75, num_hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_neurons, self.num_labels),
        )

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == 4 and x.shape[2] == 1000

        # x's shape: (batch_size, 4, 1000)
        x = self.motifs_scanner(x)

        # x's shape: (batch_size, num_motifs, 975)
        x = self.se_msr_dropout(x)
        x = self.se_msr(x)

        # x's shape: (batch_size, num_motifs, 975)
        x = self.maxpool(x)

        # x's shape: (batch_size, num_motifs, 75)
        x = x.transpose(1, 2)

        # x's shape: (batch_size, 75 `seq_len `, num_motifs `features`)
        x = self.rg_lstm_dropout(x)
        x, _ = self.rg_lstm(x)  # h0 and c0 are initialized to zero

        # x's shape: (batch_size, 75, num_motifs * 2)
        x = self.tdistributed_fcnn(x)

        # x's shape: (batch_size, 75, num_motifs // 2)
        x = x.view(x.shape[0], -1)

        # x's shape: (batch_size, (num_motifs // 2) * 75)
        x = self.classifier_dropout(x)
        x = self.classifier(x)

        # x's shape: (batch_size, num_labels)
        return x
