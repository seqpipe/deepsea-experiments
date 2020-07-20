''' Implementation of NeuralSEA
'''

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=75):
        super(_PositionalEncoding, self).__init__()

        # Compute the positional encodings once in a logarithmic scale
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class NeuralSEA(nn.Module):
    ''' NeuralSEA

    A Neural DNA SEquence Analyzer

    '''

    model_type = 'Transformer'
    nbases = 4

    def __init__(
        self,
        d_model=256,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
        activation='relu',
        nlayer=3,
        nlabel=919,
    ):

        super(NeuralSEA, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.nlayer = nlayer
        self.nlabel = nlabel

        # # base2vec: Nucleobases (A, C, G and T) embeddings
        # self.base2vec = nn.Embedding(NeuralSEA.nbases, self.d_model)

        self.conv = nn.Sequential(
            nn.Conv1d(4, self.d_model, kernel_size=26, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool1d(13, stride=13)

        # Positional Encoding
        self.pe = _PositionalEncoding(self.d_model, dropout=self.dropout)

        # NeuralSEA Transformer DNA Encoder Layer
        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation)

        # NeuralSEA Transformer DNA Encoder
        self.encoder = nn.TransformerEncoder(self._encoder_layer, self.nlayer)

        # Fully-connected NN
        self.dropout = nn.Dropout(p=dropout)
        self.fcnn = nn.Sequential(
            nn.Linear(self.d_model * 75, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, self.nlabel),
        )

        self.init_weights()

    def init_weights(self, initrange=0.1):
        # self.base2vec.weight.data.uniform_(-initrange, initrange)
        self.fcnn.bias.data.zero_()
        self.fcnn.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == 4 and x.shape[2] == 1000

        # x's shape: (batch_size, 4, 1000)
        x = self.conv(x)
        x = self.maxpool(x)

        # x's shape: (batch_size, d_model, 75)
        x = x.transpose(0, 2).transpose(1, 2)

        # x's shape: (75, batch_size, d_model)
        x = self.pe(x)

        # x's shape: (75, batch_size, d_model)
        x = self.encoder(x)
        x = x.permute(1, 2, 0)

        # x's shape: (batch_size, d_model, 75)
        x = x.view(x.shape[0], -1)

        # x's shape: (batch_size, d_model * 75)
        x = self.dropout(x)
        x = self.fcnn(x)

        # x's shape: (batch_size, nlabel)
        return x
