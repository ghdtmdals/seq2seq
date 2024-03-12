import torch
from torch import nn

from lstm import LSTMEncoder, LSTMDecoder

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass