import torch
from torch import nn

from lstm import LSTMEncoder, LSTMDecoder

class Seq2Seq(nn.Module):
    def __init__(self, dict_size, n_layers, input_dim, hidden_dim):
        super().__init__()
        self.dict_size = dict_size
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = LSTMEncoder(self.n_layers, self.input_dim, self.hidden_dim)
        self.decoder = LSTMDecoder(self.n_layers, self.input_dim, self.hidden_dim, self.dict_size)

    def forward(self, input_sequence, target_sequence):
        cell_state, hidden_state = self.encoder(input_sequence)
        outputs = self.decoder(cell_state, hidden_state, target_sequence)

        return outputs

if __name__ == "__main__":    
    input_sequence = torch.randn(8, 4, 256)
    target_sequence = torch.randn(8, 7, 256)
    model = Seq2Seq(dict_size = 10000, n_layers = 4, input_dim = 256, hidden_dim = 512)

    outputs = model(input_sequence, target_sequence)
    
    breakpoint()