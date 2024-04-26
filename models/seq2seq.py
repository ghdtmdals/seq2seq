import torch
from torch import nn

from .lstm import LSTMEncoder, LSTMDecoder

class Seq2Seq(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, source_dict_size = 306926, target_dict_size = 204764):
        super().__init__()
        self.source_dict_size = source_dict_size
        self.target_dict_size = target_dict_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.source_embedding = nn.Embedding(self.source_dict_size, self.input_dim)
        self.target_embedding = nn.Embedding(self.target_dict_size, self.input_dim)

        self.encoder = LSTMEncoder(self.n_layers, self.input_dim, self.hidden_dim)
        self.decoder = LSTMDecoder(self.n_layers, self.input_dim, self.hidden_dim, self.target_dict_size)

    def forward(self, input_sequence, target_sequence):
        input_sequence = self.source_embedding(input_sequence)
        cell_state, hidden_state = self.encoder(input_sequence)

        target_sequence = self.target_embedding(target_sequence)
        outputs = self.decoder(cell_state, hidden_state, target_sequence)

        return outputs

# if __name__ == "__main__":    
#     input_sequence = torch.randint(0, 306926, (8, 7))
#     target_sequence = torch.randint(0, 204764, (8, 10))
#     model = Seq2Seq(n_layers = 4, input_dim = 256, hidden_dim = 512)

#     outputs = model(input_sequence, target_sequence)
    
#     breakpoint()