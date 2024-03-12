import torch
from torch import nn

### LSTM with 4-layers
### 2 Different LSTMs: 1. Encoder 2. Decoder

class _LSTMBlock(nn.Module):
    ### LSTM Consists of
    ## 2 inputs [h_t: Short-term memory], [c_t: Long-term memory]
    ##
    ## Forget gate: [h_t-1, x_t: input token] -> [concat] -> to [c_t]
    ## Input gate: [h_t-1, x_t] -> [concat] -> [Sigmoid] & [tanh] outputs
    ##
    ## Update Long-term memory (Cell state): [c_t-1] x forget gate + input gate
    ## output gate (update Short-term memory (hidden state)): [h_t-1, x_t] -> [concat] -> [Sigmoid] x [c_t]

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        ### Linear Projection Layers for Forget Gate
        self.forget_xtoh = nn.Linear(input_dim, hidden_dim, bias = True)
        self.forget_htoh = nn.Linear(hidden_dim, hidden_dim, bias = True)

        ### Linear Projection Layers for Input Gate
        self.input_xtoh_sig = nn.Linear(input_dim, hidden_dim, bias = True)
        self.input_htoh_sig = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.input_xtoh_tanh = nn.Linear(input_dim, hidden_dim, bias = True)
        self.input_htoh_tanh = nn.Linear(hidden_dim, hidden_dim, bias = True)

        ### Linear Projection Layers for Output Gate
        self.output_xtoh = nn.Linear(input_dim, hidden_dim, bias = True)
        self.output_htoh = nn.Linear(hidden_dim, hidden_dim, bias = True)

    
    def forget_gate(self, hidden, x):
        ### For Output f_t
        ### Linear Projection
        x_proj = self.forget_xtoh(x)
        hidden_proj = self.forget_htoh(hidden)
        concat = x_proj + hidden_proj

        ### Sigmoid
        f_t = self.sigmoid(concat)
        
        return f_t

    def input_gate(self, hidden, x):
        ### For Output i_t
        ### Linear Projection
        i_x_proj = self.input_xtoh_sig(x)
        i_hidden_proj = self.input_htoh_sig(hidden)
        i_concat = i_x_proj + i_hidden_proj

        ### Sigmoid
        i_t = self.sigmoid(i_concat)

        ### For Output C_tilde_t
        ### Linear Projection
        c_tild_x_proj = self.input_xtoh_tanh(x)
        c_tild_hidden_proj = self.input_htoh_tanh(hidden)
        c_tild_concat = c_tild_x_proj + c_tild_hidden_proj
        
        ### Tanh
        c_tilde_t = self.tanh(c_tild_concat)
        
        return i_t, c_tilde_t

    def update_cell_state(self, cell, f_t, i_t, c_tld_t):
        ### Update Cell state c_t-1
        ### 1. c_t-1 x f_t
        forget_update = cell * f_t

        ### 2. i_t x c_tld_t
        input_update = i_t * c_tld_t

        ### sum 1, 2
        cell_updated = forget_update + input_update

        return cell_updated

    def output_gate(self, hidden, x, c_t):
        ### Update Hidden state h_t-1
        ### Linear Projection
        x_proj = self.output_xtoh(x)
        hidden_proj = self.output_htoh(hidden)
        concat = x_proj + hidden_proj

        ### 1. Sigmoid
        o_t = self.sigmoid(concat)

        ### 2. Tanh on Updated Cell State
        tanh_c_t = self.tanh(c_t)

        ### Multiply 1, 2
        hidden_updated = o_t * tanh_c_t

        return hidden_updated
    
    def forward(self, cell, hidden, x):
        ### Forget Gate
        f_t = self.forget_gate(hidden, x)

        ### Input Gate
        i_t, C_tilde_t = self.input_gate(hidden, x)

        ### Update Cell state
        c_t = self.update_cell_state(cell, f_t, i_t, C_tilde_t)

        ### Update Hidden state
        h_t = self.output_gate(hidden, x, c_t)

        return c_t, h_t

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim = 256, hidden_dim = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = 4 ### Layer Dimension from the Paper

        # self.lstm1 = _LSTMBlock(input_dim = self.input_dim, hidden_dim = self.hidden_dim)
        # self.lstm2 = _LSTMBlock(input_dim = self.hidden_dim, hidden_dim = self.hidden_dim)
        # self.lstm3 = _LSTMBlock(input_dim = self.hidden_dim, hidden_dim = self.hidden_dim)
        # self.lstm4 = _LSTMBlock(input_dim = self.hidden_dim, hidden_dim = self.hidden_dim)
        self.lstm_layers = [_LSTMBlock(input_dim = self.input_dim, hidden_dim = self.hidden_dim)]
        for i in range(3):
            self.lstm_layers.append(_LSTMBlock(input_dim = self.hidden_dim, hidden_dim = self.hidden_dim))

    def forward(self, x):
        assert len(x.shape) == 3
        cell_state = torch.zeros(self.layer_dim, x.size(0), x.size(1), self.hidden_dim)
        hidden_state = torch.zeros(self.layer_dim, x.size(0), x.size(1), self.hidden_dim)

        ### Get Hidden States for the First LSTM Layers
        for seq in range(x.size(1)):
            cell_state[0, :, seq], hidden_state[0, :, seq] = self.lstm_layers[0](cell_state[0, :, seq], hidden_state[0, :, seq], x[:, seq])

        ### Use Hidden States of Lower LSTM Layers for the Rest of the LSTM Layers
        for layer in range(1, len(self.lstm_layers)):
            for seq in range(x.size(1)):
                cell_state[layer, :, seq], hidden_state[layer, :, seq] = self.lstm_layers[layer](cell_state[layer, :, seq], hidden_state[layer, :, seq], hidden_state[layer - 1, :, seq])

        ### Final Hidden State Output at the end of the Sequence will be Passed to the Decoder
        return hidden_state[-1, :, -1]

class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

if __name__ == "__main__":
    test_tensor = torch.randn(8, 4, 256) ### batch, seq, dim
    encoder = LSTMEncoder()
    output = encoder(test_tensor)
    breakpoint()