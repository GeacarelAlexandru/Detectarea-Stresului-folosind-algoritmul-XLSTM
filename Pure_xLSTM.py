import torch
import torch.nn as nn

class ExponentialGatedRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c, n, m):
        gates = self.weight_ih(x) + self.weight_hh(h)
        i_tilde, f_tilde, o_gate, z_tilde = gates.chunk(4, 1)
        
        m_new = torch.maximum(f_tilde + m, i_tilde)
       
        i_gate = torch.exp(i_tilde - m_new)
        f_gate = torch.exp(f_tilde + m - m_new)
        
        z = torch.tanh(z_tilde)

        c_new = f_gate * c + i_gate * z
        n_new = f_gate * n + i_gate
        
        h_new = torch.sigmoid(o_gate) * (c_new / (n_new + 1e-6))
        
        return h_new, c_new, n_new, m_new

class Pure_xLSTM(nn.Module):
    """
    Motorul principal xLSTM construit în Pure PyTorch.
    """
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = ExponentialGatedRNNCell(input_size, hidden_size)

    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1) 
        
        seq_len, batch_size, _ = x.size()
        
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        n = torch.zeros(batch_size, self.hidden_size, device=x.device)
        m = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h, c, n, m = self.cell(x[t], h, c, n, m)
            outputs.append(h)
            
        out = torch.stack(outputs)
        
        if self.batch_first:
            out = out.transpose(0, 1) 
            
        return out, None