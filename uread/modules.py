
import torch
from torch import nn


class CharDecoder(nn.Module):
    def __init__(
        self,
        input_size:int,
        vocab_size:int,
        hidden_size:int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
            bias=True,
            batch_first=True,
        )
        self.get_logits = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

    def forward(self, tensor):
        batch_size = tensor.shape[0]
        c_0 = torch.zeros( (1, batch_size, self.hidden_size), device=tensor.device )
        h_0 = torch.zeros_like(c_0)
        
        seq_len = 72 # hack
        repeated = torch.unsqueeze(tensor, dim=1).expand((-1,seq_len,-1))

        tensor, _ = self.rnn(repeated, (h_0, c_0))
        tensor = self.get_logits(tensor)

        tensor = torch.swapaxes(tensor, 1, 2)
        return tensor
