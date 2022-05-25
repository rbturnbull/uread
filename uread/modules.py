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
        tensor = self.rnn(tensor)
        tensor = self.get_logits(tensor)
        return tensor
