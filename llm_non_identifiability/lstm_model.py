from torch import nn as nn


class LSTM_LLM(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens: int = 5,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_lstm: float = 0.4,
        device=None,
    ):
        super(LSTM_LLM, self).__init__()

        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_lstm)
        self.fc = nn.Linear(hidden_dim, num_tokens)

    def forward(self, src):
        src = src.to(self.embedding.weight.device)
        embedded = self.embedding(src)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        return out.permute(0, 2, 1)
