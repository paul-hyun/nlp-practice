import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,  # bidirectional=False for Lanugage Model
            dropout=dropout,
            batch_first=True,  # If False, input shape is (seq_len, batch_size, input_size).
        )

        self.dropout = nn.Dropout(dropout)
        # Note that we use "vocab_size" sence we are prediting vocab.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # |x| = (batch_size, seq_len)

        embedded = self.dropout(self.embedding(x))
        # |embedded| = (batch_size, seq_len, embedding_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # |output| = (batch_size, seq_len, hidden_dim)
        # |hidden| = (n_layers, batch_size, hidden_dim)
        # |cell| = (n_layers, batch_size, hidden_dim)

        hidden = self.dropout(output)
        # |hidden| = (batch_size, seq_len, hidden_dim)

        logit = self.fc(hidden)
        # |logit| = (batch_size, seq_len, vocab_size)

        return logit

    def compute_loss(self, batch, criterion):
        x, y = batch["input_ids"], batch["labels"]
        # |x| = (batch_size, seq_len)
        # |y| = (batch_size, seq_len)

        device = next(self.parameters()).device

        x = x.to(device)
        y = y.to(device)

        logit = self.forward(x)
        # |logit| = (batch_size, seq_len, vocab_size)

        loss = criterion(logit.view(-1, logit.size(-1)), y.view(-1))
        # |loss| = (1, )

        return loss

    def generate(self, input_ids, max_length, eos_token_id):
        device = next(self.parameters()).device
        for i in range(max_length):
            print(f"{i:02d} : {input_ids}")
            x = torch.tensor([input_ids]).to(device)
            logit = self.forward(x)
            pred = logit.argmax(dim=-1)[0, -1].item()
            if pred == eos_token_id:
                break
            input_ids.append(pred)
        return input_ids
