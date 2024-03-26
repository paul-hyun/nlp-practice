import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
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

        convs = []
        convs.append(
            nn.Conv1d(embedding_dim, hidden_dim, 3, padding="same"),
        )
        for _ in range(n_layers - 1):
            convs.append(nn.Conv1d(hidden_dim, hidden_dim, 3, padding="same"))
        self.convs = nn.Sequential(*convs)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # |x| = (batch_size, seq_len)

        embedded = self.dropout(self.embedding(x))
        # |embedded| = (batch_size, seq_len, embedding_dim)

        output = self.convs(embedded.transpose(2, 1)).transpose(2, 1)
        # |output| = (batch_size, seq_len, hidden_dim)

        hidden = self.dropout(output.max(dim=1)[0])
        # |hidden| = (batch_size, hidden_dim)

        logit = self.fc(hidden)
        # |logit| = (batch_size, output_dim)

        return logit

    def compute_loss(self, batch, criterion):
        x, y = batch["input_ids"], batch["labels"]
        # |x| = (batch_size, seq_len)
        # |y| = (batch_size, )

        device = next(self.parameters()).device

        x = x.to(device)
        y = y.to(device)

        logit = self.forward(x)
        # |logit| = (batch_size, output_dim)

        loss = criterion(logit, y)
        # |loss| = (1, )

        return loss
