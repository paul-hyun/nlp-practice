import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqAttention(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=pad_idx,
        )

        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=n_layers,
            bidirectional=True,  # bidirectional=True for Encoder
            dropout=dropout,
            batch_first=True,  # If False, input shape is (seq_len, batch_size, input_size).
        )

        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,  # encoder bidirectional=True, decode bidirectional=False
            num_layers=n_layers,
            bidirectional=False,  # bidirectional=False for Decoder (LM)
            dropout=dropout,
            batch_first=True,  # If False, input shape is (seq_len, batch_size, input_size).
        )

        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.concat_w = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        # Note that we use "vocab_size" sence we are prediting vocab.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_x, dec_x, attention_mask):
        # |enc_x| = (batch_size, seq_len_enc)
        # |dec_x| = (batch_size, seq_len_dec)
        # |attention_mask| = (batch_size, seq_len_enc)

        enc_embed = self.dropout(self.embedding(enc_x))
        # |enc_embed| = (batch_size, seq_len_enc, embedding_dim)

        enc_out, (hidden, cell) = self.encoder(enc_embed)
        # |enc_out| = (batch_size, seq_len_enc, hidden_dim)
        # |hidden| = (n_layers * 2, batch_size, hidden_dim // 2)
        # |cell| = (n_layers * 2, batch_size, hidden_dim // 2)

        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=-1)
        # |hidden| = (n_layers, batch_size, hidden_dim)
        cell = torch.cat((cell[0::2], cell[1::2]), dim=-1)
        # |hidden| = (n_layers, batch_size, hidden_dim)

        dec_embed = self.dropout(self.embedding(dec_x))
        # |dec_embed| = (batch_size, seq_len_dec, embedding_dim)

        dec_out, (hidden, cell) = self.decoder(dec_embed, (hidden, cell))
        # |dec_out| = (batch_size, seq_len_dec, hidden_dim)
        # |hidden| = (n_layers, batch_size, hidden_dim)
        # |cell| = (n_layers, batch_size, hidden_dim)

        hidden = self.dot_product_attention(dec_out, enc_out, enc_out, attention_mask)
        # |hidden| = (batch_size, seq_len_dec, hidden_dim)

        hidden = self.dropout(hidden)
        # |hidden| = (batch_size, seq_len_dec, hidden_dim * 2)

        logits = self.fc(hidden)
        # |logits| = (batch_size, seq_len_dec, vocab_size)

        return logits

    def dot_product_attention(self, Q, K, V, attention_mask):
        # |Q| = (batch_size, Q_len, hidden_dim)
        # |K| = (batch_size, K_len, hidden_dim)
        # |V| = (batch_size, K_len, hidden_dim)
        # |attention_mask| = (batch_size, K_len)

        Q = self.attn_w(Q)
        # |Q| = (batch_size, Q_len, hidden_dim)
        attn_score = torch.matmul(Q, K.transpose(-2, -1).contiguous())
        # |attn_score| = (batch_size, Q_len, K_len)
        attention_mask = attention_mask.unsqueeze(1)
        # |attention_mask| = (batch_size, 1, K_len)
        attn_score -= (1 - attention_mask) * 1e9
        # |attn_score| = (batch_size, Q_len, K_len)
        attn_prob = F.softmax(attn_score, dim=-1)
        # |attn_prob| = (batch_size, Q_len, K_len)
        attn_out = torch.matmul(attn_prob, V)
        # |attn_out| = (batch_size, Q_len, hidden_dim)
        hidden = torch.cat([Q, attn_out], dim=-1)
        # |hidden| = (batch_size, Q_len, hidden_dim * 2)
        hidden = self.concat_w(hidden)
        hidden = F.tanh(hidden)
        # |hidden| = (batch_size, Q_len, hidden_dim)
        return hidden

    def compute_loss(self, batch, criterion):
        enc_x = batch["input_ids"]
        dec_x = batch["decoder_input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        # |enc_x| = (batch_size, seq_len_enc)
        # |dec_x| = (batch_size, seq_len_dec)
        # |attention_mask| = (batch_size, seq_len_enc)
        # |y| = (batch_size, seq_len_dec)

        device = next(self.parameters()).device

        enc_x = enc_x.to(device)
        dec_x = dec_x.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        logits = self.forward(enc_x, dec_x, attention_mask)
        # |logit| = (batch_size, seq_len_dec, vocab_size)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        # |loss| = (1, )

        return loss

    def generate(self, enc_ids, max_length, bos_token_id, eos_token_id):
        device = next(self.parameters()).device
        enc_x = torch.tensor([enc_ids]).to(device)
        attention_mask = torch.ones_like(enc_x)
        dec_ids = [bos_token_id]
        for _ in range(max_length):
            dec_x = torch.tensor([dec_ids]).to(device)
            logits = self.forward(enc_x, dec_x, attention_mask)
            pred = logits.argmax(dim=-1)[0, -1].item()
            if pred == eos_token_id:
                break
            dec_ids.append(pred)
        return dec_ids[1:]
