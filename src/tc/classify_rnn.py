import argparse
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
)

from rnn import LSTMClassifier


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--test_tsv_fn", required=True)
    p.add_argument("--device", type=int, default=-1)

    config = p.parse_args()

    return config


def main(config):
    device = (
        torch.device("cpu")
        if config.device < 0 or not torch.cuda.is_available()
        else torch.device(f"cuda:{config.device}")
    )

    data = torch.load(config.model_fn, map_location=device)
    train_config = data["config"]
    label2idx = data["label2idx"]
    idx2label = data["idx2label"]

    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)

    model = LSTMClassifier(
        vocab_size=len(tokenizer),
        embedding_dim=train_config.embedding_dim,
        hidden_dim=train_config.hidden_dim,
        output_dim=len(label2idx),
        n_layers=train_config.n_layers,
        dropout=train_config.dropout,
        pad_idx=tokenizer.pad_token_id,
    )
    model.load_state_dict(data["model"])
    model.eval()
    model.to(device)

    df_test = pd.read_csv(config.test_tsv_fn, sep="\t").sample(10)
    with torch.no_grad():
        for i, row in df_test.iterrows():
            line = row.document.strip()
            if not line:
                continue

            x = tokenizer(
                line,
                truncation=True,
                max_length=train_config.max_length,
                return_tensors="pt",
            )["input_ids"]
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            # |x| = (batch_size, seq_len)

            x = x.to(device)

            logit = model(x)[0]
            prob = F.softmax(logit, dim=-1)
            # |prob| = (batch_size, output_dim)

            y = prob.argmax(dim=-1)
            # |y| = (batch_size,)

            print(f"{idx2label[y.item()]}\t{prob[y].item():.4f}\t{line}")


if __name__ == "__main__":
    config = define_config()
    main(config)
