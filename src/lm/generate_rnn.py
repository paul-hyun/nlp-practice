import argparse

import torch

from transformers import (
    T5TokenizerFast,
)

from rnn import LSTMLanguageModel


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--prompt", required=True)
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

    tokenizer = T5TokenizerFast.from_pretrained(train_config.tokenizer)

    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=train_config.embedding_dim,
        hidden_dim=train_config.hidden_dim,
        n_layers=train_config.n_layers,
        dropout=train_config.dropout,
        pad_idx=tokenizer.pad_token_id,
    )
    model.load_state_dict(data["model"])
    model.eval()
    model.to(device)

    x = tokenizer(
        f"<s>{config.prompt}",
        truncation=True,
        max_length=train_config.max_length,
        return_tensors="np",
    )["input_ids"]

    output_ids = model.generate(list(x[0]), 50, tokenizer.eos_token_id)
    result = tokenizer.decode(output_ids)
    print(result)


if __name__ == "__main__":
    config = define_config()
    main(config)
