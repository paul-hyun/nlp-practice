import argparse

import numpy as np

import torch

from transformers import (
    T5TokenizerFast,
)

from seq2seq import Seq2SeqTranslator
from seq2seq_attn import Seq2SeqAttention


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--valid_src_fn", required=True)
    p.add_argument("--gpu_id", type=int, default=-1)

    p.add_argument("--with_attention", action="store_true")

    config = p.parse_args()

    return config


def main(config):
    device = (
        torch.device("cpu")
        if config.gpu_id < 0 or not torch.cuda.is_available()
        else torch.device(f"cuda:{config.gpu_id}")
    )

    src = []
    with open(config.valid_src_fn) as f:
        for line in f:
            if line.strip():
                src.append(line.strip())
            if len(src) >= 10000:
                break
    src = list(np.random.choice(src, 10))

    data = torch.load(config.model_fn, map_location=device)
    train_config = data["config"]

    print(train_config)

    tokenizer = T5TokenizerFast.from_pretrained(train_config.tokenizer)
    tokenizer.bos_token = "<s>"

    if not config.with_attention:
        model = Seq2SeqTranslator(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=train_config.hidden_dim,
            n_layers=train_config.n_layers,
            dropout=train_config.dropout,
            pad_idx=tokenizer.pad_token_id,
        )
    else:
        model = Seq2SeqAttention(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=train_config.hidden_dim,
            n_layers=train_config.n_layers,
            dropout=train_config.dropout,
            pad_idx=tokenizer.pad_token_id,
        )

    model.load_state_dict(data["model"])
    model.eval()
    model.to(device)

    for line in src:
        x = tokenizer(
            line,
            truncation=True,
            max_length=train_config.max_length,
            return_tensors="np",
        )["input_ids"]

        output_ids = model.generate(
            list(x[0]), 50, tokenizer.bos_token_id, tokenizer.eos_token_id
        )
        result = tokenizer.decode(output_ids)
        print(f"- ko: {line}\n- en: {result}\n")


if __name__ == "__main__":
    config = define_config()
    main(config)
