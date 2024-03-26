import os
import argparse
import json
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


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

    with open(os.path.join(config.model_fn, "..", "config.json")) as f:
        data = json.loads(f.read())

    train_config = argparse.Namespace(**data["config"])
    label2idx = data["label2idx"]
    idx2label = {int(k): v for k, v in data["idx2label"].items()}

    model = AutoModelForSequenceClassification.from_pretrained(config.model_fn)
    tokenizer = AutoTokenizer.from_pretrained(config.model_fn)

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
            ).to(device)

            logit = model(**x).logits[0]
            prob = F.softmax(logit, dim=-1)
            # |prob| = (batch_size, output_dim)

            y = prob.argmax(dim=-1)
            # |y| = (batch_size,)

            print(f"{idx2label[y.item()]}\t{prob[y].item():.4f}\t{line}")


if __name__ == "__main__":
    config = define_config()
    main(config)
