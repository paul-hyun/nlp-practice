import os
import sys
import argparse
from datetime import datetime

import wandb

import torch

from transformers import (
    T5TokenizerFast,
)

from dataset import (
    TranslationDatatset,
    TranslationCollator,
)
from seq2seq import Seq2SeqTranslator
from seq2seq_attn import Seq2SeqAttention

sys.path.append("../")
from common.trainer import Trainer


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", required=True)
    p.add_argument("--train_src_fn", required=True)
    p.add_argument("--train_tgt_fn", required=True)
    p.add_argument("--valid_src_fn", required=True)
    p.add_argument("--valid_tgt_fn", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--gpu_id", type=int, default=-1)

    p.add_argument("--with_attention", action="store_true")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=5.0)

    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--max_length", type=int, default=256)

    p.add_argument("--skip_wandb", action="store_true")

    config = p.parse_args()

    return config


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_device(config):
    if torch.cuda.is_available():
        if config.gpu_id >= 0:
            if not torch.cuda.device_count() > config.gpu_id:
                raise Exception("Cannot find the GPU device.")
            device = torch.device(f"cuda:{config.gpu_id}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def wandb_init(config):
    final_model_name = f"{config.model_name}-{get_now()}"

    if config.skip_wandb:
        return final_model_name

    wandb.login()
    wandb.init(
        project="NLP_EXP_seq2seq_machine_translation",
        config=vars(config),
        id=final_model_name,
    )
    wandb.run.name = final_model_name
    wandb.run.save()

    os.makedirs(config.output_dir, exist_ok=True)

    return final_model_name


def main(config):
    print(config)
    device = get_device(config)

    tokenizer = T5TokenizerFast.from_pretrained(config.tokenizer)

    train_dataset = TranslationDatatset(config.train_src_fn, config.train_tgt_fn)
    valid_dataset = TranslationDatatset(config.valid_src_fn, config.valid_tgt_fn)

    print(f"# of training samples: {len(train_dataset)}")
    print(f"# of validation samples: {len(valid_dataset)}")

    collator = TranslationCollator(tokenizer, config.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    if not config.with_attention:
        model = Seq2SeqTranslator(
            tokenizer.vocab_size,
            config.hidden_dim,
            config.n_layers,
            config.dropout,
            tokenizer.pad_token_id,
        ).to(device)
    else:
        model = Seq2SeqAttention(
            tokenizer.vocab_size,
            config.hidden_dim,
            config.n_layers,
            config.dropout,
            tokenizer.pad_token_id,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        config,
    )

    final_model_name = wandb_init(config)

    trainer.train(
        train_loader,
        valid_loader,
        model_name=final_model_name,
    )

    if not config.skip_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = define_config()
    main(config)
