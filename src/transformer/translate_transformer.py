import os
import sys
import argparse

import numpy as np

import torch

from transformers import (
    T5TokenizerFast,
    GenerationConfig,
    T5ForConditionalGeneration,
)


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--valid_src_fn", required=True)
    p.add_argument("--gpu_id", type=int, default=-1)

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--num_beams", type=int, default=8)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--show_special_tokens", action="store_true")

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

    tokenizer = T5TokenizerFast.from_pretrained(config.model_fn)

    model = T5ForConditionalGeneration.from_pretrained(config.model_fn)
    model = model.to(device)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=config.max_length,
        early_stopping=config.early_stopping,
        do_sample=False,
        num_beams=config.num_beams,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        repetition_penalty=config.repetition_penalty,
        length_penalty=config.length_penalty,
    )

    test_input_ids = tokenizer.batch_encode_plus(
        src,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    ).input_ids.to(device)

    beam_output = model.generate(
        input_ids=test_input_ids,
        generation_config=generation_config,
    )

    for line, tgt in zip(src, beam_output):
        result = tokenizer.decode(
            tgt, skip_special_tokens=not config.show_special_tokens
        )
        print(f"- ko: {line}\n- en: {result}\n")


if __name__ == "__main__":
    config = define_config()
    main(config)
