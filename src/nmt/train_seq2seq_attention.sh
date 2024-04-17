#!/bin/bash
python3.9 train_seq2seq.py \
    --model_name ${1} \
    --train_src_fn "../../data/aihub_koen/train.ko" \
    --train_tgt_fn "../../data/aihub_koen/train.en" \
    --valid_src_fn "../../data/aihub_koen/valid.ko" \
    --valid_tgt_fn "../../data/aihub_koen/valid.en" \
    --tokenizer "../../data/aihub_koen_32k" \
    --output_dir "../../checkpoints" \
    --gpu_id 0 \
    --n_epochs 10 \
    --with_attention
