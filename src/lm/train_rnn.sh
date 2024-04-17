#!/bin/bash
python3.9 train_rnn.py \
    --model_name ${1} \
    --train_tsv_fn "../../data/kowiki/wiki_dump.txt" \
    --tokenizer "../../data/aihub_koen_32k" \
    --output_dir "../../checkpoints" \
    --gpu_id 0 \
    --n_epochs 4
