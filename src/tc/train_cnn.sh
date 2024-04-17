#!/bin/bash
python3.9 train_cnn.py \
    --model_name ${1} \
    --train_tsv_fn '../../data/nsmc/train_dataset.tsv' \
    --valid_tsv_fn '../../data/nsmc/valid_dataset.tsv' \
    --test_tsv_fn '../../data/nsmc/test_dataset.tsv' \
    --output_dir "../../checkpoints" \
    --n_epochs 20 \
    --gpu_id=0
