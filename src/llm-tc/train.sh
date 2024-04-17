#!/bin/bash
python3.9 train.py \
    --model_name ${1} \
    --train_tsv_fn '../../data/nsmc/train.tsv' \
    --test_tsv_fn '../../data/nsmc/test.tsv' \
    --output_dir "../../checkpoints" \
    --max_steps 2000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
