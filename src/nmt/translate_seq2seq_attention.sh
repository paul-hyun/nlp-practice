#!/bin/bash
python translate_seq2seq.py \
    --model_fn ${1} \
    --valid_src_fn "../../data/aihub_koen/valid.ko" \
    --gpu_id 0 \
    --with_attention