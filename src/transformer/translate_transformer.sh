#!/bin/bash
python3.9 translate_transformer.py \
    --model_fn ${1} \
    --valid_src_fn "../../data/aihub_koen/valid.ko" \
    --gpu_id 0
