#!/bin/bash
python3.9 classify_cnn.py \
    --model_fn ${1} \
    --test_tsv_fn '../../data/nsmc/test.tsv' \
    --device 0
