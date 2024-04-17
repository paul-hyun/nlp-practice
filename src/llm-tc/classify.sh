#!/bin/bash
python3.9 classify.py \
    --model_fn ${1} \
    --test_tsv_fn '../../data/nsmc/test.tsv'
