#!/bin/bash
python classify_bert.py \
    --model_fn ${1} \
    --test_tsv_fn '../../data/nsmc/test.tsv' \
    --device 0