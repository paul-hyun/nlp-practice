#!/bin/bash
python classify.py \
    --model_fn ${1} \
    --test_tsv_fn '../../data/nsmc/test.tsv'
