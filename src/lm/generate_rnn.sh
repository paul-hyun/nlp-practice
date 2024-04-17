#!/bin/bash
python3.9 generate_rnn.py \
    --model_fn ${1} \
    --prompt "${2}"
