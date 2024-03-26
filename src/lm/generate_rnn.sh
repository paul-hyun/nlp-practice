#!/bin/bash
python generate_rnn.py \
    --model_fn ${1} \
    --prompt "${2}"