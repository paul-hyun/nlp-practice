#!/bin/bash
python tokenizer_train.py \
        --train_files "../../data/kowiki/wiki_dump.txt" \
        --vocab_size 32000 \
        --output_dir "../../data/kowiki_32k"
