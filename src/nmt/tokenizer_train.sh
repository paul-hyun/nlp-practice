#!/bin/bash
python3.9 tokenizer_train.py \
        --train_files "../../data/aihub_koen/train.ko" "../../data/aihub_koen/train.en" \
        --vocab_size 32000 \
        --output_dir "../../data/aihub_koen_32k"
