#!/bin/sh
python finetune_bert_tc.py \
    --model_name ${1} \
    --train_tsv_fn '../../data/nsmc/train_dataset.tsv' \
    --valid_tsv_fn '../../data/nsmc/valid_dataset.tsv' \
    --test_tsv_fn '../../data/nsmc/test_dataset.tsv' \
    --output_dir "../../checkpoints" \
    --backbone "klue/roberta-base" \
    --batch_size_per_device 64 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --max_length 128
