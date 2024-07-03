#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
    --base_model 'google/flan-t5-xxl' \
    --batch_size 2 \
    --prompt_template 'Bamboogle-Jibberish-Full' \
    --output_file_name 'Bamboogle-flan-T5-output/Bamboogle-full-gibberish (both facts shuffled)-seed:123.json' \
    --data_path 'data/bamboogle-jibberish.json' \
    --random_seed 123 \
    --shuffle_fact1 True \
    --shuffle_fact2 True&

    # --ablate_connecting_F1F2 True&