#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
    --base_model 'meta-llama/Llama-2-7b-chat-hf' \
    --batch_size 5 \
    --prompt_template 'QASC-Full' \
    --output_file_name 'QASC-Full (F1F2 connecting).json' \
    --data_path 'data/eval_data.json' \
    --ablate_connecting_F1F2 True&