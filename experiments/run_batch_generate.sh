#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
    --base_model 'meta-llama/Llama-2-7b-chat-hf' \
    --batch_size 2 \
    --prompt_template 'QASC-QAF (fact 2 only)' \
    --output_file_name 'QASC-QAF (fact 2 only).json' \
    --data_path 'data/eval_data.json'&