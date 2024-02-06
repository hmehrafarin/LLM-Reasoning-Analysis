#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
 --base_model 'meta-llama/Llama-2-13b-chat-hf' \
 --batch_size 3 \
 --prompt_template 'Bamboogle-Full' \
 --output_file_name 'Bamboogle-full.json' \
 --data_path 'data/bamboogle-with-facts.json'&
#  --ablate_connecting_F1F2 True&
#  --shuffle_fact1 True \
#  --shuffle_fact2 True \