#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
 --base_model 'google/flan-t5-xxl' \
 --batch_size 2 \
 --prompt_template 'QASC-Full' \
 --output_file_name 'QASC-full (F1Q connecting ablation).json' \
 --data_path 'data/eval_data.json' \
 --ablate_connecting_F1Q True&
#  --shuffle_fact1 True \
#  --shuffle_fact2 True&