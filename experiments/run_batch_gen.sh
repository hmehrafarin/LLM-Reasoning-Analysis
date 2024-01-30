#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
 --base_model 'google/flan-t5-xxl' \
 --batch_size 2 \
 --prompt_template 'Bamboogle-Full' \
 --output_file_name 'generated_response_Bamboogle_Full_flan-t5 (shuffled both facts).json' \
 --shuffle_fact1 True \
 --shuffle_fact2 True \
 --data_path 'data/bamboogle-with-facts.json'&