#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \--base_model 'google/flan-t5-xxl' --batch_size 2 --prompt_template 'QASC-Full' --output_file_name 'generated_response-full-(F1F2A keyword ablation)-t5.json' --ablate_matching_words_with_answers True&