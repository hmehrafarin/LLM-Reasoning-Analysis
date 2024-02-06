#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py \
 --base_model 'google/flan-t5-xxl' \
 --batch_size 2 \
 --prompt_template 'QASC-QAF (fact 2 only)' \
 --output_file_name 'QASC-QAF (fact 2 only) \                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   )).json' \
 --data_path 'data/eval_data.json'&
#  --ablate_connecting_F1F2 True&
#  --shuffle_fact1 True \
#  --shuffle_fact2 True&