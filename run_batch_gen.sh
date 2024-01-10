#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py --base_model 'google/flan-t5-xxl' --batch_size 2 --prompt_template 'QASC_Final-Answer_QA' --output_file_name 'generated_response-QA-t5.json'&