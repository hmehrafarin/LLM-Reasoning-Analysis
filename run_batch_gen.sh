#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python batch_generate.py --base_model 'meta-llama/Llama-2-7b-chat-hf' --prompt_template 'QASC-Full' --output_file_name 'generated_response-full (1 token ablated fact 2).json' --ablate_tokens_fact2 True --num_ablations 1&