#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python perplexity_calculator.py --base_model 'meta-llama/Llama-2-13b-chat-hf' --prompt_template 'QASC_QA (perplexity)' &