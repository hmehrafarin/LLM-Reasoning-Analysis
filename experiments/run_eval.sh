#!/bin/bash

python eval.py \
    --output_file 'QASC-LLaMA-13b-Chat/QASC-Full (F1F2 connecting ablation).json' \
    --data_type 'QASC' \
    --model_type 'llama' \
    --metric 'accuracy'
