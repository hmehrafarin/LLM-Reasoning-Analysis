#!/bin/bash

python eval.py \
    --output_file 'QASC-LLaMA-7b-Chat/QASC-Full (F1Q connecting).json' \
    --data_type 'QASC' \
    --model_type 'llama' \
    --metric 'accuracy'
