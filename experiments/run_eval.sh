#!/bin/bash

python eval.py \
    --output_file 'QASC-full (F1F2 connecting ablation).json' \
    --data_type 'QASC' \
    --model_type 'flan-t5' \
    --metric 'accuracy'
