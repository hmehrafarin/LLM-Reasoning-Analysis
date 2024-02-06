#!/bin/bash

python eval.py \
    --output_file 'Bamboogle-flan-T5-output/Bamboogle-QAF.json' \
    --data_type 'Bamboogle' \
    --model_type 'llama' \
    --metric 'accuracy'
