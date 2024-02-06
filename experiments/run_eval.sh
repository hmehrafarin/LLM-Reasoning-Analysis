#!/bin/bash

python eval.py \
    --output_file 'QASC-flan-T5-xxl/QASC-QAF.json' \
    --data_type 'QASC' \
    --model_type 'flan-t5' \
    --metric 'accuracy'
