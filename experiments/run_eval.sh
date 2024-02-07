#!/bin/bash

python eval.py \
    --output_file 'QASC-QAF (fact 2 only).json' \
    --data_type 'QASC' \
    --model_type 'flan-t5' \
    --metric 'accuracy'
