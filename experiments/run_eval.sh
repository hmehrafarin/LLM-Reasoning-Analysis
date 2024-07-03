#!/bin/bash

python eval_2.py \
    --output_file 'Bamboogle-flan-T5-output/Bamboogle-QAF (fact 1 only).json' \
    --data_type 'Bamboogle' \
    --model_type 'flan-t5' \
    --metric 'rouge' \
    # --jibberish True
