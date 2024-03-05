#!/bin/bash

python eval.py \
    --output_file 'Bamboogle-LLaMA-13b-chat-output/Bamboogle-Jibberish-Full.json' \
    --data_type 'Bamboogle' \
    --model_type 'llama' \
    --metric 'rouge' \
    --jibberish True
