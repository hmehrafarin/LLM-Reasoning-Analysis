#!/bin/bash

python eval.py \
    --output_file 'Bamboogle-LLaMA-13b-chat-output/Bamboogle-jibberish-full (shuffled both facts).json' \
    --data_type 'Bamboogle' \
    --model_type 'llama' \
    --metric 'rouge' \
    --jibberish True
