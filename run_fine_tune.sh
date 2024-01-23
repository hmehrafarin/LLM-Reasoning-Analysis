#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python fine-tune.py \
--base_model 'google/flan-t5-xxl' \
--batch_size 128 \
--learning_rate 3e-4 \
--cutoff_len 256 \
--num_epochs 3 \
--val_set_size 1600 \
--prompt_template 'QASC-Full-Train' \
--output_dir './lora-qasc-T5-11b' \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--train_on_inputs True&