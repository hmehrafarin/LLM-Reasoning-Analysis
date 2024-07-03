#!/bin/bash

srun -u --nodes=1 --mem=200G --time=10:00:00 --partition=gpu --gres=gpu:1 python shuffle-experiment.py 
    # --ablate_connecting_F1F2 True&