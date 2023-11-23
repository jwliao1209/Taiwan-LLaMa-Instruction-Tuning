#!/bin/bash

python train.py --epoch 2 --batch_size 16 --accum_grad_step 1 --lora_rank 16 --lr_scheduler constant
