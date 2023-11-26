#!/bin/bash

python ppl.py \
       --base_model_path pretrain/Taiwan-LLM-7B-v2.0-chat \
       --peft_path checkpoint/epoch=4_ppl=3.649335366725922
