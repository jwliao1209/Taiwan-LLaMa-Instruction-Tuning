#!/bin/bash

python ppl.py \
       --base_model_path pretrain/Taiwan-LLM-7B-v2.0-chat \
       --peft_path checkpoint/epoch=2_ppl=3.775403222084045
