#!/bin/bash

# python infer.py \
#     --base_model_path pretrain/Taiwan-LLM-7B-v2.0-chat \
#     --peft_path checkpoint/epoch=2_ppl=3.775403222084045 \
#     --test_data_path data/public_test.json \
#     --output_path prediction_public.json

python infer.py \
    --base_model_path pretrain/Taiwan-LLM-7B-v2.0-chat \
    --peft_path checkpoint/epoch=2_ppl=3.775403222084045 \
    --test_data_path data/private_test.json \
    --output_path prediction_private.json
