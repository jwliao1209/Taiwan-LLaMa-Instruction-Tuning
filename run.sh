#!/bin/bash

if [ ! -d data ]; then
    unzip data.zip
fi

if [ ! -d pretrain ]; then
    unzip pretrain.zip
fi

if [ ! -d checkpoint ]; then
    unzip checkpoint.zip
fi

python infer.py --base_model_path "${1}" \
                --peft_path "${2}" \
                --test_data_path "${3}" \
                --output_path "${4}"
