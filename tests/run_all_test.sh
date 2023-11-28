#!/bin/bash

# download dataset and pretrain model
bash ./download.sh


# reproduce
bash ./run.sh \
    pretrain/Taiwan-LLM-7B-v2.0-chat \
    adapter_checkpoint \
    data/public_test.json \
    public_prediction.json

bash ./run.sh \
    pretrain/Taiwan-LLM-7B-v2.0-chat \
    adapter_checkpoint \
    data/private_test.json \
    private_prediction.json


# evaluate
python ppl.py --base_model_path pretrain/Taiwan-LLM-7B-v2.0-chat \
              --peft_path adapter_checkpoint \
              --test_data_path data/public_test.json
