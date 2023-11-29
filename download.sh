#!/bin/bash

# Prepare data
# if [ ! -d data ]; then
#     gdown https://drive.google.com/uc?id=113qrDhnThl9xbYyej-Qe2eKiAm7C-VdS -O data.zip
# fi

# if [ ! -d data ]; then
#     unzip data.zip
# fi


# Prepare pretrain
# if [ ! -d pretrain ]; then
#     gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O pretrain.zip
# fi

# if [ ! -d pretrain ]; then
#     unzip pretrain.zip
# fi


# Prepare checkpoint
if [ ! -d adapter_checkpoint ]; then
    gdown https://drive.google.com/uc?id=1XrA6szuQn_G2FoYhivu3knwX3j6VAMRa -O adapter_checkpoint.zip
fi

if [ ! -d adapter_checkpoint ]; then
    unzip adapter_checkpoint.zip
fi
