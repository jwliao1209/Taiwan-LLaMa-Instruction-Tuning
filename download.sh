#!/bin/bash

# data
# if [ ! -d data ]; then
#     gdown https://drive.google.com/uc?id=113qrDhnThl9xbYyej-Qe2eKiAm7C-VdS -O data.zip
# fi

# pretrain
# if [ ! -d pretrain ]; then
#     gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O pretrain.zip
# fi

# checkpoint
if [ ! -d adapter_checkpoint ]; then
    gdown https://drive.google.com/uc?id=1XrA6szuQn_G2FoYhivu3knwX3j6VAMRa -O adapter_checkpoint.zip
fi
