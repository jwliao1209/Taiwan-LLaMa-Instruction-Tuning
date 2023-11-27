#!/bin/bash

# data
if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=113qrDhnThl9xbYyej-Qe2eKiAm7C-VdS -O data.zip
fi

# pretrain
if [ ! -d pretrain ]; then
    gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O pretrain.zip
fi

# checkpoint
if [ ! -d checkpoint ]; then
    gdown https://drive.google.com/uc?id=1i0d_WDRj-DDG6G7LsTa-sdCqVfZxuHxt -O checkpoint.zip
fi
