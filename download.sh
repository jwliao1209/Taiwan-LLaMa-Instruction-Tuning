#!/bin/bash

# data
if [ ! -d data_ ]; then
    gdown https://drive.google.com/uc?id=113qrDhnThl9xbYyej-Qe2eKiAm7C-VdS -O data.zip
fi

# pretrain
if [ ! -d pretrain_ ]; then
    gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O pretrain.zip
fi

# checkpoint
if [ ! -d checkpoint_ ]; then
    gdown https://drive.google.com/uc?id=1f1zoAqTyxhNbf301d59e6fNiBv5FGVHb -O checkpoint.zip
fi
