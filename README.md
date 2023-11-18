# Taiwan-LLaMa Instruction Tuning

This repository is implementation of Homework 3 for CSIE5431 Applied Deep Learning course in 2023 Fall semester at National Taiwan University.


## Setting the Environment 
To set the environment, you can run this command:
```
pip install -r configs/requirements.txt
```


## Download dataset and model checkpoint
To download the datasets and model checkpoint, you can run the commad:
```
bash ./download.sh
```

## Reproducing best result
To reproduce our best result, you can run the commad:
```
bash ./run.sh <data file> <output file>
```


## Training
To fine-tune the Taiwan-LLaMa model, you can run the commad:
```
python train.py
```


## Operating System and Device
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Acknowledgement
We thank the Taiwan-LLaMa repository: https://github.com/MiuLab/Taiwan-LLaMa


## Citation
```bibtex
@misc{
    title  = {Taiwan-LLaMa Instruction Tuning},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Taiwan-LLaMa-Instruction-Tuning.git},
    year   = {2023}
}
```
