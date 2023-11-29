# Taiwan-LLaMa Instruction Tuning

This repository is implementation of Homework 3 for CSIE5431 Applied Deep Learning course in 2023 Fall semester at National Taiwan University.

<img width="500" alt="instruction_tuning" src="https://github.com/jwliao1209/Taiwan-LLaMa-Instruction-Tuning/assets/55970911/68bba37f-8695-4335-b296-960f9e753cd9">


## Setting the Environment 
To set the environment, you can run this command:
```
pip install -r configs/requirements.txt
```


## Download LoRA checkpoint
To download the LoRA checkpoint, you can run the command:
```
bash ./download.sh
```


## Data Format
The dataset comprises keys labeled id, instruction, and output. An example of the data structure is displayed as follows:

```
{
    "id": "2fb7d211-978f-41c8-a3ab-e51d9df06280",
    "instruction": "翻譯成文言文：\n於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。",
    "output": "帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。"
}
```


## Reproducing
To reproduce our best result, you can run the command:
```
bash ./run.sh <pretrain model folder> <lora model folder> <input data path> <output file path>
```
For example:
```
bash ./run.sh \
    pretrain/Taiwan-LLM-7B-v2.0-chat \
    adapter_checkpoint \
    data/public_test.json \
    public_prediction.json
```


## Training
To fine-tune the Taiwan-LLaMa model, you can run the command:
```
python train.py --base_model_path <pretrain model folder> \
                --train_data_path <train data path> \
                --valid_data_path <valid data path> \
                --train_num <number of used training data> \
                --epoch <number of epochs> \
                --batch_size <number of training batch size> \
                --accum_grad_step <number of accumulated gradient batch size> \
                --lr <learning rate> \
                --lr_scheduler <learning rate scheduler> \
                --warm_up_step <number of warm up step>
                --lora_rank <rank of LoRA>
```


## Inference
To inference the Taiwan-LLaMa model, you can run the command:
```
python infer.py --method <support method: lora-fine-tune, zero-shot, and few-shot> \
                --base_model_path <pretrain model folder> \
                --peft_path <lora model folder> \
                --test_data_path <test data path> \
                --output_path <output file oath>
```


## Demo
To demo the conversation with Taiwan-LLaMa model, you can run the command:
```
python demo.py --method <support method: lora-fine-tune, zero-shot, and few-shot>
               --base_model_path <pretrain model folder> \
               --peft_path <lora model folder> \
               --test_data_path <test data path>
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
