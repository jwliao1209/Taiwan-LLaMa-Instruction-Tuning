import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.dataset import ClassicalChineseDataset
from src.optimizer import get_optimizer
from src.trainer import Trainer
from utils import get_bnb_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="pretrain/Taiwan-LLM-7B-v2.0-chat",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train.json",
        help="Path to train data."
    )


    parser.add_argument("--batch_size", type=int,
                        default=20,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=2,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=1,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=3e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0, #1e-5,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="linear",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=0, #100,
                        help="number of warm up steps")

    args = parser.parse_args()

    # Prepare dataset
    with open(args.train_data_path, "r") as f:
        data = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    train_dataset = ClassicalChineseDataset(data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare model
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=train_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        # num_beams=args.num_beams,
        # logger=wandb,
        # **sampling_params,
    )
    trainer.fit(epoch=args.epoch)
