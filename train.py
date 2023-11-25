import math
import wandb

import torch
from torch.utils.data import DataLoader

from argparse import Namespace, ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.dataset import ClassicalChineseDataset, collate_func
from src.optimizer import get_optimizer
from src.trainer import Trainer
from src.utils import set_random_seeds, read_json, get_bnb_config


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--base_model_path", type=str,
                        default="pretrain/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--train_data_path", type=str,
                        default="data/train.json",
                        help="Path to train data.")
    parser.add_argument("--valid_data_path", type=str,
                        default="data/public_test.json",
                        help="Path to validation data.")
    parser.add_argument("--train_num", type=int,
                        default=4000,
                        help="number of training data")
    parser.add_argument("--batch_size", type=int,
                        default=16,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=1,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=3,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="constant",
                        help="learning rate scheduler: linear, constant, cosine, cosine_warmup")
    parser.add_argument("--warm_up_step", type=int,
                        default=0,
                        help="number of warm up steps")
    parser.add_argument("--lora_rank", type=int,
                        default=16,
                        help="rank of lora")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()

    args = parse_arguments()

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    train_data = read_json(args.train_data_path)[:args.train_num]
    valid_data = read_json(args.valid_data_path)

    train_dataset = ClassicalChineseDataset(train_data, tokenizer)
    valid_dataset = ClassicalChineseDataset(valid_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func)

    # Prepare model
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=args.lora_rank,
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

    # Prepared logger
    wandb.init(
        project="adl_hw3",
        name="experiment", 
        config={
            "tokenizer": args.base_model_path,
            "model": args.base_model_path,
            "epoch": args.epoch,
            "train_data_num": args.train_num,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warm_up_step": args.warm_up_step,
            "lora_rank": args.lora_rank,
        }
    )
    wandb.watch(model, log="all")

    # Start training
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
    )
    trainer.fit(epoch=args.epoch)
