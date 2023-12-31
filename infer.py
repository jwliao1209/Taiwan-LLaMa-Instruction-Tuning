import logging
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import get_bnb_config
from src.dataset import ClassicalChineseDataset, collate_func
from src.utils import set_random_seeds, read_json, dict_to_device, save_json


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--method", type=str,
                        default="lora-fine-tune",
                        help="support method: zero-shot, few-shot, and lora-fine-tune")
    parser.add_argument("--base_model_path", type=str,
                        default="pretrain/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--peft_path",
                        type=str,
                        default="checkpoint/epoch=4_ppl=3.649335366725922",
                        help="Path to the saved PEFT checkpoint.")
    parser.add_argument("--test_data_path", type=str,
                        default="data/public_test.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="public_prediction.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    logger = logging.getLogger("ADL Homework3: Taiwan-LLaMa Inference")

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    test_data = read_json(args.test_data_path)
    test_dataset = ClassicalChineseDataset(
        test_data, tokenizer, is_train=False,
        incontext=True if args.method == "few-shot" else False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    for _, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_new_tokens=512,
            )
            generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            generations = generations.replace(batch_data["prompt"][0], "").strip()
            prediction_list.append({"id": batch_data["id"][0], "output": generations})
            
            logger.debug(f"Question:\n{batch_data['prompt'][0]}\n")
            logger.debug(f"Answer:\n{generations}\n")

    save_json(prediction_list, args.output_path)
