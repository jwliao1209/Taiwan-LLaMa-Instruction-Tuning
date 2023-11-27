import logging
from argparse import Namespace, ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.prompt import get_prompt
from src.utils import set_random_seeds, get_bnb_config


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
    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help="batch size")
    parser.add_argument("--output_path", type=str,
                        default="public_prediction.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    logger = logging.getLogger("ADL Homework3: Taiwan-LLaMa Inference")
 
    # Prepare model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    if args.method == "lora-fine-tune":
        model = PeftModel.from_pretrained(model, args.peft_path)

    # Prepare question
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    question = input("Please input your question: ")

    prompt = get_prompt(question, incontext=True if args.method == "few-shot" else False)
    print("Prompt:", prompt)

    question_input_ids = torch.tensor(
        [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )

    model.eval()
    with torch.no_grad():
        data = question_input_ids.unsqueeze(0).to(device)
        generated_tokens = model.generate(input_ids=data)
        generation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        generation = generation.replace(prompt, "").strip()
        print("Answer:", generation)
