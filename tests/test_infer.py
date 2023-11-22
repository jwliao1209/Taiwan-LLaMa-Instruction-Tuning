import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from argparse import Namespace, ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.dataset import ClassicalChineseDataset, collate_func
from src.utils import set_random_seeds, read_json, get_bnb_config, dict_to_device, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--base_model_path", type=str,
                        default="pretrain/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--test_data_path", type=str,
                        default="data/public_train.json",
                        help="Path to test data.")
    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help="batch size")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()

    args = parse_arguments()

    # Prepare dataset
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    test_data = read_json(args.train_data_path)
    # test_dataset = ClassicalChineseDataset(test_data, tokenizer)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func)

    # Prepare model
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    bnb_config = get_bnb_config()
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model_path,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=bnb_config
    # )
    # model = PeftModel.from_pretrained(model, args.peft_path)

    # model.eval()
    # prediction_list = []
    # test_bar = tqdm(test_loader, desc=f"Testing")
    # for _, batch_data in enumerate(test_bar, start=1):
    #     with torch.no_grad():
    #         batch_data = dict_to_device(batch_data, device)
    #         generated_tokens = model.generate(
    #             input_ids=batch_data["input_ids"],
    #             attention_mask=batch_data["attention_mask"],
    #         )
    #         generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #         prediction_list.extend(
    #             [
    #                 {"id": ID, "output": pred}
    #                 for ID, pred in zip(batch_data["id"], generations)
    #             ]
    #         )
    # write_json(prediction_list, args.output_path)
