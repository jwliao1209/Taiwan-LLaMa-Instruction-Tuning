import torch
from torch.utils.data.dataset import Dataset
from src.prompt import get_prompt


class ClassicalChineseDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.data_list = self.transform(data_list)

    def pad_or_truncate(self, data, padding_token=0):
        if self.max_length >= len(data):
            return data + [padding_token] * (self.max_length - len(data))
        else:
            return data[:self.max_length]

    def transform(self, data_list):
        ids = [x["id"] for x in data_list]
        instructions = [get_prompt(x["instruction"]) for x in data_list]
        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)

        processed_data = []
        if self.is_train:
            outputs = [x["output"] for x in data_list]
            tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)

            for i in range(len(data_list)):
                instructions_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
                outputs_input_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]
                processed_data_input_ids =  instructions_input_ids + outputs_input_ids
                processed_data.append(
                    {
                        "id": ids[i],
                        "input_ids": self.pad_or_truncate(processed_data_input_ids),
                        "attention_mask": self.pad_or_truncate([1] * len(processed_data_input_ids)),
                        "labels": self.pad_or_truncate([-100] * len(instructions_input_ids) + outputs_input_ids),
                        "output_mask": self.pad_or_truncate([0] * len(instructions_input_ids) + [1] * len(outputs_input_ids)),
                    }
                )
        else:
            for i in range(len(data_list)):
                processed_data_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
                processed_data.append(
                    {
                        "id": ids[i],
                        "input_ids": processed_data_input_ids,
                        "attention_mask": [1] * len(processed_data_input_ids),
                        "prompt": instructions[i],
                    }
                )
        return processed_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if isinstance(v[0], str) else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict
