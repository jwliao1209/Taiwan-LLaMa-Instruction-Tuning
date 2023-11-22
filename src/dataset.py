import torch
from torch.utils.data.dataset import Dataset
from src.utils import get_prompt


def pad_or_truncate(data, max_length, padding_token=0):
    if max_length >= len(data):
        return data + [padding_token] * (max_length - len(data))
    else:
        return data[:max_length]


class ClassicalChineseDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = self.transform(data_list)

    def transform(self, data_list):
        instructions = [get_prompt(x["instruction"]) for x in data_list]
        outputs = [x["output"] for x in data_list]

        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)
        tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)

        processed_data = []
        for i in range(len(data_list)):
            instructions_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
            outputs_input_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]

            processed_data_input_ids =  instructions_input_ids + outputs_input_ids
            processed_data_attention_mask = [1] * len(processed_data_input_ids)
            processed_data_labels = [-100] * len(instructions_input_ids) + outputs_input_ids
            processed_data_output_mask = [0] * len(instructions_input_ids) + [1] * len(outputs_input_ids)

            processed_data_input_ids = pad_or_truncate(processed_data_input_ids, self.max_length, 0)
            processed_data_attention_mask = pad_or_truncate(processed_data_attention_mask, self.max_length, 0)
            processed_data_labels = pad_or_truncate(processed_data_labels, self.max_length, 0)
            processed_data_output_mask = pad_or_truncate(processed_data_output_mask, self.max_length, 0)

            processed_data.append(
                {
                    "input_ids": torch.tensor(processed_data_input_ids),
                    "attention_mask": torch.tensor(processed_data_attention_mask),
                    "labels": torch.tensor(processed_data_labels),
                    "output_mask": torch.tensor(processed_data_output_mask),
                }
            )
        return processed_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
