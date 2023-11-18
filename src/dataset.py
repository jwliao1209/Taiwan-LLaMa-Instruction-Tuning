import torch
from torch.utils.data.dataset import Dataset

from src.utils import get_prompt


class ClassicalChineseDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self.process()

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        return self.processed_data[index]

    def process(self):
        data_size = len(self.data_list)
        instructions = [get_prompt(x["instruction"]) for x in self.data_list]
        outputs = [x["output"] for x in self.data_list]

        # Tokenize data
        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)
        tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)
        # labels = self.tokenizer(outputs, max_length=128, padding="max_length", truncation=True)["input_ids"]

        tokenized_instructions["labels"] = []

        # Format data
        for i in range(data_size):
            instruction_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]  # .bos_token_id = 0
            output_input_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]  # .eos_token_id = 2
            tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids
            tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])  # .attention_mask = 1
            tokenized_instructions["input_ids"][i] = tokenized_instructions["input_ids"][i][:self.max_length]
            tokenized_instructions["attention_mask"][i] = tokenized_instructions["attention_mask"][i][:self.max_length]

            length = len(tokenized_instructions["input_ids"][i])
            tokenized_instructions["input_ids"][i] += [0] * (self.max_length - length)
            tokenized_instructions["attention_mask"][i] += [0] * (self.max_length - length)

            tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i])
            tokenized_instructions["attention_mask"][i] = torch.tensor(tokenized_instructions["attention_mask"][i])
            
            tokenized_instructions["labels"].append(torch.tensor(
                ([0] * len(instruction_input_ids) + output_input_ids)[:self.max_length] + [0] * (self.max_length - length)
            ))
            
            
        # tokenized_instructions["labels"] = [torch.tensor(l) for l in labels]
        
        tokenized_instructions = dict(tokenized_instructions)
        processed_data = [dict((k, tokenized_instructions[k][i]) for k in tokenized_instructions) for i in range(data_size)]
        return processed_data
