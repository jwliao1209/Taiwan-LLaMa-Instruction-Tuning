import torch
from transformers import BitsAndBytesConfig


def get_prompt(instruction: str, incontext: bool = False) -> str:
    '''Format the instruction as a prompt for LLM.'''
    if incontext:
        return f"""你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。提供你兩個例子參考:
    1. USER: 翻譯成文言文：雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。ASSISTANT: 雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。
    2. USER: 辛未，命吳堅為左丞相兼樞密使，常楙參知政事。把這句話翻譯成白話文。ASSISTANT: 初五，命令吳堅為左承相兼樞密使，常增為參知政事。
    以下的問題為文言文翻譯成白話文或白話文翻譯成文言文。請回答: USER: {instruction} ASSISTANT:"""
    else:
        return f"""你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。以下的問題為文言文翻譯成白話文或白話文翻譯成文言文。USER: {instruction} ASSISTANT:"""


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
