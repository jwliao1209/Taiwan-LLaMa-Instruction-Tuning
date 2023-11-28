import json
import torch
import random
import numpy as np


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4, ensure_ascii=False)
    return


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
