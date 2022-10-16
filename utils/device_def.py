import torch


def enable_gpu(enable:bool):
    if not enable:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"