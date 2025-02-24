import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    计算 nn.Module 模型的参数数量。
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params
