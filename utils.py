from pathlib import Path

import torch


def image_to_tensor(image, device):
    x = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return (x / 255.0).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sorted_shards(directory):
    return sorted(Path(directory).glob("*.pt"))
