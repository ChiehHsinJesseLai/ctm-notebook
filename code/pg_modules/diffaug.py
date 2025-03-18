"""
Import required libraries
"""

import torch
import torch.nn.functional as F


def DiffAugment(x, brightness, saturation,contrast, translation_x, translation_y,
                offset_x, offset_y, policy='', channels_first=True):
    """
    Applies various data augmentation techniques to the input tensor.
    """
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        x = rand_brightness(x, brightness)
        x = rand_saturation(x, saturation)
        x = rand_contrast(x, contrast)
        x = rand_translation(x, translation_x, translation_y)
        x = rand_cutout(x, offset_x, offset_y)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_brightness(x, brightness):
    """
    Randomly adjusts the brightness of the input tensor.
    """
    x = x + brightness
    return x

def rand_saturation(x, saturation):
    """
    Randomly adjusts the saturation of the input tensor.
    """
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * saturation + x_mean
    return x

def rand_contrast(x, contrast):
    """
    Randomly adjusts the contrast of the input tensor.
    """
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * contrast + x_mean
    return x

def rand_translation(x, translation_x, translation_y):
    """
    Randomly translates the input tensor.
    """
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, offset_x, offset_y, ratio=0.2):
    """
    Applies random cutout augmentation to the input tensor.
    """
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x
