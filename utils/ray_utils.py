import numpy as np
import torch
import cv2
from utils.sat_utils import utm_from_latlon


def compute_unit_coordinates(
    width, height,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    device=None,
):
    """
    Compute grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to a grid
    where top left corner is (1, 1) and bottom right corner is (-1, -1)).
    """
    dx = width % num_patches_x
    dy = height % num_patches_y
    if use_half_pix:
        min_y = 0
        max_y = height - dy
        min_x = 0
        max_x = width - dx


    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = x / ((width - 1) / 2) - 1
    y_prime = y / ((height - 1) / 2) - 1
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x)], dim=-1)
    return xyd_grid





