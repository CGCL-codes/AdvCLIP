import numpy as np
import torch

def patch_initialization(args, patch_type='rectangle'):
    noise_percentage = args.noise_percentage
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def mask_generation(args, patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[1] - 14 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location

def clamp_patch(args, patch):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    patch = torch.clamp(patch, min=min_out, max=max_out)
    return patch