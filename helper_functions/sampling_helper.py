import torch
import numpy as np
import random


# @brief: convert indices of pixels to row_Id(from top to bottom), and col_Id(from left to right)
def pixel_indices_to_rc(indices, H, W):
    row_indices = torch.div(indices, W, rounding_mode="floor")
    col_indices = torch.remainder(indices, W)
    return row_indices, col_indices


def pixel_rc_to_indices(rows, cols, H, W):
    indices = rows * W + cols
    return indices


# @brief: randomly sample N pixels
#-@return: indices of selected pixels (row first), Tensor(num, ).
def sample_pixels_random(img_h, img_w, num):
    selected_indices = random.sample(range(img_h * img_w), num)  # Tensor(num, )
    return torch.tensor(selected_indices)


# @brief: randomly sample N pixels from all valid pixels(depth > 0)
# @param depth_image: Tensor(img_h, img_w);
#-@return: indices of selected pixels (row first), Tensor(num, ).
def sample_valid_pixels_random(depth_image, num):
    mask = torch.where(depth_image > 0., torch.ones_like(depth_image), torch.zeros_like(depth_image)).flatten()
    samp_v = mask * torch.abs( torch.randn_like(mask) )  # give each pixel a random value (invalid pixels always got 0; valid pixels > 0)
    selected_indices = torch.topk(samp_v, num)[1]
    return selected_indices


# @brief: sample pixels uniformly from a frame;
#-@return rows: row_Id of sampled pixels, Tensor(num_h * num_w), dtype=torch.int64;
#-@return cols: col_Id of sampled pixels, Tensor(num_h * num_w), dtype=torch.int64.
def sample_pixels_uniformly(img_h, img_w, num_h, num_w):
    interval_h, offset_h = (img_h - num_h) // (num_h + 1), (img_h - num_h) % (num_h + 1)
    interval_w, offset_w = (img_w - num_w) // (num_w + 1), (img_w - num_w) % (num_w + 1)

    row_Ids = torch.arange(0, num_h, dtype=torch.int64) * (interval_h + 1) + interval_h + offset_h // 2  # Tensor(num_h, )
    col_Ids = torch.arange(0, num_w, dtype=torch.int64) * (interval_w + 1) + interval_w + offset_w // 2  # Tensor(num_w, )

    rows = row_Ids[..., None].repeat((1, num_w)).reshape((-1, ))  # Tensor(num_h, num_w)
    cols = col_Ids[None, ...].repeat((num_h, 1)).reshape((-1, ))  # Tensor(num_h, num_w)
    return rows, cols


# @brief: sample pixels both uniformly and randomly from a frame;
# @param num: total number of sampled pixels (uniformly sampled + randomly sampled), int;
# @param depth_image: Tensor(img_h, img_w);
def sample_pixels_mix(img_h, img_w, num_h, num_w, depth_image, num):
    # Step 1: sample (num_h * num_w) pixels uniformly
    row_indices, col_indices = sample_pixels_uniformly(img_h, img_w, num_h, num_w)

    # Step 2: sample [ num - (num_h * num_w) ] pixels randomly
    mask = torch.where(depth_image > 0., torch.ones_like(depth_image), torch.zeros_like(depth_image))  # Tensor(img_h, img_w)
    mask[row_indices, col_indices] = 0
    mask = mask.flatten()  # Tensor(img_h * img_w)
    samp_v = mask * torch.abs( torch.randn_like(mask) )  # give each pixel a random value (invalid pixels always got 0; valid pixels > 0)
    selected_indices_rand = torch.topk(samp_v, num - num_h * num_w)[1]

    # concatenate
    row_indices2, col_indices2 = pixel_indices_to_rc(selected_indices_rand, img_h, img_w)
    row_indices = torch.cat([row_indices, row_indices2], 0)
    col_indices = torch.cat([col_indices, col_indices2], 0)
    return row_indices, col_indices
