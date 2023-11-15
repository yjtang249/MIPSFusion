import numpy as np


# @brief: giving a 2D BOOL matrix，do logical_or for each row;
# @param masks: ndarray(n, m), dtype=np.bool;
#-@return: ndarray(n, ), dtype=np.bool.
def reduce_or(masks):
    masks_value = masks.astype(np.float32)
    sum_masks_value = np.sum(masks_value, axis=-1)  # ndarray(n, )
    sum_masks = sum_masks_value.astype(np.bool)  # ndarray(n, ), dtype=np.bool
    return sum_masks


# @brief: giving a 2D BOOL matrix，do logical_and for each row;
# @param masks: ndarray(n, m), dtype=np.bool;
#-@return: ndarray(n, ), dtype=np.bool.
def reduce_and(masks):
    masks_value = masks.astype(np.float32)
    mul_masks_value = np.cumprod(masks_value, axis=-1)[:, -1]  # ndarray(n, )
    mul_masks = mul_masks_value.astype(np.bool)  # ndarray(n, ), dtype=np.bool
    return mul_masks


# @brief: giving a 2D matrix，normalize each row;
# @param values: ndarray(n, m), dtype=np.float32;
# @param mask: ndarray(n, 1), dtype=np.bool;
#-@return: output matrix normalized by row(for input row with all values==0，it will output a row with all value==0).
def reduce_normalize_l1(values, mask):
    norms = np.sum(values, axis=-1, keepdims=True)  # ndarray(n, 1)
    mask_norm = np.where(norms > 0, np.ones_like(mask), np.zeros_like(mask))
    mask_final = mask & mask_norm

    normalized_values = np.where(mask_final, values / norms, np.zeros_like(values))
    return normalized_values


# @brief: giving a 2D matrix，normalize each row;
# @param values: ndarray(n, m), dtype=np.float32;
# @param mask: ndarray(n, 1), dtype=np.bool;
#-@return: output matrix normalized by row(for input row with all values==0，it will output a row with all value==0).
def reduce_normalize_l2(values, mask):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)  #  ndarray(n, 1)
    normalized_values = np.where(mask > 0., values / norms, np.zeros_like(values))
    return normalized_values


def pdf_gauss(x, mu=0., sigma=1.):
    k1 = 1 / ( sigma * np.sqrt(2 * np.pi) )
    m1 = (x - mu) / sigma
    y = k1 * np.exp(-0.5 * m1**2)
    return y


# @brief: giving n points and a center, compute distance of each point to center;
# @param pts: giving n points, ndarray(n, 3);
# @param center: center, ndarray(3, );
# -@return: distances, ndarray(n, ).
def compute_dist_to_center(pts, center):
    dist = np.linalg.norm(pts - center[None, ...], axis=-1)  # ndarray(n, )
    return dist


# @brief: giving a distance array, use gaussian distribution to fit weights of all distance values(less distance corresponds to greater weights)
# @param dist_value: distance array, ndarray(n, );
#-@return: weights, ndarray(n, ).
def convert_dist_to_weight(dist_value):
    dist_value = np.absolute(dist_value)
    max_dist = np.max(dist_value)
    sigma = max_dist / 3.  # set max_dist == 3 * sigma

    gauss_weight = pdf_gauss(dist_value, mu=0., sigma=sigma)  # ndarray(n, )
    return gauss_weight


# @param grid_entropy: ndarray(n, m), dtype=np.float32;
# @param grid_dist_weight: ndarray(n, m), dtype=np.float32;
# @param grid_mask: ndarray(n, m), dtype=np.bool;
#-@return
def compute_weights(grid_entropy, grid_dist_weight, grid_mask):
    # Step 1: compute mask
    mask1 = reduce_or(grid_entropy.astype(np.bool))[..., None]  # for each grid, whether its entropy for all submesh are all 0, ndarray(n, 1)
    mask2 = reduce_or(grid_mask)[..., None]  # for each grid, whether it is visible for at least 1 submesh, ndarray(n, 1)
    # mask = np.logical_and(mask1, mask2)
    mask = mask2

    # Step 2: inverse entropy, and get unnormalized weights
    entropy_inv = 1. * np.exp(-10. * grid_entropy)

    masked_entropy_inv = entropy_inv * grid_mask.astype(np.float32)
    masked_dist_weight = grid_dist_weight * grid_mask.astype(np.float32)

    # grid_weights = reduce_normalize_l1(masked_entropy_inv, mask)
    grid_weights = reduce_normalize_l1(masked_entropy_inv * masked_dist_weight, mask)
    # grid_weights = reduce_normalize_l1(masked_dist_weight, mask)

    return grid_weights


# @param grid_entropy: ndarray(n, m), dtype=np.float32;
# @param grid_dist_weight: ndarray(n, m), dtype=np.float32;
# @param grid_mask: ndarray(n, m), dtype=np.bool.
def compute_weights2(grid_entropy, grid_dist_weight, grid_mask):
    # Step 1: compute mask
    mask1 = reduce_or(grid_entropy.astype(np.bool))[..., None]  # ndarray(n, 1)
    mask2 = reduce_or(grid_mask)[..., None]  # ndarray(n, 1)
    mask = np.logical_and(mask1, mask2)

    # Step 2: inverse entropy, and get unnormalized weights
    entropy_inv = (1. / grid_entropy)**2
    masked_entropy_inv = entropy_inv * grid_mask.astype(np.float32)
    masked_dist_weight = grid_dist_weight * grid_mask.astype(np.float32)

    grid_weights = reduce_normalize_l1(masked_dist_weight, mask)
    return grid_weights
