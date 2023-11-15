# package imports
import torch
import torch.nn.functional as F
from math import exp, log, floor


def mse2psnr(x):
    '''
    MSE to PSNR
    '''
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


def coordinates(voxel_dim, device: torch.device):
    '''
    Params: voxel_dim: int or tuple of int
    Return: coordinates of the voxel grid
    '''
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    '''
    Params:
        bins: torch.Tensor, (Bs, N_samples)
        weights: torch.Tensor, (Bs, N_samples)
        N_importance: int
    Return:
        samples: torch.Tensor, (Bs, N_importance)
    '''
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Bs, N_samples-2
    cdf = torch.cumsum(pdf, -1) 
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1) # Bs, N_samples-1
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def batchify(fn, chunk=1024*64):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn
        def ret(inputs, inputs_dir=None):
            if inputs_dir is not None:
                return torch.cat([fn(inputs[i:i+chunk], inputs_dir[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret


def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight


def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params: 
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type=None, grad=None):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        sdf_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    if grad is not None:  # default: skip if
        eikonal_loss = (((grad.norm(2, dim=-1) - 1) ** 2) * sdf_mask / sdf_mask.sum()).sum()
        return fs_loss, sdf_loss, eikonal_loss

    return fs_loss, sdf_loss


# @brief: considered EMD loss
def get_sdf_loss2(z_vals, target_d, predicted_sdf, sdf_prob, truncation, cate_num=5, EMD_w=0.01, loss_type="l2"):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        sdf_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''
    max_class_Id = cate_num - 1

    # Step 1: compute free-space mask / truncation-region mask
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    # Step 2: free-space loss and SDF loss of classification (EMD loss)
    index_range = torch.arange(0, cate_num).to(sdf_prob)  # 每个分类的class_Id, EagerTensor(cate_num, )

    if EMD_w > 0:
        # 2.1: free-space EMD loss (for pts in front free space)
        fs_loss_all = sdf_prob * (max_class_Id - index_range).to(sdf_prob) * front_mask[..., None]   # pred class_Id should be as close as possible to max_class_Id
        fs_loss1 = torch.mean( torch.sum(fs_loss_all, dim=-1) ) / 250

        # 2.2: truncation EMD loss (for pts in truncation region)
        gt_sdf_class = ( ( (target_d - z_vals) + truncation ) / (2. * truncation) ) * max_class_Id  # gt class_Id of each point, Tensor(n_rays, pts_per_ray)
        sdf_loss_all = torch.abs(gt_sdf_class[:, :, None] - index_range[None, None, :]) * sdf_mask[..., None] * sdf_prob  # penalty value of each point, Tensor(n_rays, pts_per_ray, cate_num)
        sdf_loss1 = torch.mean( torch.sum(sdf_loss_all, dim=-1) ) / 5000

        # Step 3: free-space loss and SDF loss of regression
        fs_loss2 = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
        sdf_loss2 = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

        fs_loss = fs_loss2 + fs_loss1 * EMD_w
        sdf_loss = sdf_loss2 + sdf_loss1 * EMD_w
    else:
        fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
        sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    # fs_loss = fs_loss2
    # sdf_loss = sdf_loss2

    # if grad is not None:  # default: skip if
    #     eikonal_loss = (((grad.norm(2, dim=-1) - 1) ** 2) * sdf_mask / sdf_mask.sum()).sum()
    #     return fs_loss, sdf_loss, eikonal_loss

    return fs_loss, sdf_loss