import torch
import torch.nn.functional as F


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


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
    # Valid sdf region
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


# @brief: considered EMD loss
def get_sdf_loss(z_vals, target_d, predicted_sdf, sdf_prob, truncation, cate_num=5, EMD_w=0.01, loss_type="l2"):
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

    return fs_loss, sdf_loss