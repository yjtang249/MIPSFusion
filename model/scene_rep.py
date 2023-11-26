import numpy as np
import copy
import torch
import torch.nn as nn

from .encodings import get_encoder
from .decoder import MLP_reg
from helper_functions.utils import batchify, get_sdf_loss, mse2psnr, compute_loss


class JointEncoding(nn.Module):
    def __init__(self, config, bound_box, coords_norm_factor):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.coords_norm_factor = coords_norm_factor
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)
        self.save_initial_param()  # save initial parameters of scene_rep


    # @brief: get resolution for multi-resolution hash grid encoding
    def get_resolution(self):
        dim_max = (self.bounding_box[:, 1] - self.bounding_box[:, 0]).max()
        if self.config["grid"]["voxel_sdf"] > 10:
            self.resolution_sdf = self.config["grid"]["voxel_sdf"]
        else:
            self.resolution_sdf = int(dim_max / self.config["grid"]["voxel_sdf"])

        # print("SDF resolution:", self.resolution_sdf)


    # @brief: get encoding of this submap
    def get_encoding(self, config):
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config["pos"]["enc"], n_bins=self.config["pos"]["n_bins"])

        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder(config["grid"]["enc"], log2_hashmap_size=config["grid"]["hash_size"], desired_resolution=256)


    # @brief: get MLP
    def get_decoder(self, config):
        self.decoder = MLP_reg(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)  # MLP + MRHG


    # @brief: save initial values of encoding parameters and MLP parameters
    def save_initial_param(self):
        self.initial_dict = copy.deepcopy(self.state_dict())


    # @brief: set the encoding parameters and MLP parameters to initial values
    def recover_initial_param(self):
        self.load_state_dict(self.initial_dict)


    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args["training"]["trunc"]) * torch.sigmoid(-sdf / args["training"]["trunc"])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args["data"]["sc_factor"] * args["training"]["trunc"], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)


    def raw2outputs(self, raw, z_vals):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var

    # @param query_points: Tensor(N_rays, N_samples, 3)
    def query_sdf(self, query_points):
        return self.query_color_sdf(query_points)[..., 3:4]

    def query_color(self, query_points):
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])

    # @param query_points: Tensor(N_rays, N_samples, 3)
    def query_sdf_entropy_prob(self, query_points):
        return self.query_color_sdf(query_points)[..., 3:]

    # @param query_points: Tensor(n_rays, n_samples, 3);
    #-@return: Tensor(n_rays, n_samples, 10), RGB + SDF + entropy + prob (3+1+1+5).
    def query_color_sdf(self, query_points):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]]) / self.config["training"]["norm_factor"]

        # Step 1: get parametric encoding + pos encoding
        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)

        # Step 2: feed embeddings to decoders
        inputs_flat = inputs_flat.to(torch.float32)
        outputs = self.decoder(embed, embe_pos, inputs_flat)
        return outputs


    # @brief: do inference for a batch of given 3D points;
    # @param inputs: Tensor(n_rays * n_samples, 3);
    #-@return: Tensor(n_rays * n_samples, 10).
    def run_network(self, inputs):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # Tensor(n_rays * n_samples, 3)

        # normalize the input to [0, 1]
        if self.config["grid"]["tcnn_encoding"]:
            if self.config["grid"]["use_bound_normalize"]:
                inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            else:
                inputs_flat = (inputs_flat + self.coords_norm_factor) / (2 * self.coords_norm_factor)

        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)  # Tensor(n_rays * n_samples, 10)
        outputs = torch.reshape( outputs_flat, list(inputs.shape[:-1]) + [ outputs_flat.shape[-1] ] )  # Tensor(n_rays, n_samples, 10)
        return outputs


    # @brief: do volume rendering for each given ray,
    # @param rays_o: rays origins, Tensor(n_rays, 3);
    # @param rays_d: rays directions, Tensor(n_rays, 3);
    # @param target_d: gt depth of sampled rays, Tensor(n_rays, 1).
    def render_rays(self, rays_o, rays_d, target_d=None):
        n_rays = rays_o.shape[0]  # number of sampled rays, int

        # Step 1: sampling depth value along each given ray
        if target_d is not None:  # default
            z_samples = torch.linspace(-self.config["training"]["range_d"], self.config["training"]["range_d"], steps=self.config["training"]["n_range_d"]).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d  # Tensor(n_rays, n_range_d), device=cuda:0
            z_samples[target_d.squeeze() <= 0] = torch.linspace(self.config["cam"]["near"], self.config["cam"]["far"], steps=self.config["training"]["n_range_d"]).to(target_d)  # for those sampled pixels who have no gt depth values

            if self.config["training"]["n_samples_d"] > 0:  # default (96)
                z_vals = torch.linspace(self.config["cam"]["near"], self.config["cam"]["far"], self.config["training"]["n_samples_d"])[None, :].repeat(n_rays, 1).to(rays_o)  # Tensor(n_rays, n_samples_d), device=cuda:0
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # Tensor(n_rays, n_range_d + n_samples_d), device=cuda:0
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config["cam"]["near"], self.config["cam"]["far"], self.config["training"]["n_samples"]).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Step 2: perturb sampling depths
        if self.config["training"]["perturb"] > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Step 3: do inference and volume rendering
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # Tensor(N_rays, N_samples, 3)
        raw = self.run_network(pts)  # Tensor(N_rays, N_samples, 3), device=cuda:0
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals)

        ret = {"rgb": rgb_map, "depth": depth_map, "disp_map": disp_map, "acc_map": acc_map, "depth_var": depth_var}
        ret = {**ret, "z_vals": z_vals}

        ret["raw"] = raw
        return ret


    def forward(self, rays_o, rays_d, target_rgb, target_d, EMD_w=0.01):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config["cam"]["depth_trunc"])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config["training"]["rgb_missing"]  # Tensor(N, 1), dtype=torch.bool

        # Get render loss
        rgb_loss = compute_loss(rend_dict["rgb"] * rgb_weight, target_rgb * rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
        z_vals = rend_dict["z_vals"]  # Tensor(N_rand, N_samples + N_importance)
        sdf = rend_dict["raw"][..., 3]  # Tensor(N_rand, N_samples + N_importance)
        sdf_prob = rend_dict["raw"][..., 5:]  # Tensor(N_rand, N_samples + N_importance, class_num)
        truncation = self.config["training"]["trunc"] * self.config["data"]["sc_factor"]

        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, sdf_prob, truncation, 5, EMD_w, "l2")

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
        }

        return ret
