import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d.transforms as transforms

from helper_functions.sampling_helper import sample_pixels_uniformly, sample_pixels_random, sample_valid_pixels_random
from helper_functions.geometry_helper import pose_compose


class RandomOptimizer():
    def __init__(self, cfg, mipsfusion):
        self.cfg = cfg
        self.slam = mipsfusion
        self.dataset = self.slam.dataset
        self.device = self.slam.device

        # parameters related to particle swarm template
        # 6D pose format: [qx, qy, qz, tx, ty, tz], Tensor(6, )
        self.particle_size = self.cfg["tracking"]["RO"]["particle_size"]  # size of particle swarm template, default: 2000
        self.scaling_coefficient1 = self.cfg["tracking"]["RO"]["initial_scaling_factor"]  # initial scaling factor of each axis, default: 0.02

        self.scaling_coefficient2 = self.cfg["tracking"]["RO"]["rescaling_factor"]  # coefficient for update search size, default: 0.5
        self.sdf_weight = 1000.
        self.trunc_value = self.cfg["training"]["trunc"]

        mean, cov = torch.zeros(6), torch.eye(6)

        # PST
        self.pre_sampled_particle = np.random.multivariate_normal(mean, cov, self.particle_size).astype(np.float32)  # pre-sampled PST, ndarray(particle_size, 6)
        self.pre_sampled_particle = torch.from_numpy(self.pre_sampled_particle).to(self.device)
        self.pre_sampled_particle[0, :] = 0
        self.pre_sampled_particle = torch.clamp(self.pre_sampled_particle, -2., 2.)  # TEST

        self.no_rel_trans = torch.tensor([1., 0., 0., 0., 0., 0., 0.]).to(self.device)  # pose representing no transformation happened

        # pre-sampled pixels
        self.iW = self.cfg['tracking']['ignore_edge_W']
        self.iH = self.cfg['tracking']['ignore_edge_H']
        self.rays_dir = self.dataset.rays_d  # dir vector of each pixel (in Camera Frame), Tensor(H, W, 3)

        # self.n_rays = self.cfg["tracking"]["n_rays_h"] * self.cfg["tracking"]["n_rays_w"]  # number of pixels sampled for RO
        self.row_indices, self.col_indices = sample_pixels_uniformly(self.dataset.H, self.dataset.W,
                                                                     self.cfg["tracking"]["RO"]["n_rows"], self.cfg["tracking"]["RO"]["n_cols"])

        # camera parameters
        self.fx, self.fy, self.cx, self.cy = self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy
        self.intrinsic = torch.tensor( [ [self.fx,      0., self.cx],
                                         [     0., self.fy, self.cy],
                                         [     0.,      0.,      1.] ] ).to(self.device)

    # @brief: convert a batch of 6D poses to 7D poses(quaternion + translation vector);
    # @param batch_pose: N 6D poses: [qx, qy, qz, tx, ty, tz], Tensor(N, 6);
    #-@return: N 7D pose: [qw, qx, qy, qz, tx, ty, tz], Tensor(N, 7).
    def pose_6D_to_7D(self, batch_pose):
        imag_sq_sum = batch_pose[:, 0] ** 2 + batch_pose[:, 1] ** 2 + batch_pose[:, 2] ** 2  # Tensor(N, )
        sample_qw = torch.where(imag_sq_sum <= 1., torch.sqrt(1 - imag_sq_sum), 0.).unsqueeze(1)  # Tensor(N, 1)

        batch_pose_7D = torch.cat([sample_qw, batch_pose], dim=-1)  # 7D pose(real part first), Tensor(N, 7)
        # batch_pose_7D[:, :4] /= batch_pose_7D[:, :4].norm(dim=-1, keepdim=True)  # ensure union quaternions
        return batch_pose_7D


    # convert poses of each particle(relative pose) to absolute pose;
    # @param ref_pose_rot: Tensor(3, 3);
    # @param ref_pose_trans: Tensor(3, 1);
    # @param particle_template: Tensor(N, 7);
    #-@return abs_rot: absolute pose of each particle(rotation), Tensor(N, 3, 3);
    #-@return abs_trans: absolute pose of each particle(rotation), Tensor(N, 3, 1).
    def get_abs_pose(self, ref_pose_rot, ref_pose_trans, particle_template):
        delta_R = transforms.quaternion_to_matrix(particle_template[:, :4])
        abs_rot = ref_pose_rot @ delta_R  # Tensor(N, 3, 3)
        abs_trans = ref_pose_trans + particle_template[:, 4:, None]  # Tensor(N, 3, 1)
        return abs_rot, abs_trans


    # @brief: transform a bunch of 3D points using a given transformation;
    # @param points: Tensor(m, 3);
    # @param pose_rot:Tensor(N, 3, 3);
    # @param pose_trans: Tensor(N, 3, 1);
    #-@return: Tensor(N, m, 3).
    def batch_points_trans(self, points, pose_rot, pose_trans):
        transed_pts = pose_rot @ torch.transpose(points, 0, 1)
        transed_pts = transed_pts + pose_trans
        transed_pts = torch.transpose(transed_pts, 1, 2)
        return transed_pts


    # @brief: judging whether N pixels are out of range;
    def get_range_mask(self, pixel_coords, img_h, img_w):
        x_coords = pixel_coords[:, 0]  # Tensor(n, )
        y_coords = pixel_coords[:, 1]  # Tensor(n, )

        x_mask1 = torch.where(x_coords > 0., torch.ones_like(x_coords), torch.zeros_like(x_coords))
        x_mask2 = torch.where(x_coords < img_w, torch.ones_like(x_coords), torch.zeros_like(x_coords))
        x_mask = x_mask1 * x_mask2

        y_mask1 = torch.where(y_coords > 0., torch.ones_like(y_coords), torch.zeros_like(y_coords))
        y_mask2 = torch.where(y_coords < img_h, torch.ones_like(y_coords), torch.zeros_like(y_coords))
        y_mask = y_mask1 * y_mask2

        final_mask = x_mask * y_mask
        return final_mask


    # @brief: evaluate fitness value of each particle;
    # param abs_rot: Tensor(N, 3, 3);
    # param abs_trans: Tensor(N, 3, 1);
    # @param last_frame_pose: tracked pose of last frame(T_wl), Tensor(4, 4);
    # @param target_d: depth of selected pixels, Tensor(N, 1);
    # @param rays_d_cam: ray dir of selected pixels, Tensor(N, 3);
    #-@return fitness_value: fitness value of each particle, Tensor(N, );
    #-@return mean_masked_sdf: transformed by each particle(candidate pose)，mean predicted SDF, Tensor(N, ).
    def get_fitness(self, model, abs_rot, abs_trans, last_frame_pose, target_d, rays_d_cam):
        # Step 1: pixel coordinates --> camera coordinates
        # 1.1: compute corresponding 3D points in Camera Coordinate System
        cam_coords = rays_d_cam * target_d  # Tensor(N, 3)

        # 1.2: compute depth mask and overlapping mask
        valid_mask = torch.where(target_d > 0., torch.ones_like(target_d), torch.zeros_like(target_d)).squeeze(-1)[None, ...]  # Tensor(N, 1)

        # Step 2: for each particle (candidate pose), use it to transform sampled 3D points to World coordinates
        world_coords = self.batch_points_trans(cam_coords, abs_rot, abs_trans)  # Tensor(N, n_rays, 3)

        # Step 3: infer these N * n_rays 3D points, get predicted SDF values (metrics: m)
        pred_sdf = model.run_network(world_coords)[..., 3:4].squeeze(-1) * self.trunc_value  # Tensor(N, n_rays)

        # Step 4: for each candidate pose, compute mean predicted SDF value of all valid pixels
        mean_masked_sdf = torch.mean(valid_mask * torch.abs(pred_sdf), dim=-1)  # Tensor(N, )
        fitness_value = mean_masked_sdf * self.sdf_weight  # fitness value of each particle, Tensor(N, )

        return fitness_value, mean_masked_sdf


    # @brief: update pose (starting pose of next round);
    # @param success_flag: whether get non-empty APS, Tensor(, ), dtype=bool;
    # @param rot_cur: Tensor(3, 3);
    # @param trans_cur: Tensor(3, 1);
    # @param mean_transform: weighted mean of APS, Tensor(7, );
    #-@return rot_updated: Tensor(3, 3);
    #-@return trans_updated: Tensor(3, 1).
    def update_cur_pose(self, rot_cur, trans_cur, mean_transform):
        delta_R = transforms.quaternion_to_matrix(mean_transform[:4])  # optimal delta_R of this round, Tensor(3, 3)
        delta_t = mean_transform[4:][..., None]  # optimal delta_t of this round, Tensor(3, 1)

        rot_updated = rot_cur @ delta_R  # R_u = dR @ R_c, Tensor(3, 3)
        trans_updated = trans_cur + delta_t  # t_u = t_c + dt, Tensor(3, 1)
        return rot_updated, trans_updated


    # @brief: according to searching result of this round, update search_size(if APS is empty, mean_transform is all 0);
    # @param mean_pred_sdf: Tensor(, );
    # @param mean_transform_quat: weighted mean of APS( [qx, qy, qz, tx, ty, tz] ), Tensor(6, );
    # -@return: Tensor(1, 6).
    def update_search_size(self, mean_pred_sdf, mean_transform):
        s = torch.abs(mean_transform) + 0.0001  # Tensor(6, )
        search_size = self.scaling_coefficient2 * mean_pred_sdf * s / s.norm() + 0.0001
        return search_size[None, ...]


    # @param depth_img: gt depth image of current frame, Tensor(h, w);
    # @param last_frame_pose: tracked pose of last frame(c2w, Local Coordinate System), Tensor(4, 4);
    # @param n_iter: iter num;
    #-@return: Tensor(4, 4).
    @torch.no_grad()
    def optimize(self, model, depth_img, initial_pose, last_frame_pose, n_iter=10):
        if n_iter <= 0:
            return initial_pose

        rot_cur, trans_cur = initial_pose[:3, :3], initial_pose[:3, 3:]  # Tensor(3, 3) / Tensor(3, 1)
        search_size = self.scaling_coefficient1

        # # pixel sampling
        # # (1) random sampling
        # indice = sample_valid_pixels_random(depth_img[self.iH: -self.iH, self.iW: -self.iW], self.cfg["tracking"]["RO"]["pixel_num"])
        # indice_h, indice_w = torch.remainder(indice, self.dataset.H - self.iH * 2), torch.div(indice, self.dataset.H - self.iH * 2, rounding_mode="floor")
        # target_d = depth_img[self.iH: -self.iH, self.iW: -self.iW][indice_h, indice_w].to(self.device).unsqueeze(-1)  # Tensor(pixel_num, 1)
        # rays_d_cam = self.rays_dir[self.iH: -self.iH, self.iW: -self.iW][indice_h, indice_w, :].to(self.device)  # Tensor(pixel_num, 3)

        # (2) uniform sampling
        indice_h, indice_w = self.row_indices, self.col_indices
        target_d = depth_img[indice_h, indice_w].to(self.device).unsqueeze(-1)  # Tensor(pixel_num, 1)
        rays_d_cam = self.rays_dir[indice_h, indice_w, :].to(self.device)  # Tensor(pixel_num, 3)

        for i in range(n_iter):
            offset = i % 5
            indice_h, indice_w = self.row_indices + offset, self.col_indices + offset
            target_d = depth_img[indice_h, indice_w].to(self.device).unsqueeze(-1)  # Tensor(pixel_num, 1)
            rays_d_cam = self.rays_dir[indice_h, indice_w, :].to(self.device)  # Tensor(pixel_num, 3)

            # Step 1: recover absolute pose for each particle(pose) in template
            # Step 1.1: get delta pose from pre-sampled particles
            rescaled_pst = self.pre_sampled_particle * search_size  # Tensor(N, 6)
            rescaled_pst_7D = self.pose_6D_to_7D(rescaled_pst)  # Tensor(N, 7)

            # Step 1.2: recover to absolute pose
            abs_rot_pst, abs_trans_pst = self.get_abs_pose(rot_cur, trans_cur, rescaled_pst_7D)  # Tensor(N, 3, 3) / Tensor(N, 3, 1)

            # Step 2: *** evaluate (compute fitness value) each particle
            fitness_values, pred_mean_sdf = self.get_fitness(model, abs_rot_pst, abs_trans_pst, last_frame_pose, target_d, rays_d_cam)  # Tensor(N, ) / Tensor(N, )

            # Step 3: filter advanced particle swarm (APS)
            original_fitness = fitness_values[0]
            better_mask = torch.where(fitness_values < original_fitness, torch.ones_like(original_fitness), torch.zeros_like(original_fitness))  # Tensor(N, )
            weights = (original_fitness - fitness_values) * better_mask  # weight of each particle(with mask, 0 for non-advanced particle), Tensor(N, )
            weight_sum = torch.sum(weights) + 0.00001  # Tensor(, )

            success_flag = ( torch.count_nonzero(better_mask) > 0 )
            if success_flag:
                mean_sdf = torch.sum(weights * pred_mean_sdf) / weight_sum  # mean pred SDF of APS, Tensor(, )
            else:
                mean_sdf = pred_mean_sdf[0]

            # Step 4: update R, t
            if success_flag:
                mean_transform = torch.sum(rescaled_pst_7D * weights[:, None], dim=0) / weight_sum  # weighted mean of APS, Tensor(7, )
                mean_transform_quat = mean_transform[:4] / (mean_transform[:4].norm() + 1e-5)  # [qw, qx, qy, qz], Tensor(4, )
                mean_transform = torch.cat([mean_transform_quat, mean_transform[4:]], dim=0)  # weighted mean of APS, Tensor(7, )
                rot_cur, trans_cur = self.update_cur_pose(rot_cur, trans_cur, mean_transform)  # update current pose(starting point of next round) Tensor(3, 3) / Tensor(3, 1)
            else:
                mean_transform = self.no_rel_trans

            # Step 5: rescaling particle swarm template (update search_size)
            search_size_temp = self.update_search_size(mean_sdf, mean_transform[1:])  # Tensor(1, 6)
            search_size = torch.where(success_flag, search_size_temp, search_size_temp * 2).to(self.device)

        tracked_pose = pose_compose(rot_cur, trans_cur)  # Tensor(4, 4)
        return tracked_pose