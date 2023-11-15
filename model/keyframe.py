import torch
import numpy as np
import random

from helper_functions.sampling_helper import pixel_rc_to_indices, sample_pixels_uniformly
from helper_functions.geometry_helper import extract_first_kf_pose, compute_surface_center


class KeyFrameDatabase(object):
    # @param num_kf: max keyframe number;
    def __init__(self, config, H, W, num_kf, device) -> None:
        self.config = config
        self.keyframes = {}
        self.device = device

        self.frame_ids = None  # frame_Id of each keyframe, Tensor(N, )
        self.collected_kf_num = torch.zeros((1, ), dtype=torch.int64).share_memory_()
        self.H = H
        self.W = W

        self.n_rays_h = self.config["sampling"]["kf_n_rays_h"]  # number of rows for saving a keyframe downsampled
        self.n_rays_w = self.config["sampling"]["kf_n_rays_w"]  # number of cols for saving a keyframe downsampled
        self.num_rays_to_save = self.n_rays_h * self.n_rays_w  # pixel num saved for each keyframe
        self.row_indices, self.col_indices = sample_pixels_uniformly(self.H, self.W, self.n_rays_h, self.n_rays_w)  # get downsampled pixels (row_Ids, col_Ids)
        self.rays = torch.zeros( (num_kf, self.num_rays_to_save, 7) )  # [direction, rgb, depth]

        # keyframe, localMLP related vars
        self.create_MLP_data(num_kf)
        self.create_overlapping_pts_data()


    def __len__(self):
        return len(self.frame_ids)


    def get_length(self):
        return self.__len__()


    # @brief: create vars related to localMLPs
    def create_MLP_data(self, num_kf):
        num_localMLP = self.config["mapping"]["localMLP_num"]
        # this tensor records the information of each localMLP:
        # (1) col[0:1]: whether this localMLP is created and used (1/0);
        # (2) col[1:4]: xyz center (in World Coordinate System);
        # (3) col[5:]: xyz axis-aligned length (in World Coordinate System)
        self.localMLP_info = torch.zeros( (num_localMLP, 7) ).share_memory_()

        self.localMLP_max_len = torch.tensor(self.config["mapping"]["localMLP_max_len"])[None, ...].repeat((num_localMLP, 1)).share_memory_()  # Tensor(num_localMLP, 3)
        self.localMLP_adjacent = torch.zeros( (self.config["mapping"]["localMLP_num"], self.config["mapping"]["localMLP_num"]) ).share_memory_()  # whether 2 localMLPs are adjacent(0/1), Tensor(num_localMLP, num_localMLP)

        # this tensor records the related localMLP_Id of each keyframe (-1 means none)
        self.keyframe_localMLP = torch.full( (num_kf, 2), fill_value=-1 ).share_memory_()

        # keyframe_Id of each submap's first keyframe
        self.localMLP_first_kf = torch.full( (num_localMLP, ), fill_value=-1 ).share_memory_()

        self.keyframe_mutex_mask = torch.zeros((1, num_kf), dtype=torch.int64).share_memory_()


    # @brief: create vars for loop closure optimization
    def create_overlapping_pts_data(self):
        self.ovlp_rays_h = self.config["mapping"]["overlapping"]["n_rays_h"]
        self.ovlp_rays_w = self.config["mapping"]["overlapping"]["n_rays_w"]
        self.ovlp_depth = torch.zeros( (1, self.ovlp_rays_h * self.ovlp_rays_w) ).share_memory_()
        self.ovlp_rays = torch.zeros( (1, self.ovlp_rays_h * self.ovlp_rays_w, 3) ).share_memory_()
        self.ovlp_pts_mask = torch.zeros( (1, self.ovlp_rays_h * self.ovlp_rays_w), dtype=torch.bool ).share_memory_()

        self.near_kf_num = 10
        self.nearest_kf_Ids = torch.full( (1, self.near_kf_num), fill_value=-1, dtype=torch.int64 ).share_memory_()
        self.nearest_kf_mask = torch.zeros( (1, self.near_kf_num, self.ovlp_rays_h * self.ovlp_rays_w) ).share_memory_()


    # @brief: Sampling strategy for current keyframe rays
    # @param rays: Tensor(1, H * W, 7);
    def sample_single_keyframe_rays(self, rays, option='uniform'):
        if option == 'random':
            idxs = random.sample(range(0, self.H * self.W), self.num_rays_to_save)
        elif option == 'uniform':  # default
            idxs = pixel_rc_to_indices(self.row_indices, self.col_indices, self.H, self.W)
        elif option == 'filter_depth':
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.config["cam"]["depth_trunc"])
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, 7]
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), self.num_rays_to_save)
        else:
            raise NotImplementedError()
        rays = rays[:, idxs]  # Tensor(1, self.num_rays_to_save, 7), device=cpu
        return rays


    # @brief: modify self.keyframe_localMLP;
    # @param kf_Id: Tensor(, );
    # @param localMLP_Id1: Tensor(, );
    # @param localMLP_Id2: Tensor(, ), default: None.
    def add_keyframe_localMLP(self, kf_Id, localMLP_Id1, localMLP_Id2=None):
        if localMLP_Id2 is None:
            if self.keyframe_localMLP[kf_Id][0] == -1:
                self.keyframe_localMLP[kf_Id][0] = localMLP_Id1
            else:
                self.keyframe_localMLP[kf_Id][1] = localMLP_Id1
        else:
            self.keyframe_localMLP[kf_Id][0] = localMLP_Id1
            self.keyframe_localMLP[kf_Id][1] = localMLP_Id2


    # @brief: modify self.localMLP_info and self.localMLP_first_kf (if needed)
    # @param localMLP_Id: Tensor(, );
    # @param localMLP_center: Tensor(3, );
    # @param localMLP_len: Tensor(3, );
    def modify_localMLP_info(self, localMLP_Id, localMLP_center, localMLP_len):
        self.localMLP_info[localMLP_Id][1:4] = localMLP_center
        self.localMLP_info[localMLP_Id][4:7] = localMLP_len


    # @brief: set localMLP_Id1 and localMLP_Id2 are adjacent localMLPs (having overlapping keyframes)
    def add_adjcent_pair(self, localMLP_Id1, localMLP_Id2):
        if localMLP_Id1 is not None and localMLP_Id2 is not None:
            self.localMLP_adjacent[localMLP_Id1][localMLP_Id2] = 1
            self.localMLP_adjacent[localMLP_Id2][localMLP_Id1] = 1


    # @brief: find all adjacent localMLP pairs;
    #-@return adja_pairs: all adjacent localMLP pairs, each line represents a pair, Tensor(n, 2);
    #-@return part_localMLP: localMLP_Ids of all participated localMLPs, Tensor(m, ).
    def find_adjacent_localMLP_pair(self):
        localMLP_num = self.localMLP_adjacent.shape[0]
        adja_pairs = []
        part_localMLP = []
        for i in range(localMLP_num):
            for j in range(localMLP_num):
                if j <= i:
                    continue
                if self.localMLP_adjacent[i][j] > 0:
                    adja_pairs.append( torch.tensor([j, i], dtype=torch.int32) )
                    if i not in part_localMLP:
                        part_localMLP.append(i)
                    if j not in part_localMLP:
                        part_localMLP.append(j)
        adja_pairs = torch.stack(adja_pairs, 0)  # Tensor(n, 2)
        adja_pairs = torch.sort(adja_pairs, -1)[0]
        part_localMLP = torch.tensor(part_localMLP)  # localMLP_Ids of all participated localMLPs, Tensor(m, )
        part_localMLP = torch.sort(part_localMLP)[0]
        return adja_pairs, part_localMLP


    # @brief: fill localMLP info and localMLP's first keyframe for a newly created localMLP;
    #-@return: Tensor(, ).
    def modify_new_localMLP_info(self, localMLP_center, localMLP_len, kf_Id):
        new_localMLP_Id = torch.count_nonzero(self.localMLP_info[:, 0])  # next available localMLP_Id
        updated_line = torch.cat( [torch.ones((1, )), localMLP_center, localMLP_len], 0 )  # Tensor(7, )
        if new_localMLP_Id < self.localMLP_info.shape[0]:
            self.localMLP_info[new_localMLP_Id] = updated_line
            self.localMLP_first_kf[new_localMLP_Id] = kf_Id
        else:
            kf_Id_tensor = torch.tensor([kf_Id], dtype=torch.int64)
            self.localMLP_info = torch.cat([self.localMLP_info, updated_line.unsqueeze(0)], 0).share_memory_()
            self.localMLP_first_kf = torch.cat([self.localMLP_first_kf, kf_Id_tensor], 0).share_memory_()

            localMLP_max_len_new = torch.tensor(self.config["mapping"]["localMLP_max_len"])[None, ...]
            self.localMLP_max_len = torch.cat([self.localMLP_max_len, localMLP_max_len_new], 0).share_memory_()

            localMLP_adjacent_new_l = torch.zeros_like(self.localMLP_adjacent[-1].unsqueeze(0))
            self.localMLP_adjacent = torch.cat([self.localMLP_adjacent, localMLP_adjacent_new_l], 0)
            localMLP_adjacent_new_c = torch.zeros_like(self.localMLP_adjacent[:, -1].unsqueeze(0))
            self.localMLP_adjacent = torch.cat([self.localMLP_adjacent, localMLP_adjacent_new_c], -1).share_memory_()
        return new_localMLP_Id


    # @brief: insert the frame_Id (of a keyframe) to list
    def attach_ids(self, frame_ids):
        if self.frame_ids is None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = torch.cat([self.frame_ids, frame_ids], dim=0)


    # @brief: Add keyframe rays to the keyframe database: (1) add frame_Id of keyframe; (2) store the rays
    def add_keyframe(self, batch, filter_depth=False):
        rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])  # Tensor(1, H * W, 7), device=cpu
        if filter_depth:
            rays = self.sample_single_keyframe_rays(rays, 'filter_depth')
        else:  # default
            rays = self.sample_single_keyframe_rays(rays)  # Tensor(1, num_rays_to_save, 7), device=cpu
        
        if not isinstance(batch['frame_id'], torch.Tensor):  # default: skip
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.attach_ids(batch['frame_id'])

        # Store the rays
        self.rays[len(self.frame_ids)-1] = rays


    # @brief: find all keyframes which relates to active localMLP and an inactive localMLP;
    # @param active_localMLP_Id: Tensor(, );
    # @param keyframe_ref: Tensor(N_kf, );
    #-@return: Tensor(kf_num, ), 0/1.
    def update_mutex_mask(self, active_localMLP_Id, keyframe_ref, kf_num):
        mask1 = torch.where(keyframe_ref[:kf_num] == -2, torch.ones_like(keyframe_ref[:kf_num]), torch.zeros_like(keyframe_ref[:kf_num])).to(torch.bool)  # Tensor(kf_num, )

        mask2_1 = self.keyframe_localMLP[:kf_num, 0] == active_localMLP_Id
        mask2_2 = self.keyframe_localMLP[:kf_num, 1] == active_localMLP_Id
        mask2 = torch.logical_or(mask2_1, mask2_2)

        final_mask = -1 * torch.logical_and(mask1, mask2).to(torch.int64)  # Let ActiveMap process optimize first
        self.keyframe_mutex_mask[0][:kf_num] = final_mask


    # @brief: get corresponding localMLP_Id of eacg keyframe
    def get_kf_localMLP_Id(self):
        condition1 = torch.where(self.keyframe_localMLP[:, 0] < 0, torch.ones_like(self.keyframe_localMLP[:, 0]), torch.zeros_like(self.keyframe_localMLP[:, 0]))
        condition2 = torch.where(self.keyframe_localMLP[:, 1] < 0, torch.ones_like(self.keyframe_localMLP[:, 1]), torch.zeros_like(self.keyframe_localMLP[:, 1]))
        condition_flag = condition1 + condition2  # Tensor(N_kf, ), dtype=tf.int32
        selected_localMLP_Id = torch.where(condition_flag == 0, self.keyframe_localMLP[:, 1], self.keyframe_localMLP[:, 0])  # Tensor(N_kf, ), dtype=tf.int32
        selected_localMLP_Id = torch.where(selected_localMLP_Id >= 0, selected_localMLP_Id, torch.zeros_like(selected_localMLP_Id))  # Tensor(N_kf, ), dtype=tf.int32
        return selected_localMLP_Id


    # @brief: extract a given localMLP's first keyframe pose in World Coordinate System;
    # @param kf_localMLP_Ids: corresponding localMLP_Id of each keyframe, Tensor(n, )/Tensor(, );
    # @param kf_poses: tensor stored world of all first keyframes and overlapping keyframes, Tensor(n, 4, 4);
    #-@return first_kf_pose: first keyframes' poses of given localMLPs, Tensor(n, 4, 4)/Tensor(4, 4);
    #-@return first_kf_Ids: first keyframes' kf_Id of given localMLPs, Tensor(n, )/Tensor(, ).
    def extract_first_kf_pose(self, kf_localMLP_Ids, kf_poses):
        first_kf_Ids = self.localMLP_first_kf[kf_localMLP_Ids]  # first keyframe's kf_Id of each localMLP, Tensor(n, )/Tensor(, )
        first_kf_pose = kf_poses[first_kf_Ids]  # first keyframe's pose(c2w, in World Coordinate System) of each localMLP, Tensor(n, 4, 4)/Tensor(4, 4)
        return first_kf_pose, first_kf_Ids


    # # @brief: get poses in World Coordinate System of selected keyframes;
    # # @param kf_Ids: Tensor(n, );
    # # @param localMLP_idx: the related localMLP index that given local pose belongs to, Tensor(n, 1);
    # # @param keyframe_ref: Tensor(n, );
    # # @param est_c2w_data: local pose of each given keyframe, Tensor(n, 4, 4);
    # #-@return poses_world: Tensor(n, 4, 4).
    # def extract_kf_world_poses(self, kf_Ids, localMLP_idx, keyframe_ref, est_c2w_data, kf_c2w):
    #     kf_frame_Ids = kf_Ids * self.config["mapping"]["keyframe_every"]  # Tensor(n, )
    #     ref_localMLP_Id = torch.gather(self.keyframe_localMLP[kf_Ids], 1, localMLP_idx).squeeze(-1)  # corresponding localMLP_Id of each selected keyframe, Tensor(n, )
    #     first_poses = extract_first_kf_pose(ref_localMLP_Id, self.localMLP_first_kf, kf_c2w)  # corresponding localMLP's pose of each selected keyframe, Tensor(n, 4, 4)
    #
    #     first_kf_pose_world = kf_c2w[kf_Ids]
    #     poses_local = est_c2w_data[kf_frame_Ids]
    #     keyframe_ref = keyframe_ref[..., None, None].to(self.device)  # Tensor(n, 1, 1)
    #     poses_world = torch.where(keyframe_ref == -1, first_kf_pose_world, first_poses @ poses_local)
    #     return poses_world

    # @brief: get poses in World Coordinate System of selected keyframes;
    # @param kf_Ids: Tensor(n, );
    # @param localMLP_idx: the related localMLP index that given local pose belongs to, Tensor(n, 1);
    # @param keyframe_ref: Tensor(n, );
    # @param est_c2w_data: local pose of each given keyframe, Tensor(n, 4, 4);
    # @param first_kf_pose: each localMLP's first kf pose(in World Coordinate System), Tensor(localMLP_num, 4, 4);
    #-@return poses_world: Tensor(n, 4, 4).
    def extract_kf_world_poses(self, kf_Ids, localMLP_idx, keyframe_ref, est_c2w_data, first_kf_pose):
        kf_frame_Ids = kf_Ids * self.config["mapping"]["keyframe_every"]  # Tensor(n, )
        ref_localMLP_Id = torch.gather(self.keyframe_localMLP[kf_Ids], 1, localMLP_idx).squeeze(-1)  # corresponding localMLP_Id of each selected keyframe, Tensor(n, )
        first_poses = first_kf_pose[ref_localMLP_Id]  # corresponding localMLP's pose of each selected keyframe, Tensor(n, 4, 4)

        poses_local = est_c2w_data[kf_frame_Ids]
        keyframe_ref = keyframe_ref[..., None, None].to(self.device)  # Tensor(n, 1, 1)
        poses_world = torch.where(keyframe_ref == -1, first_poses, first_poses @ poses_local)
        return poses_world


    # @brief: find keyframes which are bound to more than 1 localMLPs
    def find_ovlp_kf_Ids(self, kf_num=None):
        if kf_num is None:
            kf_num = self.collected_kf_num[0].clone()
        keyframe_localMLP = self.keyframe_localMLP[:kf_num, :].clone()
        condition1 = torch.where( keyframe_localMLP[:, 0] >= 0, torch.ones_like(keyframe_localMLP[:, 0]), torch.zeros_like(keyframe_localMLP[:, 0]) )
        condition2 = torch.where(keyframe_localMLP[:, 1] >= 0, torch.ones_like(keyframe_localMLP[:, 0]), torch.zeros_like(keyframe_localMLP[:, 0]))
        ovlp_kf_Ids = torch.where(condition1 * condition2 > 0)[0]
        return ovlp_kf_Ids


    # @brief: giving a keyframe, and other related keyframe_Ids, compute center distance between this keyframe and each related keyframe;
    # @param kf_center: Tensor(3, );
    # @param related_kf_Ids: keyframe_Ids of related keyframes, Tensor(n, );
    # @param related_kf_pose: world poses of related keyframes, Tensor(n, 4, 4);
    #-@return: distances, Tensor(n, ).
    def sort_center_dist_kf(self, kf_center, related_kf_Ids, related_kf_pose):
        # Step 1: compute surface centers of each related keyframe (in Camera Coordinate System)
        related_rays = self.rays[related_kf_Ids]  # Tensor(n, self.num_rays_to_save, 7)
        surface_centers = compute_surface_center(related_rays)  # Tensor(n, 3)

        # Step 2: cam coords --> world coords
        related_kf_rot = related_kf_pose[:, :3, :3]  # rot mat w2c, Tensor(n, 3, 3)
        related_kf_trans = related_kf_pose[:, :3, 3]  # trans vec w2c, Tensor(n, 3)
        rotated_pts = torch.sum(surface_centers[:, None, :] * related_kf_rot, -1)  # Tensor(n, 3)
        transed_pts = rotated_pts + related_kf_trans

        # Step 3: compute center distance between given keyframe and each related keyframe
        dists = torch.norm(transed_pts - kf_center[None, ...], dim=-1)  # Tensor(n, )
        return dists


    # @brief: Sample rays from self.rays as well as frame_ids
    # @param bs:
    #-@return sample_rays: sampled rays, Tensor(bs, 7);
    #-@return kf_ids: corresponding keyframe_Id of each ray, Tensor(bs, ).
    def sample_global_rays(self, bs):
        num_kf = self.get_length()  # collected keyframe num so far, int
        idxs = torch.tensor( random.sample(range(num_kf * self.num_rays_to_save), bs) )  # ray_Id of sampled rays, Tensor(bs, ), device=cpu
        sample_rays = self.rays[:num_kf].reshape(-1, 7)[idxs]  # Tensor(bs, 7), device=cpu

        kf_ids = torch.div(idxs, self.num_rays_to_save, rounding_mode="floor")  # corresponding keyframe_Id of each ray, Tensor(bs, )
        frame_ids = self.frame_ids[kf_ids]

        return sample_rays, kf_ids


    # @brief: Sample rays from given keyframes;
    # @param kf_Ids: keyframe_Ids sampled from, Tensor(n, );
    # @param bs: pixel num to sample, int;
    #-@return sample_rays: sampled rays, Tensor(bs, 7);
    #-@return kf_indices: corresponding keyframe indices of each ray, Tensor(bs, ).
    def sample_rays_from_given(self, kf_Ids, bs):
        num_kf = kf_Ids.shape[0]
        idxs = torch.tensor( random.sample( range(num_kf * self.num_rays_to_save), bs ) )  # ray_Id of sampled rays, Tensor(bs, ), device=cpu
        sample_rays = self.rays[kf_Ids].reshape(-1, 7)[idxs]  # Tensor(bs, 7), device=cpu

        kf_indices = torch.div(idxs, self.num_rays_to_save, rounding_mode="floor")  # corresponding keyframe indices of each ray, Tensor(bs, )
        return sample_rays, kf_indices


    #-@return: Tensor(num_kf, ), 0/1.
    def get_related_keyframes(self, localMLP_Id, num_kf):
        keyframe_localMLP = self.keyframe_localMLP[:num_kf, :]
        related_mat = torch.where(keyframe_localMLP == localMLP_Id, torch.ones_like(keyframe_localMLP), torch.zeros_like(keyframe_localMLP))
        keyframes_mask = torch.sum(related_mat, dim=-1)  # Tensor(num_kf, )
        return keyframes_mask


    def get_related_keyframes2(self, localMLP_Id, num_kf, localMLP_Id_exclude):
        keyframe_localMLP = self.keyframe_localMLP[:num_kf, :]

        # Step 1: filter out all keyframes that relates to localMLP_Id
        related_mat1 = torch.where(keyframe_localMLP == localMLP_Id, torch.ones_like(keyframe_localMLP), torch.zeros_like(keyframe_localMLP))
        keyframes_mask1 = torch.sum(related_mat1, dim=-1)  # Tensor(num_kf, )

        # Step 2: filter out all keyframes that relates to localMLP_Id_exclude
        related_mat2 = torch.where(keyframe_localMLP == localMLP_Id_exclude, torch.ones_like(keyframe_localMLP), torch.zeros_like(keyframe_localMLP))
        keyframes_mask2 = torch.sum(related_mat2, dim=-1)  # Tensor(num_kf, )

        keyframes_mask = torch.logical_and( keyframes_mask1.to(torch.bool), torch.logical_not(keyframes_mask2) )
        return keyframes_mask


    # @brief: giving a localMLP_Id and keyframe-localMLP relationships of some keyframes, for each keyframe, judge the given localMLP is its first or second related localMLP;
    # @param keyframe_localMLP: Tensor(n, 2);
    # @param localMLP_Id: Tensor(, );
    #-@return: Tensor(n, ), -1/0/1.
    def get_related_localMLP_index(self, keyframe_localMLP, localMLP_Id):
        col1_mask = torch.where(keyframe_localMLP[:, 0] == localMLP_Id, torch.ones_like(keyframe_localMLP[:, 0]), torch.zeros_like(keyframe_localMLP[:, 0]))
        col2_mask = torch.where(keyframe_localMLP[:, 1] == localMLP_Id, 2*torch.ones_like(keyframe_localMLP[:, 1]), torch.zeros_like(keyframe_localMLP[:, 1]))
        idx = torch.stack([col1_mask, col2_mask], dim=-1)  # Tensor(n, 2)
        hit_idx = torch.max(idx, -1)[0] - 1  # 0/1: given localMLP is this keyframe's first/second related localMLP; -1: given localMLP is not related to this keyframe
        return hit_idx


    # @brief: convert selected keyframes' local pose(meybe in its first or second related localMLP's CS) to local pose in given localMLP's Local Coordinate System;
    # @param keyframe_localMLP: keyframe-localMLP relationships of selected keyframes, Tensor(n, 2);
    # @param hit_idx: given localMLP is this keyframe's first/second related localMLP(0/1), Tensor(n, );
    # @param given_first_kf_pose: first keyframe pose(in World Coordinate System) of given localMLP, Tensor(4, 4)
    # @param poses_local: Tensor(n, 4, 4);
    #-@return: selected keyframes' local poses in given localMLP's Local Coordinate System, Tensor(n, 4, 4);
    def convert_given_local_pose(self, keyframe_localMLP, hit_idx, kf_poses, given_first_kf_pose, poses_local):
        hit_idx = hit_idx.to(self.device)  # Tensor(n, )
        first_kf_poses = extract_first_kf_pose(keyframe_localMLP[:, 0], self.localMLP_first_kf, kf_poses)  # first keyframe pose of each keyframe's first-related localMLP, Tensor(n, 4, 4)
        given_first_kf_pose_inv = given_first_kf_pose.inverse().unsqueeze(0)  # Tensor(1, 4, 4)

        pose_local_transed = given_first_kf_pose_inv @ first_kf_poses @ poses_local
        poses_local_given = torch.where(hit_idx[..., None, None] == 0, poses_local, pose_local_transed)
        return poses_local_given


    # @brief: get world pose of given keyframes;
    # @param keyframe_Ids: Tensor(n, );
    # @param keyframe_ref: Tensor(n, );
    # @param kf_poses: Tensor(n_kf, 4, 4);
    # @param poses_local: Tensor(n, 4, 4);
    #-@return: world pose of asking keyframes, Tensor(n, 4, 4).
    def convert_given_world_pose(self, keyframe_Ids, keyframe_ref, kf_poses, poses_local):
        first_kf_poses = extract_first_kf_pose(self.keyframe_localMLP[keyframe_Ids][:, 0], self.localMLP_first_kf, kf_poses)  # first keyframe pose of each keyframe's first-related localMLP, Tensor(n, 4, 4)
        pose_world_trans = first_kf_poses @ poses_local

        pose_world = kf_poses[keyframe_Ids]  # Tensor(n, 4, 4)
        pose_world_final = torch.where(keyframe_ref[..., None, None] == -1, pose_world, pose_world_trans)
        return pose_world_final


    # @brief:
    # @param localMLP_Id: Tensor(, );
    # @param num_kf: number of keyframes collected so far, int;
    # @param overlap_kf_flag: Tensor(num_kf, );
    # @param process_flag: Process_flag of invoking process (1: ActiveMap process, -1: InactiveMap process), int;
    #-@return: Tensor(num_kf, ).
    def get_related_keyframes_exclude(self, localMLP_Id, num_kf, overlap_kf_flag, process_flag):
        # Step 1: find all related keyframes
        keyframe_localMLP = self.keyframe_localMLP[:num_kf, :]
        related_mat = torch.where(keyframe_localMLP == localMLP_Id, torch.ones_like(keyframe_localMLP), torch.zeros_like(keyframe_localMLP))
        keyframes_mask = torch.sum(related_mat, dim=-1)  # Tensor(num_kf, ), 0/1

        # Step 2: excluded related overlapping keyframes, which are optimized by given process last time
        if torch.count_nonzero(overlap_kf_flag[:num_kf]) > 0:
            condition = (overlap_kf_flag[:num_kf] == process_flag)
            overlap_mask = torch.where(condition, torch.zeros_like(keyframes_mask), torch.ones_like(keyframes_mask))
            keyframes_mask = overlap_mask * keyframes_mask

        return keyframes_mask


    # @brief: sample rays in a submap globally (which will sample pixels from first and last keyframes individually);
    # @param first_kf_Id: keyframe_Id of this localMLP's first keyframe, Tensor(, );
    # @param related_kf_ids: keyframe_Ids of this localMLP's related keyframes, Tensor(n', );
    # @param pix_num:
    # -@return sampled_rays: sampled rays from related keyframes, Tensor(pix_num, 7);
    # -@return kf_ids: corresponding keyframe_Id of each sampled ray, Tensor(pix_num, ).
    def sample_rays_in_submap(self, first_kf_Id, related_kf_ids, pix_num):
        # Step 1: find all keyframes related to given localMLP
        related_kf_num = related_kf_ids.shape[0]

        # Step 2: sample rays from first keyframe and other related keyframes respectively
        # 2.1: sampling from first keyframe
        pix_num_first = max(pix_num // related_kf_num,  pix_num // 10)
        idx_first = torch.tensor(random.sample(range(self.num_rays_to_save), pix_num_first))  # ray_Id of sampled rays, Tensor(pix_num_first, )
        first_rays = self.rays[first_kf_Id].reshape((-1, 7))[idx_first]  # Tensor(pix_num_first, 7)
        first_kf_indices = torch.zeros_like(idx_first)  # corresponding keyframe infices of each ray sampled from first keyframe, Tensor(pix_num_first, )
        first_kf_ids = torch.ones_like(idx_first) * first_kf_Id  # corresponding keyframe infices of each ray sampled from first keyframe, Tensor(pix_num_first, )

        if related_kf_num > 1:
            if related_kf_num > 2:
                # 2.2: sampling from latest keyframe
                last_kf_Id = related_kf_ids[-1]
                pix_num_last = max(pix_num // related_kf_num,  pix_num // 5)
                idx_last = torch.tensor(random.sample(range(self.num_rays_to_save), pix_num_last))  # ray_Id of sampled rays, Tensor(pix_num_last, )
                last_rays = self.rays[last_kf_Id].reshape((-1, 7))[idx_last]  # Tensor(pix_num_last, 7)
                last_kf_indices = torch.ones_like(idx_last) * (related_kf_num - 1)  # corresponding keyframe infices of each ray sampled from last keyframe, Tensor(pix_num_last, )
                last_kf_ids = torch.ones_like(idx_last) * last_kf_Id  # corresponding keyframe infices of each ray sampled from first keyframe, Tensor(pix_num_first, )

                other_kf_ids = related_kf_ids[1:-1]
                pix_num_other = pix_num - pix_num_first - pix_num_last
                other_kf_num = related_kf_num - 2
            else:
                other_kf_ids = related_kf_ids[1:]
                pix_num_other = pix_num - pix_num_first
                other_kf_num = related_kf_num - 1

            # 2.3: sampling from other related keyframes (except first keyframe)
            idx_other = torch.tensor(random.sample(range(other_kf_num * self.num_rays_to_save), pix_num_other))  # ray_Id of sampled rays, Tensor(pix_num_other, )
            other_rays = self.rays[other_kf_ids].reshape((-1, 7))[idx_other]  # Tensor(pix_num_other, 7)

            other_kf_indices = torch.div(idx_other, self.num_rays_to_save, rounding_mode="floor")  # corresponding keyframe infices of each ray sampled from other keyframes, Tensor(pix_num_other, )
            other_kf_ids = other_kf_ids[other_kf_indices]  # corresponding keyframe_Id of each ray, Tensor(pix_num_other, )
            other_kf_indices = other_kf_indices + 1

            if related_kf_num > 2:
                sampled_rays = torch.cat([first_rays, other_rays, last_rays], dim=0)
                kf_indices = torch.cat([first_kf_indices, other_kf_indices, last_kf_indices], dim=0)
                kf_ids = torch.cat([first_kf_ids, other_kf_ids, last_kf_ids], dim=0)
            else:
                sampled_rays = torch.cat([first_rays, other_rays], dim=0)
                kf_indices = torch.cat([first_kf_indices, other_kf_indices], dim=0)
                kf_ids = torch.cat([first_kf_ids, other_kf_ids], dim=0)
        else:
            sampled_rays = first_rays
            kf_indices = first_kf_indices
            kf_ids = first_kf_ids
        return sampled_rays, kf_ids, kf_indices


    # @brief: sample rays in given keyframes;
    # @param given_kf_ids: keyframe_Ids of given related keyframes, Tensor(n', );
    # @param pix_num: total pixel number to sample, int;
    #-@return sampled_rays: sampled rays from related keyframes, Tensor(pix_num, 7);
    #-@return kf_ids: corresponding keyframe_Id of each sampled ray, Tensor(pix_num, ).
    def sample_rays_in_given_kf(self, given_kf_ids, pix_num):
        # Step 1: find all keyframes related to given localMLP
        given_kf_num = given_kf_ids.shape[0]

        # Step 2: sample rays from given keyframes
        idx = torch.tensor(random.sample(range(given_kf_num * self.num_rays_to_save), pix_num))  # ray_Id of sampled rays, Tensor(pix_num_other, )
        sampled_rays = self.rays[given_kf_ids].reshape((-1, 7))[idx]  # Tensor(pix_num_other, 7)

        kf_indices = torch.div(idx, self.num_rays_to_save, rounding_mode="floor")  # corresponding keyframe infices of each ray sampled from other keyframes, Tensor(pix_num_other, )
        kf_ids = given_kf_ids[kf_indices]  # corresponding keyframe_Id of each ray, Tensor(pix_num_other, )

        return sampled_rays, kf_ids, kf_indices


    # @brief: extract related vars of a given localMLP_Id;
    # @param localMLP_Id: given localMLP_Id, Tensor(, );
    # @param kf_poses: tensor storing all first keyframes' world poses, Tensor(num_kf, 4, 4);
    # @param est_c2w_data: tensor storing all keyframes' local poses(in its first related localMLP), Tensor(num_frame, 4, 4);
    # @param kf_ref: keyframe ref type of each keyframe, Tensor(num_kf, );
    # @param process_flag: Process_flag of invoking process (1: ActiveMap process, -1: InactiveMap process), int;
    #-@return first_kf_pose: first keyframe's pose in World Coordinate System of given localMLP, Tensor(4, 4);
    #-@return first_kf_Id: first keyframe's kf_Id of given localMLP, Tensor(, );
    #-@return poses_local: local pose (in given localMLP's coordinate system) of related keyframes, Tensor(selected_num_kf, 4, 4);
    #-@return avail_kf_Ids: keyframe_Ids of all available keyframes related to given localMLP, Tensor(selected_num_kf, );
    #-@return avail_kf_frame_Ids: frame_Ids of all available keyframes related to given localMLP, Tensor(selected_num_kf, );
    #-@return avail_kf_ref: Tensor(n', );
    #-@return avail_ovlp_kf_idx: indices of available overlapping keyframes in avail_kf_Ids, Tensor(k', )
    #-@return avail_ovlp_kf_Ids: keyframe_Ids of available overlapping keyframes, Tensor(k', ).
    def extract_localMLP_vars(self, localMLP_Id, kf_poses, est_c2w_data, kf_ref, process_flag):
        num_kf = self.collected_kf_num[0].clone()  # collected keyframe num so far, Tensor(, )
        # Step 1: get overlapping keyframes mutex mask (only those overlapping keyframes which bind to currrent active localMLP and another inactive localMLP have non-zero values)
        ovlp_mutex = self.keyframe_mutex_mask.clone()[0, :num_kf]  # Tensor(num_kf, ), 0/1/-1
        ovlp_mutex_mask = torch.where(ovlp_mutex==process_flag, torch.zeros_like(ovlp_mutex), torch.ones_like(ovlp_mutex))  # Tensor(num_kf, ), 0/1

        # Step 2: find first keyframe of given localMLP (world pose and keyframe_Id)
        first_kf_pose, first_kf_Id = self.extract_first_kf_pose(localMLP_Id, kf_poses)  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )
        first_kf_pose = first_kf_pose.detach()

        # Step 3: find all available keyframes (1.it must be related keyframe; 2.for overlapping keyframe, its last optimization must be done in another process)
        related_kf_mask = self.get_related_keyframes(localMLP_Id, num_kf)  # Tensor(num_kf, ), 0/1
        kf_mask = related_kf_mask * ovlp_mutex_mask  # mask of all available keyframes, Tensor(num_kf, ) 0/1

        avail_kf_Ids = torch.where(kf_mask > 0)[0]  # keyframe_Ids of all available keyframes, Tensor(n', )
        avail_kf_ref = kf_ref[avail_kf_Ids]  # keyframe_ref type of all available keyframes, Tensor(n', ), n(>=0)/-1/-2
        avail_kf_frame_Ids = avail_kf_Ids * self.config["mapping"]["keyframe_every"]  # frame_Ids of all available keyframes, Tensor(n', )

        avail_ovlp_kf_idx = torch.where(avail_kf_ref == -2)[0]  # overlapping keyframes' indices in avail_kf_Ids, Tensor(k', )
        avail_ovlp_kf_Ids = avail_kf_Ids[avail_ovlp_kf_idx]  # keyframe_Ids of all available overlapping keyframes, Tensor(k', )

        # Step 4: extract local pose of all available keyframes in localMLP_Id's Local Coordinate System
        # 4.1: local pose of all ordinary keyframes (whose keyframe_ref >= 0)
        first_pose_local = torch.eye(4).to(self.device)
        poses_local = est_c2w_data[avail_kf_frame_Ids]  # local pose of all related keyframes (indexed by frame_Id)
        poses_local[0] = first_pose_local

        # 4.2: for available keyframes which are first keyframe of another localMLP: firstly extract their world poses, and then convert them to local poses
        ano_first_kf_idx = torch.where( torch.logical_and(avail_kf_Ids != first_kf_Id, avail_kf_ref == -1) )[0]
        if ano_first_kf_idx.shape[0] > 0:
            ano_first_kf_Ids = avail_kf_Ids[ano_first_kf_idx]
            ano_first_kf_poses_world = kf_poses[ano_first_kf_Ids]  # Tensor(m', 4, 4)
            ano_first_kf_poses_local = first_kf_pose.inverse().unsqueeze(0) @ ano_first_kf_poses_world
            poses_local[ano_first_kf_idx] = ano_first_kf_poses_local

        # 4.3: for available keyframes which are overlapping keyframes
        if avail_ovlp_kf_idx.shape[0] > 0:
            ovlp_pose_local = poses_local[avail_ovlp_kf_idx]
            keyframe_localMLP = self.keyframe_localMLP[avail_ovlp_kf_Ids]  # Tensor(k', )
            localMLP_hit_dix = self.get_related_localMLP_index(keyframe_localMLP, localMLP_Id)  # Tensor(k', )
            ovlp_pose_local_given = self.convert_given_local_pose(keyframe_localMLP, localMLP_hit_dix, kf_poses, first_kf_pose, ovlp_pose_local)  # Tensor(k', )
            poses_local[avail_ovlp_kf_idx] = ovlp_pose_local_given

        return first_kf_pose, first_kf_Id, poses_local, avail_kf_Ids, avail_kf_frame_Ids, avail_kf_ref, avail_ovlp_kf_idx, avail_ovlp_kf_Ids


    # @brief: extract related vars of a given localMLP_Id;
    # @param localMLP_Id: given localMLP_Id, Tensor(, );
    # @param kf_poses: tensor storing all first keyframes' world poses, Tensor(num_kf, 4, 4);
    # @param est_c2w_data: tensor storing all keyframes' local poses(in its first related localMLP), Tensor(num_frame, 4, 4);
    # @param kf_ref: keyframe ref type of each keyframe, Tensor(num_kf, );
    # @param process_flag: Process_flag of invoking process (1: ActiveMap process, -1: InactiveMap process), int;
    #-@return first_kf_pose: first keyframe's pose in World Coordinate System of given localMLP, Tensor(4, 4);
    #-@return first_kf_Id: first keyframe's kf_Id of given localMLP, Tensor(, );
    #-@return poses_local: local pose (in given localMLP's coordinate system) of related keyframes, Tensor(selected_num_kf, 4, 4);
    #-@return avail_kf_Ids: keyframe_Ids of all available keyframes related to given localMLP, Tensor(selected_num_kf, );
    #-@return avail_kf_frame_Ids: frame_Ids of all available keyframes related to given localMLP, Tensor(selected_num_kf, );
    #-@return avail_kf_ref: Tensor(n', );
    #-@return avail_ovlp_kf_idx: indices of available overlapping keyframes in avail_kf_Ids, Tensor(k', )
    #-@return avail_ovlp_kf_Ids: keyframe_Ids of available overlapping keyframes, Tensor(k', ).
    def extract_localMLP_vars_given(self, localMLP_Id, given_kf_Ids, kf_poses, est_c2w_data, kf_ref):
        given_kf_Ids = torch.sort(given_kf_Ids)[0]

        # Step 1: find first keyframe of given localMLP (world pose and keyframe_Id)
        first_kf_pose, first_kf_Id = self.extract_first_kf_pose(localMLP_Id, kf_poses)  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )
        first_kf_pose = first_kf_pose.detach()

        # Step 2:
        given_kf_ref = kf_ref[given_kf_Ids]  # keyframe_ref type of all given keyframes, Tensor(n', ), n(>=0)/-1/-2
        given_kf_frame_Ids = given_kf_Ids * self.config["mapping"]["keyframe_every"]  # frame_Ids of all given keyframes, Tensor(n', )

        given_ovlp_kf_idx = torch.where(given_kf_ref == -2)[0]  # overlapping keyframes' indices in given_kf_Ids, Tensor(k', )
        given_ovlp_kf_Ids = given_kf_Ids[given_ovlp_kf_idx]  # keyframe_Ids of all given overlapping keyframes, Tensor(k', )

        # Step 3: extract local pose of all given keyframes in given localMLP_Id's Local Coordinate System
        # 3.1: local pose of all ordinary keyframes (whose keyframe_ref >= 0)
        poses_local = est_c2w_data[given_kf_frame_Ids]  # local pose of all given keyframes (indexed by frame_Id)

        if given_kf_Ids[0] == first_kf_Id:
            first_pose_local = torch.eye(4).to(self.device)
            poses_local[0] = first_pose_local

        # 3.2: for given keyframes which are first keyframe of another localMLP: firstly extract their world poses, and then convert them to local poses
        ano_first_kf_idx = torch.where( torch.logical_and(given_kf_Ids != first_kf_Id, given_kf_ref == -1) )[0]
        if ano_first_kf_idx.shape[0] > 0:
            ano_first_kf_Ids = given_kf_Ids[ano_first_kf_idx]
            ano_first_kf_poses_world = kf_poses[ano_first_kf_Ids]  # Tensor(m', 4, 4)
            ano_first_kf_poses_local = first_kf_pose.inverse().unsqueeze(0) @ ano_first_kf_poses_world
            poses_local[ano_first_kf_idx] = ano_first_kf_poses_local

        # 3.3: for given keyframes which are overlapping keyframes
        if given_ovlp_kf_idx.shape[0] > 0:
            ovlp_pose_local = poses_local[given_ovlp_kf_idx]
            keyframe_localMLP = self.keyframe_localMLP[given_ovlp_kf_Ids]  # Tensor(k', )
            localMLP_hit_dix = self.get_related_localMLP_index(keyframe_localMLP, localMLP_Id)  # Tensor(k', )
            ovlp_pose_local_given = self.convert_given_local_pose(keyframe_localMLP, localMLP_hit_dix, kf_poses, first_kf_pose, ovlp_pose_local)  # Tensor(k', )
            poses_local[given_ovlp_kf_idx] = ovlp_pose_local_given

        return first_kf_pose, first_kf_Id, poses_local, given_kf_Ids, given_kf_frame_Ids, given_kf_ref, given_ovlp_kf_idx, given_ovlp_kf_Ids


    # def sample_global_keyframe(self, window_size, n_fixed=1):
    #     '''
    #     Sample keyframe globally
    #     Window size: limit the window size for keyframe
    #     n_fixed: sample the last n_fixed keyframes
    #     '''
    #     if window_size >= len(self.frame_ids):
    #         return self.rays[:len(self.frame_ids)], self.frame_ids
    #
    #     current_num_kf = len(self.frame_ids)
    #     last_frame_ids = self.frame_ids[-n_fixed:]
    #
    #     # Random sampling
    #     idx = random.sample(range(0, len(self.frame_ids) - n_fixed), window_size)
    #
    #     # Include last n_fixed
    #     idx_rays = idx + list(range(current_num_kf-n_fixed, current_num_kf))
    #     select_rays = self.rays[idx_rays]
    #
    #     return select_rays, torch.cat([self.frame_ids[idx], last_frame_ids], dim=0)
    #
    #
    # @torch.no_grad()
    # def sample_overlap_keyframe(self, batch, frame_id, est_c2w_list, k_frame, n_samples=16, n_pixel=100, dataset=None):
    #     '''
    #     NICE-SLAM strategy for selecting overlapping keyframe from all previous frames
    #
    #     batch: Information of current frame
    #     frame_id: id of current frame
    #     est_c2w_list: estimated c2w of all frames
    #     k_frame: num of keyframes for BA i.e. window size
    #     n_samples: num of sample points for each ray
    #     n_pixel: num of pixels for computing overlap
    #     '''
    #     c2w_est = est_c2w_list[frame_id]
    #
    #     indices = torch.randint( dataset.H * dataset.W, (n_pixel, ) )
    #     rays_d_cam = batch['direction'].reshape(-1, 3)[indices].to(self.device)
    #     target_d = batch['depth'].reshape(-1, 1)[indices].repeat(1, n_samples).to(self.device)
    #     rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:3, :3], -1)
    #     rays_o = c2w_est[None, :3, -1].repeat(rays_d.shape[0], 1).to(self.device)
    #
    #     t_vals = torch.linspace(0., 1., steps=n_samples).to(target_d)
    #     near = target_d * 0.8
    #     far = target_d + 0.5
    #     z_vals = near * (1.-t_vals) + far * (t_vals)
    #     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    #     pts_flat = pts.reshape(-1, 3).cpu().numpy()
    #
    #     key_frame_list = []
    #
    #     for i, frame_id in enumerate(self.frame_ids):
    #         frame_id = int(frame_id.item())
    #         c2w = est_c2w_list[frame_id].cpu().numpy()
    #         w2c = np.linalg.inv(c2w)
    #         ones = np.ones_like(pts_flat[:, 0]).reshape(-1, 1)
    #         pts_flat_homo = np.concatenate([pts_flat, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
    #         cam_cord_homo = w2c@pts_flat_homo  # (N, 4, 1)=(4,4)*(N, 4, 1)
    #         cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    #         K = np.array( [ [self.config['cam']['fx'], .0, self.config['cam']['cx']],
    #                         [.0, self.config['cam']['fy'], self.config['cam']['cy']],
    #                         [.0, .0, 1.0] ] ).reshape(3, 3)
    #         cam_cord[:, 0] *= -1
    #         uv = K@cam_cord
    #         z = uv[:, -1:]+1e-5
    #         uv = uv[:, :2] / z
    #         uv = uv.astype(np.float32)
    #         edge = 20
    #         mask = (uv[:, 0] < self.config['cam']['W'] - edge) * (uv[:, 0] > edge) * \
    #                (uv[:, 1] < self.config['cam']['H'] - edge) * (uv[:, 1] > edge)
    #         mask = mask & (z[:, :, 0] < 0)
    #         mask = mask.reshape(-1)
    #         percent_inside = mask.sum()/uv.shape[0]
    #         key_frame_list.append( {'id': frame_id, 'percent_inside': percent_inside, 'sample_id':i} )
    #
    #     key_frame_list = sorted( key_frame_list, key=lambda i: i['percent_inside'], reverse=True )
    #     selected_keyframe_list = [ dic['sample_id'] for dic in key_frame_list if dic['percent_inside'] > 0.00 ]
    #     selected_keyframe_list = list( np.random.permutation(np.array(selected_keyframe_list))[:k_frame] )
    #
    #     last_id = len(self.frame_ids) - 1
    #
    #     if last_id not in selected_keyframe_list:
    #         selected_keyframe_list.append(last_id)
    #
    #     return self.rays[selected_keyframe_list], selected_keyframe_list