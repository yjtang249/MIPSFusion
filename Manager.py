import numpy as np
import torch

from helper_functions.geometry_helper import get_frame_surface_bbox, pts_in_bbox, project_to_pixel
from helper_functions.sampling_helper import sample_pixels_uniformly
from helper_functions.printTime import printCurrentDatetime


class Manager():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = SLAM.device
        self.dataset = self.slam.dataset
        self.kfSet = self.slam.kfSet
        self.poseCorrector = self.slam.poseCorrector
        self.cr_threshold = self.config["mapping"]["min_containing_ratio"]
        self.cr_threshold_mo = self.config["mapping"]["min_containing_ratio_mo"]
        self.cr_threshold_back = self.config["mapping"]["min_containing_ratio_back"]
        self.min_cr_localMLP_len = torch.tensor(self.config["mapping"]["min_cr_localMLP_len"])  # Tensor(3, )
        self.localMLP_max_len = torch.tensor(self.config["mapping"]["localMLP_max_len"])

        self.create_loop_vars()
        self.K = torch.tensor([ [self.dataset.fx, 0., self.dataset.cx],
                                [0., self.dataset.fy, self.dataset.cy],
                                [0., 0., 1.] ])


    # @brief: create vars for loop closure triggering
    def create_loop_vars(self):
        self.double_binding_counter = 0
        self.db_active_localMLP_Id = -1
        self.db_mo_localMLP_Id = -1
        self.thres_db_time= 4

        self.wait_loop = False
        self.localMLP_Id_wait = -1
        self.localMLP_Id_actual = -1


    # @brief: judge whether the waiting loop closure actually meets the requirements of trigging;
    #-@return: whether the wait loop can be triggered.
    def get_loop_flag(self, mo_localMLP_Id, active_localMLP_Id, cr_mo, batch, pose_world, force_detect=False):
        if force_detect or ( self.wait_loop and (self.localMLP_Id_wait == mo_localMLP_Id and self.localMLP_Id_actual == active_localMLP_Id) ):
            if cr_mo >= self.cr_threshold_back:
                # if all above conditions are meet, then we should judge whether this keyframe has enough overlapping region with previous keyframes
                switch_flag, target_d, rays_d, pts_mask, top_kf_Ids, top_kf_mask = self.find_overlapping_region(batch, pose_world, active_localMLP_Id, mo_localMLP_Id,
                                                                          self.slam.kf_c2w.detach().cpu(), self.slam.est_c2w_data.detach().cpu(), self.slam.keyframe_ref.detach(),
                                                                          self.config["mapping"]["overlapping"]["n_rays_h"], self.config["mapping"]["overlapping"]["n_rays_w"])

                if switch_flag:
                    self.kfSet.ovlp_depth[0] = target_d
                    self.kfSet.ovlp_rays[0] = rays_d
                    self.kfSet.ovlp_pts_mask[0] = pts_mask
                    self.kfSet.nearest_kf_Ids[0][:top_kf_Ids.shape[0]] = top_kf_Ids
                    self.kfSet.nearest_kf_mask[0][:top_kf_Ids.shape[0], ...] = top_kf_mask
                    self.wait_loop = False
                    return True
        return False


    # @brief: when a keyframe is bound to 2 localMLPs, judge whether accumulate counter or trigger active submap switch
    def process_double_binding(self, active_localMLP_Id, mo_localMLP_Id, cr_mo, batch, pose_world):
        switch_flag = False
        if self.double_binding_counter == 0:
            self.double_binding_counter += 1
            self.db_active_localMLP_Id = active_localMLP_Id
            self.db_mo_localMLP_Id = mo_localMLP_Id
        else:
            if active_localMLP_Id == self.db_active_localMLP_Id and mo_localMLP_Id == self.db_mo_localMLP_Id:
                if self.double_binding_counter >= self.thres_db_time:
                    # binding the same 2 localMLPs for too many times, if current keyframe has enough overlapping region with mo_localMLP, active submap switch will be triggered
                    switch_flag = self.get_loop_flag(mo_localMLP_Id, active_localMLP_Id, cr_mo, batch, pose_world, force_detect=True)
                    if switch_flag:
                        self.double_binding_counter = 0
                    else:
                        # self.double_binding_counter += 1
                        self.double_binding_counter = 0  # 20230816 modified
                else:
                    self.double_binding_counter += 1
            else:
                self.double_binding_counter = 0
                self.db_active_localMLP_Id = active_localMLP_Id
                self.db_mo_localMLP_Id = mo_localMLP_Id
        return switch_flag


    # @brief: convert pose in Local Coordinate System to pose in World Coordinate System;
    # @param pose_local: Tensor(4, 4);
    # @param localMLP_Id: Tensor(, );
    #-@return: Tensor(4, 4).
    def convert_pose_to_world(self, pose_local, localMLP_Id):
        first_kf_pose, _ = self.kfSet.extract_first_kf_pose(localMLP_Id, self.slam.kf_c2w)  # get first keyframe's pose in World Coordinate System, Tensor(1, 4, 4)
        pose_world = first_kf_pose @ pose_local  # Tensor(4, 4)
        return pose_world


    # @brief: convert pose in World Coordinate System to pose in Local Coordinate System;
    # @param pose_world: Tensor(4, 4);
    # @param localMLP_Id: Tensor(, );
    #-@return: Tensor(4, 4).
    def convert_pose_to_local(self, pose_world, localMLP_Id):
        first_kf_pose, _ = self.kfSet.extract_first_kf_pose(localMLP_Id, self.slam.kf_c2w)  # get first keyframe's pose in World Coordinate System, Tensor(1, 4, 4)
        pose_local = first_kf_pose.inverse() @ pose_world  # Tensor(4, 4)
        return pose_local


    # @brief: giving a keyframe, compute the center distance between it and each used localMLP, then sort all used localMLP based on center distance;
    # @param kf_center: the center of given keyframe in World Coordinate System, Tensor(3, )
    # @param used_localMLP_num: int
    def sort_center_dist(self, kf_center, used_localMLP_num):
        localMLP_center = self.kfSet.localMLP_info[:used_localMLP_num, 1:4]  # Tensor(used_localMLP_num, 3)
        dists = torch.norm(localMLP_center - kf_center[None, ...], dim=-1)  # Tensor(used_localMLP_num, )
        return dists


    # @brief: find top K nearest localMLP w.r.t. given keyframe;
    # @param frustum_xyz_center: Tensor(3, );
    #-@return: selected localMLP_Ids, Tensor(k', ).
    def find_nearest_localMLP_topK(self, frustum_xyz_center, k=3):
        used_localMLP_num = int(torch.sum(self.kfSet.localMLP_info[:, 0]).numpy())  # num of used localMLP

        if used_localMLP_num <= k:
            return torch.arange(used_localMLP_num)
        else:
            dists = self.sort_center_dist(frustum_xyz_center, used_localMLP_num)
            near_localMLPs = torch.argsort(dists, 0)
            return near_localMLPs[:k]


    # @brief: find top K nearest localMLP(excluding given localMLP) w.r.t. given keyframe;
    # @param given_localMLP_Id: Id of localMLP which should be excluded, Tensor(, );
    # @param frustum_xyz_center: Tensor(3, );
    #-@return: selected localMLP_Ids, Tensor(k', ).
    def find_nearest_localMLP_topK_exclude(self, given_localMLP_Id, frustum_xyz_center, k=3):
        used_localMLP_num = int(torch.sum(self.kfSet.localMLP_info[:, 0]).numpy())  # num of used localMLP
        avail_localMLP_num = used_localMLP_num - 1  # num of used localMLP excluding active localMLP

        if avail_localMLP_num == 0:
            return torch.arange(used_localMLP_num)
        elif avail_localMLP_num <= k:
            used_localMLP_Ids = torch.arange(used_localMLP_num)
            avail_idx = torch.where(used_localMLP_Ids != given_localMLP_Id)[0]
            return used_localMLP_Ids[avail_idx]
        else:
            dists = self.sort_center_dist(frustum_xyz_center, used_localMLP_num)
            dists[given_localMLP_Id] = 100000.
            near_localMLPs = torch.argsort(dists, 0)
            return near_localMLPs[:k]


    # @brief: given several localMLPs and a depth image, approximately judge the containing ratio of this depth image in each localMLP;
    # @param depth_img: Tensor(H, W);
    # @param rays_d: Tensor(H, W ,3);
    # @param pose_world: Tensor(4, 4);
    # @param localMLP_Ids: localMLP_Ids of selected localMLPs, Tensor(k, );
    #-@return top_localMLP_Id: Tensor(, );
    #-@return containing_ratio: Tensor(, ).
    def find_highest_containing_ratio(self, depth_img, rays_d, pose_world, localMLP_Ids, rays_h=15, rays_w=20, depth_num=11):
        pixel_num = rays_h * rays_w
        k = localMLP_Ids.shape[0]

        # Step 1: sample pixels and points
        # 1.1: sample pixels
        indice_h, indice_w = sample_pixels_uniformly(self.dataset.H, self.dataset.W, rays_h, rays_w)
        target_d = depth_img[indice_h, indice_w]  # gt depth value of each sampled ray, Tensor(pixel_num, )
        rays_d_cam = rays_d[indice_h, indice_w]  # Tensor(pixel_num, 3)

        rays_o = pose_world[:3, -1].repeat(pixel_num, 1)  # apply translation(camera --> world), Tensor(pixel_num, 3)
        rays_d = torch.sum(rays_d_cam[..., None, :] * pose_world[None, :3, :3], -1)  # apply rotation(camera --> world): rotate direction of sampled rays, Tensor(pixel_num, 3)

        # # 1.2: sampling depth values
        # t_scales = torch.linspace(0, 1, depth_num)[None, ...].repeat(pixel_num, 1)  # Tensor(pixel_num, depth_num)
        # t_depth = target_d[..., None] * t_scales  # Tensor(pixel_num, depth_num)

        # 1.3: get sampled 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * target_d[..., :, None]  # Tensor(pixel_num, depth_num, 3)

        # Step 2: judge whether each points in bbox
        center_len = self.kfSet.localMLP_info[localMLP_Ids][:, 1:]  # Tensor(k, 6)
        xyz_min = center_len[:, :3] - 0.5 * center_len[:, 3:]  # Tensor(k, 3)
        xyz_max = center_len[:, :3] + 0.5 * center_len[:, 3:]  # Tensor(k, 3)
        mask = pts_in_bbox(pts.reshape((-1, 3)), xyz_min, xyz_max)  # Tensor(pixel_num * depth_num, k)
        # mask = mask.reshape((pixel_num, depth_num, k))  # Tensor(pixel_num * depth_num, k)
        score = torch.count_nonzero(mask, dim=0)  # Tensor(k, )

        # Step 3: get localMLP with highest score
        top_indices = torch.argsort(score, descending=True)
        top_localMLP_Ids = localMLP_Ids[top_indices]

        # # Step 4: compute containing ratio of top localMLP
        # depth_mask = torch.where(target_d[..., None] > 0., torch.ones_like(t_depth), torch.zeros_like(t_depth))  # Tensor(pixel_num, depth_num)
        # in_mask = mask.reshape((pixel_num, depth_num, k))[..., top_indices[0]].to(depth_mask) * depth_mask  # Tensor(pixel_num, depth_num)
        # containing_ratio = torch.sum(in_mask) / torch.sum(depth_mask)

        return top_localMLP_Ids[0]


    # @brief: giving a depth image with pose, and a localMLP, compute the proportion of surface points contained in the localMLP's range;
    # @param depth_img: Tensor(H, W);
    # @param rays_d: Tensor(H, W, 3);
    # @param pose_world: Tensor(4, 4);
    # @param localMLP_Id: localMLP_Id of selected localMLP, Tensor(1, );
    def compute_containing_ratio(self, depth_img, rays_d, pose_world, localMLP_Id, rays_h=150, rays_w=200, localMLP_center=None, localMLP_len=None):
        pixel_num = rays_h * rays_w

        # Step 1: sample pixels and points
        # 1.1: sample pixels
        indice_h, indice_w = sample_pixels_uniformly(self.dataset.H, self.dataset.W, rays_h, rays_w)
        target_d = depth_img[indice_h, indice_w]  # Tensor(pixel_num, )
        rays_d_cam = rays_d[indice_h, indice_w]  # Tensor(pixel_num, 3)

        rays_o = pose_world[:3, -1].repeat(pixel_num, 1)  # apply translation(camera --> world), Tensor(pixel_num, 3)
        rays_d = torch.sum(rays_d_cam[..., None, :] * pose_world[None, :3, :3], -1)  # apply rotation(camera --> world): rotate direction of sampled rays, Tensor(pixel_num, 3)

        # # 1.2: sampling depth values
        # t_scales = torch.linspace(0, 1, depth_num)[None, ...].repeat(pixel_num, 1)  # Tensor(pixel_num, depth_num)
        # t_depth = target_d[..., None] * t_scales  # Tensor(pixel_num, depth_num)

        # 1.3: get sampled 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * target_d[..., None, None]  # Tensor(pixel_num, depth_num, 3)

        # Step 2: judge whether each points in bbox
        center_len = self.kfSet.localMLP_info[localMLP_Id][1:]  # this localMLP's center and length, Tensor(6, )

        if localMLP_center is None:
            localMLP_center = center_len[:3]

        if localMLP_len is None:
            localMLP_len = center_len[3:]
            localMLP_len = torch.where(localMLP_len < self.min_cr_localMLP_len, self.min_cr_localMLP_len, localMLP_len)

        xyz_min = localMLP_center - 0.5 * localMLP_len  # Tensor(3, )
        xyz_max = localMLP_center + 0.5 * localMLP_len  # Tensor(3, )
        mask = pts_in_bbox(pts.reshape((-1, 3)), xyz_min[None, ...], xyz_max[None, ...])  # Tensor(pixel_num * depth_num, 1)

        # Step 3: compute containing ratio of top localMLP
        depth_mask = torch.where(target_d[..., None] > 0., torch.ones_like(target_d[..., None]), torch.zeros_like(target_d[..., None]))  # Tensor(pixel_num, depth_num)
        mask = mask.to(depth_mask) * depth_mask  # Tensor(pixel_num * depth_num)

        valid_pts_num = torch.count_nonzero(depth_mask)
        pts_in_num = torch.count_nonzero(mask)
        containing_ratio = pts_in_num / valid_pts_num
        return containing_ratio


    # @brief: giving a keyframe which triggers active submap switch to a previous localMLP, find the top K overlapping keyframes,
    #    the intersection of them is overlapping region.
    # @param depth_img: Tensor(H, W);
    # @param rays_d: Tensor(H, W ,3);
    # @param pose_world: pose (c2w) of given keyframe, Tensor(4, 4);
    # @param localMLP_Id: the Id of localMLP which will be switched to, Tensor(, );
    # @param kf_poses: world pose of all first keyframes, Tensor(n_kf, 4, 4);
    # @param keyframe_ref:
    #-@return switch_prev: wheteher overlapping region is enough;
    #-@return target_d: depth of sampled surface points in the overlapping keyframe, Tensor(N, 3);
    #-@return rays_d_cam: dir (in Camera Coordinate System) of sampled surface points in the overlapping keyframe, Tensor(N, 3);
    #-@return mask_final: mask of whether each sampled points are in overlapping region, Tensor(N, );
    #-@return topK_kf_Ids: top K nearest keyframes, Tensor(K, );
    #-@return top_kf_masks: whether each sampled surface is seen to each keyframe, Tensor(K, N).
    def find_overlapping_region(self, batch, pose_world, active_localMLP_Id, localMLP_Id, kf_poses, est_c2w_data, keyframe_ref, rays_h=24, rays_w=32):
        depth_img = batch["depth"].squeeze(0)  # Tensor(H, W)
        rays_d = batch["direction"].squeeze(0)  # Tensor(H, W ,3)
        frame_id = int(batch["frame_id"])
        kf_id = frame_id // self.config["mapping"]["keyframe_every"]
        pixel_num = rays_h * rays_w  # number of sampled pixels
        num_kf = self.kfSet.collected_kf_num[0].clone()
        first_kf_pose, first_kf_Id = self.kfSet.extract_first_kf_pose(localMLP_Id, kf_poses)  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )
        first_kf_pose = first_kf_pose.cpu()
        given_pose_local = first_kf_pose.inverse() @ pose_world  # given keyframe's local pose in given localMLP's coordinate system, Tensor(4, 4)

        # Step 1: sampling pixels from given keyframe, and convert them to world coordinates
        # 1.1: sample pixels
        indice_h, indice_w = sample_pixels_uniformly(self.dataset.H, self.dataset.W, rays_h, rays_w)
        target_d = depth_img[indice_h, indice_w]  # Tensor(pixel_num, )
        rays_d_cam = rays_d[indice_h, indice_w]  # Tensor(pixel_num, 3)

        rays_o = pose_world[:3, -1].repeat(pixel_num, 1)  # apply translation(camera --> world), Tensor(pixel_num, 3)
        rays_d = torch.sum(rays_d_cam[..., None, :] * pose_world[None, :3, :3], -1)  # apply rotation(camera --> world): rotate direction of sampled rays, Tensor(pixel_num, 3)

        # 1.2: get sampled 3D points (in World Coordinate System)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * target_d[..., None, None]  # Tensor(pixel_num, 1, 3)
        pts = pts.reshape((-1, 3))  # Tensor(pixel_num, 3)


        # Step 2: among all related keyframes of given localMLP, find top K nearest keyframes based on center distance
        # 2.1: find all related keyframes of given localMLP
        # related_kf_mask = self.kfSet.get_related_keyframes(localMLP_Id, num_kf)  # Tensor(num_kf, ), 0/1
        related_kf_mask = self.kfSet.get_related_keyframes2(localMLP_Id, num_kf, active_localMLP_Id)  # Tensor(num_kf, ), 0/1

        related_kf_num = torch.count_nonzero(related_kf_mask)
        related_kf_Ids = torch.where(related_kf_mask > 0)[0]  # keyframe_Ids of all related keyframes, Tensor(n', )
        related_kf_frame_Ids = related_kf_Ids * self.config["mapping"]["keyframe_every"]  # frame_Ids of all related keyframes, Tensor(n', )
        related_kf_ref = keyframe_ref[related_kf_Ids]  # keyframe_ref type of all related keyframes, Tensor(n', ), n(>=0)/-1/-2
        related_pose_world = self.kfSet.convert_given_world_pose(related_kf_Ids, related_kf_ref, kf_poses, est_c2w_data[related_kf_frame_Ids].detach())  # Tensor(n', 4, 4)

        # 2.2: select top K nearest keyframes of given keyframe
        if related_kf_num <= self.kfSet.near_kf_num:
            topK_kf_Ids = related_kf_Ids  # Tensor(n', )
            topK_kf_pose_world = related_pose_world  # Tensor(n', 4, 4)
        else:
            center_given_kf = pts.mean(dim=0, dtype=torch.float32)  # center of given keyframe in World Coordinate System, Tensor(3, )
            related_kf_dist = self.kfSet.sort_center_dist_kf(center_given_kf, related_kf_Ids, related_pose_world)

            topK_kf_idx = torch.argsort(related_kf_dist, 0)[:self.kfSet.near_kf_num]
            topK_kf_Ids = related_kf_Ids[topK_kf_idx]  # Tensor(k, )
            topK_kf_pose_world = related_pose_world[topK_kf_idx]  # world poses of all selected keyframes, Tensor(k, 4, 4)
        topK_kf_pose_w2c = torch.inverse(topK_kf_pose_world)  # pose w2c, Tensor(k, 4, 4)

        # Step 3: compute containing mask of given keyframe w.r.t each selected keyframe
        # 3.1: for each selected keyframe, convert all sampled 3D points to its camera coordinate system
        topK_kf_rot_w2c = topK_kf_pose_w2c[:, :3, :3]  # rot mat w2c, Tensor(k, 3, 3)
        topK_kf_trans_w2c = topK_kf_pose_w2c[:, :3, 3]  # trans vec w2c, Tensor(k, 3)

        rotated_pts = torch.sum(pts[None, :, None, :] * topK_kf_rot_w2c[:, None, :, :], -1)  # Tensor(k, pixel_num, 3)
        transed_pts = rotated_pts + topK_kf_trans_w2c[:, None, :]  # Tensor(k, pixel_num, 3)
        transed_pts = torch.reshape(transed_pts, (-1, 3))  # Tensor(k * pixel_num, 3)

        # 3.2: cam coords --> pixel coords
        uv = project_to_pixel(self.K, transed_pts.unsqueeze(-1))  # Tensor(k * pixel_num, 2)
        edge = 20
        mask = (uv[:, 0] < self.config['cam']['W']-edge)*(uv[:, 0] > edge) * (uv[:, 1] < self.config['cam']['H']-edge)*(uv[:, 1] > edge)
        mask = mask & (transed_pts[..., -1] < 0)  # camera coordinates with z < 0 means lying in front of the camera
        top_kf_masks = mask.reshape(-1, pixel_num)  # whether each sampled surface is seen to each keyframe, Tensor(k, pixel_num)
        mask_pts = top_kf_masks.any(dim=0)  # Tensor(pixel_num, )

        # 3.3: judge whether each points in bbox
        center_len = self.kfSet.localMLP_info[localMLP_Id][1:]  # Tensor(6, )
        localMLP_center, localMLP_len = center_len[:3], center_len[3:]

        xyz_min = localMLP_center - 0.5 * localMLP_len  # Tensor(3, )
        xyz_max = localMLP_center + 0.5 * localMLP_len  # Tensor(3, )
        mask_in = pts_in_bbox(pts, xyz_min[None, ...], xyz_max[None, ...]).squeeze(-1)  # Tensor(pixel_num, )

        mask_final = torch.logical_and(mask_pts, mask_in)
        valid_pts_num = torch.count_nonzero(mask_final)
        learned_ratio = valid_pts_num / pixel_num

        if valid_pts_num >= self.config["mapping"]["overlapping"]["min_pts"]:
            # Step 4: judge whether this keyframe has enough correspondences with selected nearest keyframes, if not, loop will not be triggered
            pose_local_ini, pose_local_bf = self.slam.current_pose_switch_submap(frame_id, kf_id, active_localMLP_Id, localMLP_Id)
            rectify_flag, corre_num, pose_local_final = self.poseCorrector.switch_pose_rectifying(batch, pose_local_ini, pose_local_bf, localMLP_Id, active_localMLP_Id, topK_kf_Ids, top_kf_masks)
            if rectify_flag:
                self.slam.rectified_local_pose[0] = pose_local_final
                # TEST
                print("(TEST) Loop triggered, %d correspondences are found when rectifying local pose" % corre_num)
                # END TEST
            switch_prev = rectify_flag
        else:
            switch_prev = False

        return switch_prev, target_d, rays_d_cam, mask_final, topK_kf_Ids, top_kf_masks


    #################################################################### msg processing functions ####################################################################
    # @brief: process and insert a new keyframe
    # @param batch
    # @param active_localMLP_Id
    # @param pose_local: pose in Local Coordinate System of this keyframe, Tensor(4, 4);
    # @param keyframe_Id: Tensor(, );
    # -@return: 1: a new keyframe was added and bound with 2 localMLPs, active submap changed (to a previous localMLP);
    #          2: a new keyframe was added and bound with 1 or 2 localMLP(s), active submap didn't change;
    #          3: a new keyframe was added and a new localMLP was created, active submap changed. (to the new localMLP)
    @torch.no_grad()
    def process_keyframe(self, batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force=False):
        if self.wait_loop:
            return self.process_keyframe_wait_loop(batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force)
        else:
            return self.process_keyframe_normal(batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force)


    @torch.no_grad()
    def process_keyframe_normal(self, batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force=False):
        # Step 1: preparation
        # 1.1: get this keyframe's world_pose, and frustum_bbox
        pose_world = self.convert_pose_to_world(pose_local, active_localMLP_Id).cpu()
        frustum_xyz_center, frustum_xyz_len = get_frame_surface_bbox(pose_world, batch["depth"].squeeze(0), batch["direction"].squeeze(0), self.config["cam"]["near"], self.config["cam"]["far"])

        # 1.2: find most overlapping localMLP (in top 3 who has the highest containing ratio)
        # nearest_localMLPs = self.find_nearest_localMLP_topK(frustum_xyz_center, 3)
        nearest_localMLPs = self.find_nearest_localMLP_topK_exclude(active_localMLP_Id, frustum_xyz_center, 3)

        mo_localMLP_Id = self.find_highest_containing_ratio(batch["depth"].squeeze(0), batch["direction"].squeeze(0), pose_world, nearest_localMLPs)  # Tensor(, ) / Tensor(, )
        cr_mo = self.compute_containing_ratio(batch["depth"].squeeze(0), batch["direction"].squeeze(0), pose_world, mo_localMLP_Id)
        same_mlp_flag = (active_localMLP_Id == mo_localMLP_Id)


        # Step 2: judgement
        # case 1: before expanded, active localMLP already contains most part of this keyframe
        cr_active = self.compute_containing_ratio(batch["depth"].squeeze(0), batch["direction"].squeeze(0), pose_world, active_localMLP_Id)
        if force or cr_active >= self.cr_threshold:
            if same_mlp_flag == False and cr_mo >= self.cr_threshold_mo:  # case 1.1: this keyframe will be bound to 2 localMLPs, but active localMLP won't be switched
                switch_flag = self.process_double_binding(active_localMLP_Id, mo_localMLP_Id, cr_mo, batch, pose_world)
                flag = self.send_msg1(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, mo_localMLP_Id, pose_world, switch_flag)
                if switch_flag:
                    show_str = "double binding, active submap switch"
                else:
                    show_str = "double binding, unchanged"
            else:  # case 1.2: this keyframe will only be bound to active localMLP
                flag = self.send_msg2(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id)
                self.double_binding_counter = 0
                show_str = "unchanged"

            # TEST
            localMLP_center1, localMLP_len1 = self.kfSet.localMLP_info[active_localMLP_Id][1:4], self.kfSet.localMLP_info[active_localMLP_Id][4:7]
            print(printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (%s)" %
                (keyframe_Id, batch["frame_id"], cr_active, self.slam.active_localMLP_Id[0], localMLP_center1[0], localMLP_center1[1], localMLP_center1[2],
                 localMLP_len1[0], localMLP_len1[1], localMLP_len1[2], show_str) )
            # TEST
            return flag

        # case 2: localMLP length needs to be expanded
        localMLP_center1, localMLP_len1 = self.kfSet.localMLP_info[active_localMLP_Id][1:4], self.kfSet.localMLP_info[active_localMLP_Id][4:7]
        new_localMLP_center1, new_localMLP_len1 = self.localMLP_expand_rule(localMLP_center1, localMLP_len1, frustum_xyz_center, frustum_xyz_len, self.kfSet.localMLP_max_len[active_localMLP_Id])
        cr_active_new = self.compute_containing_ratio(batch["depth"].squeeze(0), batch["direction"].squeeze(0), pose_world,
                                                      active_localMLP_Id, localMLP_center=new_localMLP_center1, localMLP_len=new_localMLP_len1)
        if cr_active_new >= self.cr_threshold:
            if same_mlp_flag == False and cr_mo >= self.cr_threshold_mo:  # case 2.1: this keyframe will be bound to 2 localMLPs, but active localMLP won't be switched
                switch_flag = self.process_double_binding(active_localMLP_Id, mo_localMLP_Id, cr_mo, batch, pose_world)
                flag = self.send_msg1(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, mo_localMLP_Id, pose_world, switch_flag)
                if switch_flag:
                    show_str = "double binding, active submap switch"
                else:
                    show_str = "double binding, expanded"
            else:  # case 2.2: this keyframe will only be bound to active localMLP
                flag = self.send_msg2(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id)
                self.double_binding_counter = 0
                show_str = "expanded"

            # TEST
            print(printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (%s)" %
                  ( keyframe_Id, batch["frame_id"], cr_active_new, self.slam.active_localMLP_Id[0], new_localMLP_center1[0], new_localMLP_center1[1], new_localMLP_center1[2],
                    new_localMLP_len1[0], new_localMLP_len1[1], new_localMLP_len1[2], show_str) )
            # TEST
            return flag

        self.double_binding_counter = 0
        # case 3~5: this keyframe will be bound to a previous localMLP, or triggers creating a new localMLP
        if same_mlp_flag:  # case 3: active localMLP and MO localMLP are same, create a new localMLP
            flag, _ = self.send_msg3(keyframe_Id, frame_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, pose_world)
            if self.wait_loop:
                self.wait_loop = False

            # TEST
            print( printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (new localMLP)" %
                 (keyframe_Id, batch["frame_id"], cr_active_new, self.slam.active_localMLP_Id[0], new_localMLP_center1[0], new_localMLP_center1[1],
                 new_localMLP_center1[2], new_localMLP_len1[0], new_localMLP_len1[1], new_localMLP_len1[2]))
            # TEST
        else:
            if cr_mo < self.cr_threshold_back:  # case 4: active localMLP and MO localMLP are different, create a new localMLP
                flag, _ = self.send_msg3(keyframe_Id, frame_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, pose_world)
                if self.wait_loop:
                    self.wait_loop = False

                # TEST
                print(printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (new localMLP)" %
                     (keyframe_Id, batch["frame_id"], cr_active_new, self.slam.active_localMLP_Id[0], new_localMLP_center1[0], new_localMLP_center1[1],
                     new_localMLP_center1[2], new_localMLP_len1[0], new_localMLP_len1[1], new_localMLP_len1[2]) )
                # TEST
            else:  # case 5: camera moves to an existing localMLP's range
                # judge whether this keyframe has enough overlapping region with previous keyframes
                switch_flag, target_d, rays_d, pts_mask, top_kf_Ids, top_kf_mask = self.find_overlapping_region(batch, pose_world, active_localMLP_Id, mo_localMLP_Id,
                                                                          self.slam.kf_c2w.detach().cpu(), self.slam.est_c2w_data.detach().cpu(), self.slam.keyframe_ref.detach(),
                                                                          self.config["mapping"]["overlapping"]["n_rays_h"], self.config["mapping"]["overlapping"]["n_rays_w"])

                if switch_flag:  # case 5.1: active submap switch to the previous localMLP
                    flag = self.send_msg1(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, mo_localMLP_Id, pose_world, True)
                    self.kfSet.ovlp_depth[0] = target_d
                    self.kfSet.ovlp_rays[0] = rays_d
                    self.kfSet.ovlp_pts_mask[0] = pts_mask
                    self.kfSet.nearest_kf_Ids[0][:top_kf_Ids.shape[0]] = top_kf_Ids
                    self.kfSet.nearest_kf_mask[0][:top_kf_Ids.shape[0], ...] = top_kf_mask

                    if self.wait_loop:
                        self.wait_loop = False
                    show_str = "switch to prev"
                else:  # case 5.2: overlapping region is too small, create a new localMLP and wait for loop
                    flag, new_localMLP_Id = self.send_msg3(keyframe_Id, frame_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, pose_world)
                    self.wait_loop = True
                    self.localMLP_Id_wait = mo_localMLP_Id
                    self.localMLP_Id_actual = new_localMLP_Id

                    show_str = "wait loop, new localMLP"

                # TEST
                print(printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (%s)" %
                    (keyframe_Id, batch["frame_id"], cr_active_new, self.slam.active_localMLP_Id[0], new_localMLP_center1[0], new_localMLP_center1[1],
                     new_localMLP_center1[2], new_localMLP_len1[0], new_localMLP_len1[1], new_localMLP_len1[2], show_str) )
                # TEST
        return flag


    @torch.no_grad()
    def process_keyframe_wait_loop(self, batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force=False):
        # Step 1: get this keyframe's world_pose, and frustum_bbox
        pose_world = self.convert_pose_to_world(pose_local, active_localMLP_Id).cpu()
        frustum_xyz_center, frustum_xyz_len = get_frame_surface_bbox(pose_world, batch["depth"].squeeze(0), batch["direction"].squeeze(0), self.config["cam"]["near"], self.config["cam"]["far"])

        # Step 2: compute containing ratio of this keyframe and waited localMLP
        cr_wt = self.compute_containing_ratio(batch["depth"].squeeze(0), batch["direction"].squeeze(0), pose_world, self.localMLP_Id_wait)
        if force or cr_wt < self.cr_threshold_back:
            return self.process_keyframe_normal(batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force)

        # Step 3: judge whether there is enough overlapping region (if not, do normal keyframe judgement)
        switch_to_prev = self.get_loop_flag(self.localMLP_Id_wait, active_localMLP_Id, cr_wt, batch, pose_world)
        if switch_to_prev == False:
            return self.process_keyframe_normal(batch, active_localMLP_Id, pose_local, frame_Id, keyframe_Id, force)
        else:
            flag = self.send_msg1(keyframe_Id, frustum_xyz_center, frustum_xyz_len, active_localMLP_Id, self.localMLP_Id_wait, pose_world, switch_to_prev)
            show_str = "switch to prev"

            # TEST
            localMLP_center1, localMLP_len1 = self.kfSet.localMLP_info[self.localMLP_Id_wait][1:4], self.kfSet.localMLP_info[self.localMLP_Id_wait][4:7]
            print( printCurrentDatetime() + "!!!!!!! Keyframe_%d, frame_Id=%d, containing ratio=%.4f; active localMLP Id=%d, center=(%.2f, %.2f, %.2f), length=(%.2f, %.2f, %.2f) -- (%s)" %
                ( keyframe_Id, batch["frame_id"], cr_wt, self.slam.active_localMLP_Id[0], localMLP_center1[0], localMLP_center1[1], localMLP_center1[2],
                  localMLP_len1[0], localMLP_len1[1], localMLP_len1[2], show_str))
            # TEST
            return flag


    # @brief: message Type 1 ---- binding a keyframe to 2 localMLPs (overlapping keyframe);
    def send_msg1(self, kf_Id, kf_surface_center, kf_surface_len, localMLP_Id1, localMLP_Id2, pose_world, active_switch=False):
        if active_switch:
            self.kfSet.localMLP_max_len[localMLP_Id2] = torch.tensor(self.config["mapping"]["localMLP_max_len_back"])

        # Step 1: compute expanded center and length of localMLPs
        # 1.1: for localMLP_1 (active localMLP)
        localMLP_center1, localMLP_len1 = self.kfSet.localMLP_info[localMLP_Id1][1:4], self.kfSet.localMLP_info[localMLP_Id1][4:7]
        new_localMLP_center1, new_localMLP_len1 = self.localMLP_expand_rule(localMLP_center1, localMLP_len1, kf_surface_center, kf_surface_len, self.kfSet.localMLP_max_len[localMLP_Id1])

        # 1.2: for localMLP_2 (most overlapping localMLP)
        localMLP_center2, localMLP_len2 = self.kfSet.localMLP_info[localMLP_Id2][1:4], self.kfSet.localMLP_info[localMLP_Id2][4:7]
        if active_switch == False:
            new_localMLP_center2, new_localMLP_len2 = localMLP_center2, localMLP_len2  # MO localMLP will not expand
        else:
            new_localMLP_center2, new_localMLP_len2 = self.localMLP_expand_rule(localMLP_center2, localMLP_len2, kf_surface_center, kf_surface_len, self.kfSet.localMLP_max_len[localMLP_Id2])

        # Step 2: modify keyframe-related and localMLP-related vars
        if active_switch == False:
            self.kfSet.add_keyframe_localMLP(kf_Id, localMLP_Id1=localMLP_Id1, localMLP_Id2=localMLP_Id2)
        else:
            self.kfSet.add_keyframe_localMLP(kf_Id, localMLP_Id1=localMLP_Id2, localMLP_Id2=localMLP_Id1)

        self.kfSet.modify_localMLP_info(localMLP_Id1, new_localMLP_center1, new_localMLP_len1)
        self.kfSet.modify_localMLP_info(localMLP_Id2, new_localMLP_center2, new_localMLP_len2)
        self.kfSet.add_adjcent_pair(localMLP_Id1, localMLP_Id2)

        # Step 3: since this keyframe must be overlapping keyframe, modify overlapping keyframe-related vars
        self.slam.keyframe_ref[kf_Id] = -2
        # self.slam.kf_c2w[kf_Id] = pose_world

        if active_switch:
            # # self.slam.est_c2w_data[kf_Id] = self.convert_pose_to_local(pose_world, localMLP_Id2)
            # self.slam.keyframe_ref[kf_Id] = -3
            self.slam.prev_active_localMLP_Id[0] = self.slam.active_localMLP_Id[0]
            self.slam.active_localMLP_Id[0] = localMLP_Id2
            self.slam.overlap_kf_flag[kf_Id] = -1  # active submap changed, next time overlapping kf should be optimized in ActiveMap process
            self.kfSet.update_mutex_mask(localMLP_Id2, self.slam.keyframe_ref, self.kfSet.collected_kf_num[0]+1)
            return 1
        else:
            # self.slam.keyframe_ref[kf_Id] = -2
            self.slam.overlap_kf_flag[kf_Id] = -1  # active submap unchanged, next time overlapping kf should be optimized in ActiveMap process
            return 2


    # @brief: message Type 2 ---- binding a keyframe to 1 localMLP;
    # @param kf_Id: Tensor(, );
    # @param kf_surface_center: Tensor(3, );
    # @param kf_surface_len: Tensor(3, );
    # @param localMLP_Id: Tensor(, )
    def send_msg2(self, kf_Id, kf_surface_center, kf_surface_len, localMLP_Id):
        # Step 1: compute expanded center and length of this localMLP
        localMLP_center, localMLP_len = self.kfSet.localMLP_info[localMLP_Id][1:4], self.kfSet.localMLP_info[localMLP_Id][4:7]
        new_localMLP_center, new_localMLP_len = self.localMLP_expand_rule(localMLP_center, localMLP_len, kf_surface_center, kf_surface_len, self.kfSet.localMLP_max_len[localMLP_Id])

        # Step 2: modify keyframe-related and localMLP-related vars
        self.kfSet.add_keyframe_localMLP(kf_Id, localMLP_Id1=localMLP_Id)
        self.kfSet.modify_localMLP_info(localMLP_Id, new_localMLP_center, new_localMLP_len)
        return 2


    # @brief: message Type 3 ---- create a new localMLP and bind the keyframe to it;
    # @param kf_Id: Tensor(, );
    # @param kf_surface_center: Tensor(3, );
    # @param kf_surface_len: Tensor(3, );
    # @param localMLP_Id: Tensor(, );
    # @param pose_world: pose in World Coordinate System of this keyframe, Tensor(4, 4)
    def send_msg3(self, kf_Id, frame_Id, kf_surface_center, kf_surface_len, active_localMLP_Id, pose_world):
        # Step 1: get initial center and axis-aligned length of the new localMLP
        localMLP_ini_center, localMLP_ini_len = self.localMLP_create_rule(kf_surface_center, kf_surface_len)

        # Step 2: modify keyframe-related and localMLP-related vars
        new_localMLP_Id = self.kfSet.modify_new_localMLP_info(localMLP_ini_center, localMLP_ini_len, kf_Id)
        self.kfSet.add_keyframe_localMLP(kf_Id, new_localMLP_Id, active_localMLP_Id)
        self.kfSet.add_adjcent_pair(active_localMLP_Id, new_localMLP_Id)

        # Step 3: modify active localMLP-related vars
        self.slam.prev_active_localMLP_Id[0] = self.slam.active_localMLP_Id[0]
        self.slam.active_localMLP_Id[0] = new_localMLP_Id

        # Step 4: since this keyframe must be new localMLP's first keyframe, modify first keyframe-related vars
        self.slam.keyframe_ref[kf_Id] = -1
        self.slam.kf_c2w[kf_Id] = pose_world
        self.slam.est_c2w_data[frame_Id] = torch.eye(4).to(self.device)
        self.kfSet.update_mutex_mask(new_localMLP_Id, self.slam.keyframe_ref, self.kfSet.collected_kf_num[0])
        return 3, new_localMLP_Id
    ######################################## END msg processing functions ########################################


    ######################################## helper functions ########################################
    # @brief:
    #-@return xyz_center_new: center of expanded localMLP, Tensor(3, );
    #-@return xyz_len_new: axis-aligned of expanded localMLP, Tensor(3, ).
    def localMLP_expand_rule(self, localMLP_center, localMLP_len, kf_surface_center, kf_surface_len, localMLP_max_len=None):
        if localMLP_max_len is None:
            localMLP_max_len = self.localMLP_max_len

        kf_xyz_min, kf_xyz_max = kf_surface_center - 0.5 * kf_surface_len, kf_surface_center + 0.5 * kf_surface_len  # Tensor(3, ) / Tensor(3, )
        mlp_xyz_min, mlp_xyz_max = localMLP_center - 0.5 * localMLP_len, localMLP_center + 0.5 * localMLP_len  # Tensor(3, ) / Tensor(3, )
        contain_cond_min = (kf_xyz_min >= mlp_xyz_min)
        contain_cond_max = (kf_xyz_max <= mlp_xyz_max)
        contain_cond = torch.cat([contain_cond_min, contain_cond_max], dim=0)  # Tensor(6, )

        if contain_cond.all():  # localMLP does not need to be expanded in any direction
            return localMLP_center, localMLP_len
        else:
            expand_dir = ~contain_cond  # along which directions the localMLP should be expanded (x_neg, y_neg, z_neg, x_pos, y_pos, z_pos)
            xyz_min, _ = torch.min(torch.stack([kf_xyz_min, mlp_xyz_min], -1), -1)  # Tensor(3, )
            xyz_max, _ = torch.max(torch.stack([kf_xyz_max, mlp_xyz_max], -1), -1)  # Tensor(3, )

            # for X axis
            if localMLP_len[0] >= localMLP_max_len[0]:  # length along X axis cannot be expanded anymore
                x_center_new, x_len_new = localMLP_center[0], localMLP_len[0]
            else:
                if xyz_max[0] - xyz_min[0] <= localMLP_max_len[0]:  # case 1: expanded length will not be clamped
                    x_len_new = xyz_max[0] - xyz_min[0]
                    x_center_new = xyz_min[0] + 0.5 * x_len_new
                else:  # expanded length will be clamped
                    if (expand_dir[0] and expand_dir[3]) == False:  # case 2: either x_pos or x_neg can be expanded
                        x_pos_expand, x_neg_expand = torch.abs(xyz_max[0] - mlp_xyz_max[0]), torch.abs(mlp_xyz_min[0] - xyz_min[0])
                        if x_pos_expand > 0:  # case 2.1: x_pos should be expanded
                            x_pos_expand = localMLP_max_len[0] - localMLP_len[0]
                            x_neg_expand = 0
                            x_center_new = localMLP_center[0] + 0.5 * x_pos_expand
                        else:  # case 2.2: x_neg should be expanded
                            x_pos_expand = 0
                            x_neg_expand = localMLP_max_len[0] - localMLP_len[0]
                            x_center_new = localMLP_center[0] - 0.5 * x_neg_expand
                        x_len_new = localMLP_max_len[0]
                    else:  # case 3: both x_pos and x_neg should be expanded
                        x_pos_expand, x_neg_expand = torch.abs(xyz_max[0] - mlp_xyz_max[0]), torch.abs(mlp_xyz_min[0] - xyz_min[0])
                        x_expand_len = localMLP_max_len[0] - localMLP_len[0]  # max expanded length along X axis
                        x_pos_expand_clamp = x_expand_len * x_pos_expand / (x_pos_expand + x_neg_expand)
                        x_neg_expand_clamp = x_expand_len * x_neg_expand / (x_pos_expand + x_neg_expand)
                        x_max_new, x_min_new = mlp_xyz_max[0] + x_pos_expand_clamp, mlp_xyz_min[0] - x_neg_expand_clamp
                        x_len_new = x_max_new - x_min_new
                        x_center_new = x_min_new + 0.5 * x_len_new

            # for Y axis
            if localMLP_len[1] >= localMLP_max_len[1]:  # length along Y axis cannot be expanded anymore
                y_center_new, y_len_new = localMLP_center[1], localMLP_len[1]
            else:
                if xyz_max[1] - xyz_min[1] <= localMLP_max_len[1]:  # case 1: expanded length will not be clamped
                    y_len_new = xyz_max[1] - xyz_min[1]
                    y_center_new = xyz_min[1] + 0.5 * y_len_new
                else:  # expanded length will be clamped
                    if (expand_dir[1] and expand_dir[4]) == False:  # case 2: either Y_pos or Y_neg can be expanded
                        y_pos_expand, y_neg_expand = torch.abs(xyz_max[1] - mlp_xyz_max[1]), torch.abs(mlp_xyz_min[1] - xyz_min[1])
                        if y_pos_expand > 0:  # case 2.1: Y_pos should be expanded
                            y_pos_expand = localMLP_max_len[1] - localMLP_len[1]
                            y_neg_expand = 0
                            y_center_new = localMLP_center[1] + 0.5 * y_pos_expand
                        else:  # case 2.2: y_neg should be expanded
                            y_pos_expand = 0
                            y_neg_expand = localMLP_max_len[1] - localMLP_len[1]
                            y_center_new = localMLP_center[1] - 0.5 * y_neg_expand
                        y_len_new = localMLP_max_len[1]
                    else:  # case 3: both Y_pos and Y_neg should be expanded
                        y_pos_expand, y_neg_expand = torch.abs(xyz_max[1] - mlp_xyz_max[1]), torch.abs(mlp_xyz_min[1] - xyz_min[1])
                        y_expand_len = localMLP_max_len[1] - localMLP_len[1]  # max expanded length along Y axis
                        y_pos_expand_clamp = y_expand_len * y_pos_expand / (y_pos_expand + y_neg_expand)
                        y_neg_expand_clamp = y_expand_len * y_neg_expand / (y_pos_expand + y_neg_expand)
                        y_max_new, y_min_new = mlp_xyz_max[1] + y_pos_expand_clamp, mlp_xyz_min[1] - y_neg_expand_clamp
                        y_len_new = y_max_new - y_min_new
                        y_center_new = y_min_new + 0.5 * y_len_new

            # for Z axis
            if localMLP_len[2] >= localMLP_max_len[2]:  # length along Z axis cannot be expanded anymore
                z_center_new, z_len_new = localMLP_center[2], localMLP_len[2]
            else:
                if xyz_max[2] - xyz_min[2] <= localMLP_max_len[2]:  # case 1: expanded length will not be clamped
                    z_len_new = xyz_max[2] - xyz_min[2]
                    z_center_new = xyz_min[2] + 0.5 * z_len_new
                else:  # expanded length will be clamped
                    if (expand_dir[2] and expand_dir[5]) == False:  # case 2: either Z_pos or Z_neg can be expanded
                        z_pos_expand, z_neg_expand = torch.abs(xyz_max[2] - mlp_xyz_max[2]), torch.abs(mlp_xyz_min[2] - xyz_min[2])
                        if z_pos_expand > 0:  # case 2.1: Z_pos should be expanded
                            z_pos_expand = localMLP_max_len[2] - localMLP_len[2]
                            z_neg_expand = 0
                            z_center_new = localMLP_center[2] + 0.5 * z_pos_expand
                        else:  # case 2.2: Z_neg should be expanded
                            z_pos_expand = 0
                            z_neg_expand = localMLP_max_len[2] - localMLP_len[2]
                            z_center_new = localMLP_center[2] - 0.5 * z_neg_expand
                        z_len_new = localMLP_max_len[2]
                    else:  # case 3: both Z_pos and Z_neg should be expanded
                        z_pos_expand, z_neg_expand = torch.abs(xyz_max[2] - mlp_xyz_max[2]), torch.abs(mlp_xyz_min[2] - xyz_min[2])
                        z_expand_len = localMLP_max_len[2] - localMLP_len[2]  # max expanded length along Z axis
                        z_pos_expand_clamp = z_expand_len * z_pos_expand / (z_pos_expand + z_neg_expand)
                        z_neg_expand_clamp = z_expand_len * z_neg_expand / (z_pos_expand + z_neg_expand)
                        z_max_new, z_min_new = mlp_xyz_max[2] + z_pos_expand_clamp, mlp_xyz_min[2] - z_neg_expand_clamp
                        z_len_new = z_max_new - z_min_new
                        z_center_new = z_min_new + 0.5 * z_len_new

            xyz_center_new = torch.stack([x_center_new, y_center_new, z_center_new], 0)
            xyz_len_new = torch.stack([x_len_new, y_len_new, z_len_new], 0)
            return xyz_center_new, xyz_len_new


    # @brief: determine the center and axis-aligned length of a newly created localMLP;
    # @param kf_center: Tensor(3, );
    # @param kf_len: Tensor(3, );
    #-@return localMLP_ini_center: Tensor(3, );
    #-@return localMLP_ini_len: Tensor(3, ).
    def localMLP_create_rule(self, kf_center, kf_len):
        localMLP_ini_center = kf_center
        localMLP_ini_len = kf_len
        return localMLP_ini_center, localMLP_ini_len
    ######################################## END helper functions ########################################