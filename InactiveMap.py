import os
import copy
import time

import numpy as np
import torch

from model.scene_rep import JointEncoding
from helper_functions.printTime import printCurrentDatetime
from helper_functions.geometry_helper import extract_first_kf_pose, find_related_localMLPs, compute_avg_SDF_difference, compute_avg_RGB_difference
from Logger import Logger


class InactiveMap():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = self.slam.device
        self.dataset = self.slam.dataset
        self.kfSet = self.slam.kfSet
        self.poseCorrector = self.slam.poseCorrector
        self.logger = Logger(self.config, SLAM)
        self.shared_model = self.slam.shared_model

        self.shared_flag = self.slam.shared_flag
        self.mesh_flag = self.slam.mesh_flag
        self.logger = self.slam.logger
        self.model_list = []
        self.active_localMLP_Id = torch.zeros((1, ), dtype=torch.int64)
        self.process_flag = -1  # process flag of InactiveMap process
        self.last_saved_ckpt = torch.zeros((1, ), dtype=torch.int64)
        self.sleep_time = self.config["mapping"]["inactive"]["sleep_time"]

        self.last_opt_localMLP_Id = torch.zeros((1, ), dtype=torch.int64)  # selected localMLP_Id of last optimization
        self.inactive_pause = torch.zeros((1, ), dtype=torch.int64)

        # global BA related vars
        self.active_model_copy = JointEncoding(config, self.slam.bounding_box, self.slam.coords_norm_factor).to(self.device).share_memory()  # copy of latest active localMLP
        self.active_model_copy_Id = ( -1 * torch.ones((1, )) ).share_memory_()  # localMLP Id

        self.do_globalBA = self.slam.do_globalBA
        self.counter = 0
        self.trunc_value = self.config["training"]["trunc"]
        self.glob_BA_round = torch.zeros((1, ), dtype=torch.int32).share_memory_()

        # loop closure related vars
        self.ovlp_depth = self.kfSet.ovlp_depth  # Tensor(1, n_rays_h * n_rays_w, 3)
        self.ovlp_rays = self.kfSet.ovlp_rays  # Tensor(1, n_rays_h * n_rays_w)
        self.ovlp_pts_mask = self.kfSet.ovlp_pts_mask  # Tensor(1, n_rays_h * n_rays_w)


    # @brief: Create optimizer for mapping (embedding + MLP)
    def create_optimizer(self, model):
        trainable_parameters = [{'params': model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
        map_optimizer = torch.optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        return map_optimizer


    # @brief: process request (from ActiveMap process) for active submap switch
    def process_active_submap_switch(self):
        localMLP_Id = self.slam.localMLP_Id_a2i[0]  # localMLP_Id that ActiveMap process sent

        if self.shared_flag[0] == 1:
            # case 1: switch to a new localMLP
            if localMLP_Id >= len(self.model_list):
                recvd_model = copy.deepcopy(self.shared_model).to(self.device)
                self.model_list.append(recvd_model)
            else:
                self.model_list[localMLP_Id].load_state_dict(self.shared_model.state_dict())
            self.shared_flag[0] = 0
            self.active_localMLP_Id[0] = self.slam.active_localMLP_Id[0]
        else:
            # case 2: switch to a previous localMLP
            asked_localMLP_Id = self.slam.localMLP_Id_asked[0]  # localMLP_Id that ActiveMap process asked
            if asked_localMLP_Id < 0:
                print("(InactiveMap process) !!!!!! ERROR: Invalid asked localMLP_Id !!!!!!")
                raise RuntimeError

            if localMLP_Id >= len(self.model_list):
                recvd_model = copy.deepcopy(self.shared_model).to(self.device)
                self.model_list.append(recvd_model)
            else:
                self.model_list[localMLP_Id].load_state_dict(self.shared_model.state_dict())

            # self.global_BA_overlapping(self.slam.last_ovlp_kf_Id[0].clone(), self.slam.last_switch_frame[0].clone())

            self.shared_model.load_state_dict(self.model_list[asked_localMLP_Id].state_dict())
            self.shared_flag[0] = -1
            self.active_localMLP_Id[0] = self.slam.active_localMLP_Id[0].clone()

            while True:
                if self.inactive_pause[0] == 1:
                    pass
                else:
                    break


    # @brief: get latest weights of active localMLP
    def merge_active_model_copy(self):
        if self.active_model_copy_Id[0] >= 0:
            localMLP_Id = self.active_model_copy_Id[0].clone()
            if localMLP_Id != self.active_localMLP_Id:
                return

            if localMLP_Id >= len(self.model_list):
                new_model = copy.deepcopy(self.active_model_copy).to(self.device)
                self.model_list.append(new_model)
            else:
                self.model_list[int(localMLP_Id)].load_state_dict(self.active_model_copy.state_dict())


    # @brief: contsruct optimizable pose
    # @param poses: Tensor(n, 4, 4)
    def get_pose_param_optim(self, poses):
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])  # Parameter(n, 3)
        cur_rot = torch.nn.parameter.Parameter(self.slam.matrix_to_tensor(poses[:, :3, :3]))  # rot mat --> quat, Parameter(n, 4)
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config["mapping"]["inactive"]['lr_rot']},
                                           {"params": cur_trans, "lr": self.config["mapping"]["inactive"]['lr_trans']}])
        return cur_rot, cur_trans, pose_optimizer


    # @brief: feed sampled points to model;
    # @param local_poses: poses that convert camera coordinates to local coordinates for each sampled pts, Tensor(N, 4, 4);
    # @param model
    # @param rays_d_cam: dir of sampled rays, Tensor(N, 3)
    # @param target_d: depth of sampled rays, Tensor(N, 1)
    def infer_pts(self, local_poses, model, rays_d_cam, target_d):
        # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3) (Camera coords -> Local coords)
        rays_d = torch.sum(rays_d_cam[..., None, None, :] * local_poses[..., None, :3, :3], -1)
        rays_o = local_poses[..., None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)  # Tensor(N, 3)
        rays_d = rays_d.reshape(-1, 3)  # Tensor(N, 3)
        pts_local = (rays_o[..., None, :] + rays_d[..., None, :] * target_d[..., :, None]).reshape(-1, 3)  # Tensor(N, 3)

        rgb_sdf = model.run_network(pts_local)  # Tensor(n, 10)
        pred_rgb = rgb_sdf[..., :3]  # Tensor(n, 3)
        pred_sdf = rgb_sdf[..., 3:4] * self.trunc_value  # Tensor(n, 1)
        return pred_rgb, pred_sdf


    # @brief:
    # @param rays: rays info of sampled pixels, Tensor(N, 7)
    # @param kf_indices: corresponding keyframe index in ovlp_kf_pose of each sampled ray, Tensor(N, );
    # @param ovlp_kf_pose: world pose of each sampled ray's corresponding keyframe, Tensor(N, 4, 4);
    # @param localMLP_Id1
    # @param localMLP_Id2
    # @param first_kf_pose1
    # @param first_kf_pose2
    def get_SDF_dif(self, rays, ovlp_kf_pose, localMLP_Id1, localMLP_Id2, first_kf_pose1, first_kf_pose2):
        rays_d_cam = rays[..., :3].to(self.device)  # direction (in Camera Coords) of sampled pixels(rays), Tensor(N, 3), device=cuda
        target_s = rays[..., 3:6].to(self.device)  # gt RGB of sampled pixels(rays), Tensor(N, 3), device=cuda:0
        target_d = rays[..., 6:7].to(self.device)  # gt depth of sampled pixels(rays), Tensor(N, 1), device=cuda:0
        depth_mask = torch.where(target_d > 0., torch.ones_like(target_d), torch.zeros_like(target_d))  # Tensor(N, 1)

        # Step 2: get local poses (in 2 Coordinate Systems respectively) of each selected surface points
        local_poses1 = first_kf_pose1.inverse() @ ovlp_kf_pose  # Tensor(N, 4, 4)
        local_poses2 = first_kf_pose2.inverse() @ ovlp_kf_pose  # Tensor(N, 4, 4)

        pred_rgb1, pred_sdf1 = self.infer_pts(local_poses1, self.model_list[int(localMLP_Id1)], rays_d_cam, target_d)
        pred_rgb2, pred_sdf2 = self.infer_pts(local_poses2, self.model_list[int(localMLP_Id2)], rays_d_cam, target_d)

        loss_SDF_avg = compute_avg_SDF_difference(pred_sdf1, pred_sdf2, depth_mask)
        loss_rgb_avg = compute_avg_RGB_difference(pred_rgb1, pred_rgb2, depth_mask)
        loss_avg = loss_SDF_avg + 0. * loss_rgb_avg
        return loss_avg


    # @brief: compute SDF difference using giving points;
    # @param target_d: depth of sampled rays, Tensor(N, 1);
    # @param rays_d_cam: dir of sampled rays (in Camera Coordinate System), Tensor(N, 3);
    # @param mask:  Tensor(N, 1)
    # @param ovlp_kf_pose: world pose of overlapping keyframes, Tensor(1, 4, 4);
    # @param localMLP_Id1
    # @param localMLP_Id2
    # @param first_kf_pose1
    # @param first_kf_pose2
    def get_SDF_dif2(self, target_d, rays_d_cam, mask, ovlp_kf_pose, localMLP_Id1, localMLP_Id2, first_kf_pose1, first_kf_pose2):
        rays_d_cam = rays_d_cam.to(self.device)  # direction (in Camera Coords) of sampled pixels(rays), Tensor(N, 3), device=cuda
        target_d = target_d.to(self.device)  # gt depth of sampled pixels(rays), Tensor(N, 1), device=cuda:0
        mask = mask.to(target_d)

        # Step 2: get local poses (in 2 Coordinate Systems respectively) of each selected surface points
        local_poses1 = first_kf_pose1.inverse() @ ovlp_kf_pose  # Tensor(N, 4, 4)
        local_poses2 = first_kf_pose2.inverse() @ ovlp_kf_pose  # Tensor(N, 4, 4)

        pred_rgb1, pred_sdf1 = self.infer_pts(local_poses1, self.model_list[int(localMLP_Id1)], rays_d_cam, target_d)
        pred_rgb2, pred_sdf2 = self.infer_pts(local_poses2, self.model_list[int(localMLP_Id2)], rays_d_cam, target_d)

        loss_SDF_avg = compute_avg_SDF_difference(pred_sdf1, pred_sdf2, mask)
        loss_rgb_avg = compute_avg_RGB_difference(pred_rgb1, pred_rgb2, mask)
        loss_avg = loss_SDF_avg + 0. * loss_rgb_avg
        return loss_avg


    def extract_mesh(self, frame_id):
        self.merge_active_model_copy()
        first_kf_Ids = torch.where(self.slam.keyframe_ref == -1)[0]  # keyframe_Id of each localMLP's first keyframe
        first_kf_pose = self.slam.kf_c2w[first_kf_Ids].detach()
        self.logger.extract_all_mesh(frame_id, self.model_list, first_kf_pose)


    # @brief: select an inactive localMLP (alternately) and do n_iters local BA for it
    def local_BA(self):
        localMLP_num = len(self.model_list)  # active + inactive

        # preparation: select a inactive localMLP to do local BA
        available_localMLP_Id = (self.last_opt_localMLP_Id[0] + 1) % localMLP_num
        if available_localMLP_Id == self.active_localMLP_Id[0]:
            available_localMLP_Id = (self.last_opt_localMLP_Id[0] + 1) % localMLP_num
        selected_model = self.model_list[ int(available_localMLP_Id) ]

        pose_optimizer = None
        map_optimizer = self.create_optimizer(selected_model)

        # Step 1: select all related keyframes' local poses / kf_Ids / frame_Ids
        first_kf_pose, first_kf_Id, poses, kf_ids_all, frame_ids_all, related_kf_ref, related_ov_kf_idx, related_ov_kf_Ids \
            = self.kfSet.extract_localMLP_vars(available_localMLP_Id, self.slam.kf_c2w, self.slam.est_c2w_data, self.slam.keyframe_ref, self.process_flag)

        # Step 2: construct optimizer for fixed kf_poses and optimizable kf_poses (current_pose will bot be optimized)
        if len(kf_ids_all) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            poses_all = poses_fixed
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)  # pose of this localMLP's first kf is always fixed
            cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
            pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

        if map_optimizer is not None:
            map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        # Step 3: Step 3: perform n_iters BA
        for i in range(self.config['mapping']['iters']):
            # 3.1: sampling pixels from stored keyframes
            rays, kf_ids, kf_indices = self.kfSet.sample_rays_in_submap(first_kf_Id, kf_ids_all, self.config['mapping']['sample'])

            # 3.2: extract info of selected rays
            rays_d_cam = rays[..., :3].to(self.device)  # direction (in Camera Coords) of sampled pixels(rays), Tensor(N, 3), device=cuda
            target_s = rays[..., 3:6].to(self.device)  # gt RGB of sampled pixels(rays), Tensor(N, 3), device=cuda:0
            target_d = rays[..., 6:7].to(self.device)  # gt depth of sampled pixels(rays), Tensor(N, 1), device=cuda:0

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3) (Camera coords -> World coords)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[kf_indices, None, :3, :3], -1)
            rays_o = poses_all[kf_indices, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 3.3: inference
            ret = selected_model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret)
            loss.backward(retain_graph=True)

            # 3.5: update model
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    map_optimizer.step()
                else:
                    print('Wait update')
                map_optimizer.zero_grad()

            # 3.6: update poses
            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # get SE3 poses to do forward pass
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)  # SE3 poses
                pose_optimizer.zero_grad()
        # END for

        # # Step 4: update related keyframe's poses
        # if pose_optimizer is not None and len(kf_ids_all) > 1:
        #     for i in range( len( kf_ids_all[1:] ) ):
        #         pose_local = self.slam.matrix_from_tensor(cur_rot[i: i + 1], cur_trans[i: i + 1]).detach().clone()[0]  # optimized local pose, Tensor(4, 4)
        #         if related_kf_ref[1:][i] >= 0:  # for ordinary keyframes
        #             frame_id = frame_ids_all[1:][i]  # frame_Id of this keyframe
        #             self.slam.est_c2w_data[frame_id] = pose_local
        #         else:  # for overlapping keyframes
        #             kf_id = kf_ids_all[1:][i]
        #             pose_world = first_kf_pose @ pose_local  # Tensor(4, 4)
        #             self.slam.kf_c2w[kf_id] = pose_world

        # Step 4: update related keyframe's poses
        if pose_optimizer is not None and len(kf_ids_all) > 1:
            for i in range(len(kf_ids_all[1:])):
                pose_local = self.slam.matrix_from_tensor(cur_rot[i: i + 1], cur_trans[i: i + 1]).detach().clone()[0]  # optimized local pose, Tensor(4, 4)
                if related_kf_ref[1:][i] >= 0:  # for ordinary keyframes
                    frame_id = frame_ids_all[1:][i]  # frame_Id of this keyframe
                    self.slam.est_c2w_data[frame_id] = pose_local
                elif related_kf_ref[1:][i] == -1:  # for first keyframes of other localMLPs
                    kf_id = kf_ids_all[1:][i]
                    pose_world = first_kf_pose @ pose_local  # Tensor(4, 4)
                    self.slam.kf_c2w[kf_id] = pose_world
                else:  # for overlapping keyframes
                    frame_id = frame_ids_all[1:][i]
                    kf_id = kf_ids_all[1:][i]
                    if available_localMLP_Id == self.kfSet.keyframe_localMLP[kf_id, 0]:  # if selected localMLP is its first related localMLP
                        self.slam.est_c2w_data[frame_id] = pose_local
                    else:  # if selected localMLP is its second related localMLP
                        pose_world = first_kf_pose @ pose_local
                        first_kf_pose_another = extract_first_kf_pose(self.kfSet.keyframe_localMLP[kf_id, 0], self.kfSet.localMLP_first_kf, self.slam.kf_c2w).detach()
                        pose_local_another = first_kf_pose_another.inverse() @ pose_world
                        self.slam.est_c2w_data[frame_id] = pose_local_another

        if related_ov_kf_Ids.shape[0] > 0:
            self.slam.overlap_kf_flag[related_ov_kf_Ids] = self.process_flag

        self.last_opt_localMLP_Id[0] = available_localMLP_Id
    # END local_BA()


    # # @brief: do global BA
    # def global_BA(self):
    #     self.merge_active_model_copy()
    #
    #     # Step 1: find all adjacent localMLPs pairs
    #     adja_pairs, part_localMLPs = self.kfSet.find_adjacent_localMLP_pair()  # Tensor(k, 2) / Tensor(m, )
    #     if part_localMLPs.shape[0] >= 2 and part_localMLPs.shape[0] == len(self.model_list):
    #         pass
    #     else:
    #         return
    #
    #     # Step 2: find keyframe_Id and keyframe pose of all first keyframes and overlapping keyframes
    #     first_ovlp_kf_Ids = torch.where( torch.logical_or(self.slam.keyframe_ref==-1, self.slam.keyframe_ref==-2) )[0]
    #     # first_ovlp_kf_Ids = torch.cat([torch.zeros((1, )), first_ovlp_kf_Ids], 0)  # kf_Ids of all first keyframes and overlapping keyframes
    #     first_ovlp_kf_pose = self.slam.kf_c2w[first_ovlp_kf_Ids]
    #
    #     # 2.2
    #     poses_fixed = torch.nn.parameter.Parameter(first_ovlp_kf_pose[:1]).to(self.device)  # pose of keyframe_0 is always fixed
    #     cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(first_ovlp_kf_pose[1:])
    #     pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
    #     poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
    #
    #     if pose_optimizer is not None:
    #         pose_optimizer.zero_grad()
    #
    #     # Step 3: do global BA
    #     bs = max(self.config["mapping"]["sample"] // adja_pairs.shape[0], self.config["mapping"]["sample"] // 5)
    #     for i in range(self.config["mapping"]["global_BA"]["n_iter"]):
    #         loss = 0.
    #         for adja_pair in adja_pairs:
    #             localMLP_Id1 = adja_pair[0]
    #             localMLP_Id2 = adja_pair[1]
    #             # 3.1: find overlapping region of these 2 localMLPs
    #             related_kf_indices = find_related_localMLPs(self.kfSet.keyframe_localMLP[first_ovlp_kf_Ids], localMLP_Id1, localMLP_Id2)  # indice of related_kf in first_ovlp_kf_Ids
    #             related_kf_Ids = first_ovlp_kf_Ids[related_kf_indices]
    #
    #             # 3.2: sample pixels from related keyframes
    #             rays, kf_indices = self.kfSet.sample_rays_from_given(related_kf_Ids, bs)
    #
    #             # 3.3: get loss term of this adjacent pair
    #             loss_avg = self.get_SDF_dif(rays, kf_indices, first_ovlp_kf_Ids, poses_all, related_kf_indices, localMLP_Id1, localMLP_Id2)
    #             loss += 5. * loss_avg
    #
    #         loss.backward(retain_graph=True)
    #         # 3.4: update poses
    #         if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
    #             pose_optimizer.step()
    #             pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # get SE3 poses to do forward pass
    #             poses_all = torch.cat([poses_fixed, pose_optim], dim=0)  # SE3 poses
    #             pose_optimizer.zero_grad()
    #
    #     # Step 4: update related keyframe's poses
    #     if pose_optimizer is not None and len(first_ovlp_kf_Ids) > 1:
    #         for i in range( len(first_ovlp_kf_Ids[1:]) ):
    #             kf_id = first_ovlp_kf_Ids[1:][i]
    #             pose_world = poses_all[1:][i]  # Tensor(4, 4)
    #             self.slam.kf_c2w[kf_id] = pose_world
    #
    #     self.glob_BA_round[0] = self.glob_BA_round[0] + 1


    # @brief: do global BA with overlapping keyframe
    # @param ovlp_kf_Id: keyframe_Id of the overlapping keyframe which triggers the loop, Tensor(, );
    # @param ovlp_frame_Id: frame_Id of the overlapping keyframe which triggers the loop, Tensor(, );
    def global_BA_overlapping(self, ovlp_kf_Id, ovlp_frame_Id):
        self.merge_active_model_copy()
        key_pose_local = self.slam.est_c2w_data[ovlp_frame_Id].detach()  # local pose of this overlapping in previous active localMLP
        kf_num = self.kfSet.collected_kf_num[0]

        # Step 1: find all adjacent localMLP pairs
        adja_pairs, part_localMLPs = self.kfSet.find_adjacent_localMLP_pair()  # Tensor(k, 2) / Tensor(m, )
        if part_localMLPs.shape[0] >= 2 and part_localMLPs.shape[0] == len(self.model_list):
            pass
        else:
            return
        localMLP_first_kf = self.kfSet.localMLP_first_kf.detach()
        keyframe_ref = self.slam.keyframe_ref.detach()

        # Step 2: extract optimizeble poses and optimizer
        # Step 2.1: find keyframe_Id and keyframe pose(world) of all first keyframes (these poses are optimizable)
        first_kf_Ids = torch.where(keyframe_ref == -1)[0]  # each localMLP's first keyframe
        first_kf_pose = self.slam.kf_c2w[first_kf_Ids].detach()  # *** optimizable vars

        # 2.2: construct pose optimizer
        first_poses_fixed = torch.nn.parameter.Parameter(first_kf_pose[:1]).to(self.device)  # pose of keyframe_0 is always fixed
        cur_rot, cur_trans, pose_optimizer = self.slam.get_pose_param_optim(first_kf_pose[1:])
        first_pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
        first_poses_all = torch.cat([first_poses_fixed, first_pose_optim], dim=0)

        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        # 2.3: find keyframe_Id and pose(world) of all keyframes which are bound to more than 1 localMLP (these poses are not optimizable)
        ovlp_kf_Ids = self.kfSet.find_ovlp_kf_Ids(kf_num)  # kf_Ids of selected keyframes (excluding latest overlapping keyframe)
        kf_localMLP_ovlp = self.kfSet.keyframe_localMLP[ovlp_kf_Ids].detach()  # keyframe-localMLP relationships of selected keyframes
        kf_ref_ovlp = self.slam.keyframe_ref[ovlp_kf_Ids].detach()  # keyframe ref type of selected keyframes
        localMLP_idx = torch.zeros_like(ovlp_kf_Ids)
        ovlp_kf_poses = self.kfSet.extract_kf_world_poses(ovlp_kf_Ids, localMLP_idx[..., None], kf_ref_ovlp, self.slam.est_c2w_data.detach(), first_poses_all.detach()).detach()  # world poses (non-optimizable)


        # Step 3: do global BA
        bs = max(self.config["mapping"]["sample"] // adja_pairs.shape[0], self.config["mapping"]["sample"] // 4)
        for i in range(20):
            loss = 0.
            for adja_pair in adja_pairs:
                localMLP_Id1 = adja_pair[0]
                localMLP_Id2 = adja_pair[1]

                # 3.1: find these 2 localMLP's first keyframes' poses (extract them from first_poses_all)
                first_kf_pose1 = first_poses_all[localMLP_Id1]
                first_kf_pose2 = first_poses_all[localMLP_Id2]

                # 3.2: find overlapping region of these 2 localMLPs
                ovlp_kf_indices = find_related_localMLPs(kf_localMLP_ovlp, localMLP_Id1, localMLP_Id2)  # indice of related_kf in ovlp_kf_Ids, Tensor(m, )
                if ovlp_kf_indices.shape[0] == 0:
                    continue

                related_ovlp_kf_Ids = ovlp_kf_Ids[ovlp_kf_indices]  # related overlapping keyframes' kf_Id, Tensor(m, )
                related_ovlp_kf_poses = ovlp_kf_poses[ovlp_kf_indices]  # related overlapping keyframes' pose in World Coordinate System, Tensor(m, 4, 4)

                # 3.3: sample pixels from related overlapping keyframes, and find corresponding pose of each selected ray
                rays, kf_indices = self.kfSet.sample_rays_from_given(related_ovlp_kf_Ids, bs)  # Tensor(bs, 7) / Tensor(bs, )
                selected_ovlp_kf_poses = related_ovlp_kf_poses[kf_indices]  # Tensor(bs, 4, 4)

                # 3.4: get loss term of this adjacent pair
                loss_avg = self.get_SDF_dif(rays, selected_ovlp_kf_poses, localMLP_Id1, localMLP_Id2, first_kf_pose1, first_kf_pose2)
                loss += 5. * loss_avg
            # end for

            # 3.4: compute loss term of the overlapping keyframe which triggers the loop, only using filtered points in overlappin region
            ovlp_depth = self.ovlp_depth[0].unsqueeze(-1)  # depth of sampled pts in Camera Coordinate System, Tensor(n_rays_h * n_rays_w, 1)
            ovlp_rays = self.ovlp_rays[0]  # dir of sampled pts in Camera Coordinate System, Tensor(n_rays_h * n_rays_w, 3)
            ovlp_pts_mask = self.ovlp_pts_mask[0].unsqueeze(-1)  # overlapping mask, Tensor(n_rays_h * n_rays_w, 1)

            # 3.4.1: extract first keyframe's pose of these 2 related localMLPs
            localMLP_Id1, localMLP_Id2 = self.kfSet.keyframe_localMLP[ovlp_kf_Id].detach()
            first_kf_pose1 = first_poses_all[localMLP_Id1]  # active localMLP_Id after switch
            first_kf_pose2 = first_poses_all[localMLP_Id2]  # active localMLP_Id before switch

            # 3.4.2: compute loss term
            key_kf_pose_world = first_kf_pose2.detach() @ key_pose_local  # world pose of this keyframe, Tensor(4, 4)
            loss_ovlp = self.get_SDF_dif2(ovlp_depth, ovlp_rays, ovlp_pts_mask, key_kf_pose_world[None, ...], localMLP_Id1, localMLP_Id2, first_kf_pose1, first_kf_pose2)
            loss += 100. * loss_ovlp

            loss.backward(retain_graph=True)

            # 3.5: update poses (accumulate gradients)
            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                first_pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # get SE(3) poses to do forward pass
                first_poses_all = torch.cat([first_poses_fixed, first_pose_optim], dim=0)  # SE3 poses
                pose_optimizer.zero_grad()

                # since each first kf pose is updated, all overlapping keyframes' poses should also be updated
                ovlp_kf_poses = self.kfSet.extract_kf_world_poses(ovlp_kf_Ids, localMLP_idx[..., None], kf_ref_ovlp, self.slam.est_c2w_data.detach(), first_poses_all.detach()).detach()  # world poses (non-optimizable)
        # END for

        # Step 4: update related keyframe's poses (only first keyframes are updated)
        if pose_optimizer is not None and len(first_kf_Ids) > 1:
            for i in range( len(first_kf_Ids[1:]) ):
                kf_id = first_kf_Ids[1:][i]
                pose_world = first_poses_all[1:][i].detach()  # Tensor(4, 4)
                self.slam.kf_c2w[kf_id] = pose_world
    # END global_BA_overlapping()


    # @brief: do global BA with overlapping keyframe
    def global_BA(self):
        ovlp_kf_Id = self.slam.key_keyframe_Id[0]  # keyframe_Id of the overlapping keyframe which triggers the loop, Tensor(, );
        ovlp_frame_Id = ovlp_kf_Id * self.config["mapping"]["keyframe_every"]  # frame_Id of the overlapping keyframe which triggers the loop, Tensor(, );
        self.merge_active_model_copy()

        # Step 1: find all adjacent localMLP pairs
        adja_pairs, part_localMLPs = self.kfSet.find_adjacent_localMLP_pair()  # Tensor(k, 2) / Tensor(m, )
        if part_localMLPs.shape[0] >= 2 and part_localMLPs.shape[0] == len(self.model_list):
            pass
        else:
            return

        # Step 2: get local poses (in previous and current active localMLP respectively) of overlapping keyframe which triggered the loop
        kf_num = self.kfSet.collected_kf_num[0] - 1  # excluding the latest overlapping keyframe
        ovlp_kf_local_pose_prev = self.slam.temp_local_pose[0]
        ovlp_kf_local_pose_aft = self.slam.est_c2w_data[ovlp_frame_Id]
        localMLP_Id_aft, localMLP_Id_prev = self.kfSet.keyframe_localMLP[ovlp_kf_Id]
        self.poseCorrector.pose_graph_optimize(kf_num, adja_pairs, ovlp_kf_local_pose_prev, ovlp_kf_local_pose_aft, localMLP_Id_prev, localMLP_Id_aft)

        print(printCurrentDatetime() + "(InactiveMap process) Global BA finished!")


    # @brief: entry function of Active Mapping process
    def run(self):
        print(printCurrentDatetime() + "(Inactive Mapping process) Process starts!!! (PID=%d)" % os.getpid())

        while True:
            if self.slam.inactive_start[0] == 1:
                break
            if self.slam.seq_end[0] == 1:  # sequence end
                break

            # save checkpoint when ActiveMap process sent request
            if self.slam.ckpt_frame_Id[0] > self.last_saved_ckpt[0]:
                ckpt_frame_Id = self.slam.ckpt_frame_Id[0].clone()
                self.logger.save_ckpt_inactive(ckpt_frame_Id, self.model_list, self.slam.active_localMLP_Id[0])
                self.last_saved_ckpt[0] = ckpt_frame_Id

        # main process
        print(printCurrentDatetime() + "(InactiveMap) Begin to optimize inactive localMLPs...")
        while True:
            if self.slam.seq_end[0] == 1:  # sequence end
                break

            if self.shared_flag[0] > 0:
                self.process_active_submap_switch()
                time.sleep(1)

            if self.mesh_flag > 0:
                frame_id = int(self.mesh_flag[0])
                self.extract_mesh(frame_id)
                self.mesh_flag[0] = 0

            if self.do_globalBA[0]:
                self.global_BA()
                self.do_globalBA[0] = False

            # save checkpoint when ActiveMap process sent request
            if self.slam.ckpt_frame_Id[0] > self.last_saved_ckpt[0]:
                ckpt_frame_Id = self.slam.ckpt_frame_Id[0].clone()
                self.logger.save_ckpt_inactive(ckpt_frame_Id, self.model_list, self.slam.active_localMLP_Id[0])
                self.last_saved_ckpt[0] = ckpt_frame_Id

            # do ordinary local BA for inactive localMLPs alternately and continually
            self.local_BA()
            time.sleep(self.sleep_time)

            # self.logger.extract_a_mesh(self.slam.tracked_frame_Id, 0, self.model_list[0])  # for debug

            # if self.counter % 5 == 0:
            #     self.global_BA()
            self.counter += 1
            if self.counter % 10 == 0:
                torch.cuda.empty_cache()

        self.logger.save_ckpt_inactive(self.slam.tracked_frame_Id[0], self.model_list, self.slam.active_localMLP_Id[0], final=True)
        print(printCurrentDatetime() + "InactiveMap Process ends!!!! PID=", os.getpid())
