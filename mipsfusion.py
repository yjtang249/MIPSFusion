import os
import torch
import torch.optim as optim
import numpy as np
import random
import time
import cv2
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion
from RandomOptimizer import RandomOptimizer
from PoseCorrector import PoseCorrector
from Logger import Logger
from Manager import Manager
from InactiveMap import InactiveMap
from helper_functions.printTime import printCurrentDatetime
from helper_functions.sampling_helper import pixel_rc_to_indices, sample_pixels_random, sample_valid_pixels_random, sample_pixels_mix
from helper_functions.geometry_helper import get_frame_surface_bbox, extract_first_kf_pose


class MIPSFusion():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.create_active_localMLP_vars()

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.kfSet = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box, self.coords_norm_factor).to(self.device)  # active localMLP
        self.randomOptimizer = RandomOptimizer(self.config, self)
        self.logger = Logger(self.config, self)
        self.poseCorrector = PoseCorrector(self.config, self)
        self.manager = Manager(self.config, self)

        self.create_shared_data()
        self.create_global_BA_data()
        self.inactive_map = InactiveMap(self.config, self)
        self.inactive_start = torch.zeros((1, )).share_memory_()  # whether InactiveMap process starts
        self.seq_end = torch.zeros((1,)).share_memory_()  # whether the input sequence ends
        self.process_flag = 1


    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    # @brief:  Get the pose representation axis-angle or quaternion
    def get_pose_representation(self):
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
        elif self.config['training']['rot_rep'] == "quat":  # default
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError


    # @brief: create pose tensor (absolute pose and relative pose)
    def create_pose_data(self):
        self.num_frames = self.dataset.num_frames
        self.num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  # max keyframe number

        # type of each keyframe. -1: first keyframe of a submap; -2: overlapping keyframe(wo switch); -3: overlapping keyframe(w switch); n(n>=0): ref keyframe_Id of this keyframe
        self.keyframe_ref = torch.full((self.num_kf, ), -3, dtype=torch.int32).share_memory_()

        # for each overlapping keyframe(indexed by kf_Id), this tensor records the lastly optimized process (1: ActiveMap, -1: InactiveMap)
        self.overlap_kf_flag = torch.zeros((self.num_kf, )).share_memory_()

        self.kf_c2w = torch.zeros((self.num_kf, 4, 4)).to(self.device).share_memory_()  # store absolute pose in World Coordinate System of each localMLP's first keyframe or overlapping keyframes
        self.est_c2w_data = torch.zeros((self.num_frames, 4, 4)).to(self.device).share_memory_()  # store absolute pose in Local Coordinate System of each frame (keyframe)
        self.est_c2w_data_rel = torch.eye(4).repeat((self.num_frames, 1, 1)).to(self.device).share_memory_()  # store relative pose in Local Coordinate System of each frame
        self.load_gt_pose()  # load gt poses
        self.temp_local_pose = torch.zeros((1, 4, 4)).to(self.device).share_memory_()  # pose of triggering overlapping keyframe in previous active_localMLP's CS
        self.rectified_local_pose = torch.eye(4).unsqueeze(0).share_memory_()  # pose of triggering overlapping keyframe in latter active_localMLP's CS (rectified)

        self.optim_cur = self.config['mapping']['optim_cur']  # whether optimize current pose in local BA


    # @brief: create active localMLP-related vars
    def create_active_localMLP_vars(self):
        self.active_localMLP_Id = torch.zeros((1, )).to(torch.int64).share_memory_()  # localMLP_Id of currently active localMLP
        self.prev_active_localMLP_Id = torch.full((1, ), fill_value=-1).to(torch.int64).share_memory_()  # active localMLP_Id of before latest switch

        self.active_first_kf = torch.zeros((1, )).to(torch.int64).share_memory_()  # first kf's keyframe_Id of currently active submap
        self.last_switch_frame = torch.zeros((1, )).to(torch.int64).share_memory_()  # frame_Id of last active submap switch
        self.last_ovlp_kf_Id = -1 * torch.ones(1, ).to(torch.int64).share_memory_()


    # @brief: get axis-aligned bounds
    def create_bounds(self):
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)
        self.coords_norm_factor = torch.from_numpy(np.array(self.config["mapping"]["localMLP_max_len"])).to(self.device)


    # @brief: Create the keyframe database
    def create_kf_database(self, config):
        if self.num_kf is None:
            self.num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  # max keyframe number

        return KeyFrameDatabase(config, self.dataset.H, self.dataset.W, self.num_kf, self.device)


    # @brief: create shared data between ActiveMap and InactiveMap processes
    def create_shared_data(self):
        self.localMLP_Id_a2i = torch.zeros((1, ), dtype=torch.int64).share_memory_()
        self.localMLP_Id_i2a = torch.zeros((1, ), dtype=torch.int64).share_memory_()
        self.localMLP_Id_asked = torch.full((1,), fill_value=-1, dtype=torch.int64).share_memory_()

        self.shared_model = JointEncoding(self.config, self.bounding_box, self.coords_norm_factor).to(self.device).share_memory()

        #  0: nothing; 1: (a2i) switch to a new localMLP; 2: (a2i) switch to a previous localMLP; -1: i2a
        self.shared_flag = torch.zeros((1, )).share_memory_()
        self.mesh_flag = torch.zeros((1, )).share_memory_()
        self.tracked_frame_Id = torch.zeros((1, ), dtype=torch.int64).share_memory_()
        self.ckpt_frame_Id = torch.zeros((1,), dtype=torch.int64).share_memory_()


    def create_global_BA_data(self):
        self.do_globalBA = torch.zeros((1, ), dtype=torch.bool).share_memory_()
        self.key_keyframe_Id = torch.zeros((1, ), dtype=torch.int64).share_memory_()

    # @brief: Load the ground truth pose, fill self.pose_gt
    def load_gt_pose(self):
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose


    # @brief: randomly select samples from the image
    def select_samples(self, H, W, samples):
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice


    # @brief: get the training loss from all loss terms
    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']  # default weight: 5
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']  # default weight: 1
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]  # default weight: 1000
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]  # default weight: 10
        return loss             


    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)  # gt pose(c2w) of first pose, Tensor(4, 4), device=cuda:0
        c2w_local = torch.eye(4).to(self.device)

        # Step 1: fill keyframe-related, localMLP-related vars for the first keyframe
        self.kf_c2w[0] = c2w
        self.est_c2w_data[0] = c2w_local
        self.keyframe_ref[0] = -1

        self.kfSet.localMLP_first_kf[0] = 0
        xyz_center, xyz_len = get_frame_surface_bbox(batch["c2w"].squeeze(0), batch["depth"].squeeze(0), batch["direction"].squeeze(0),
                                                     self.config["cam"]["near"], self.config["cam"]["far"])
        self.kfSet.localMLP_info[0] = torch.cat( [torch.ones((1)), xyz_center, xyz_len], 0 )
        self.kfSet.keyframe_localMLP[0, 0] = 0
        self.kfSet.localMLP_first_kf[0] = 0
        self.kfSet.collected_kf_num[0] = self.kfSet.collected_kf_num[0] + 1

        # Step 2: training scene representation (default: n_iters=500)
        self.model.train()
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])  # sample pixels every round(default: 2048)
            indice_h, indice_w = torch.remainder(indice, self.dataset.H), torch.div(indice, self.dataset.H, rounding_mode="floor")  # selected col_Id, row_Id

            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)  # dir of sampled rays in Camera Coords, Tensor(sample_num, 3), device=cuda:0
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)  # gt RGB of sampled rays in Camera Coords, Tensor(sample_num, 3), device=cuda:0
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)  # gt depth of sampled rays in Camera Coords, Tensor(sample_num, 1), device=cuda:0

            rays_o = c2w_local[None, :3, -1].repeat(self.config['mapping']['sample'], 1)  # Tensor(N, 3)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_local[:3, :3], -1)  # Tensor(N, 3)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame must be keyframe and will stay fixed in the following optimization
        self.kfSet.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        
        print('First frame mapping done')
        return ret, loss


    # @brief: do initialization for a newly created localMLP
    def initialize_new_localMLP(self, batch, n_iters=100):
        print(printCurrentDatetime() + "(ActiveMap) Begin to initialize localMLP_%d..." % self.active_localMLP_Id[0])
        self.create_optimizer()

        # Step 2: training scene representation (default: n_iters=500)
        self.model.train()
        c2w_local = torch.eye(4).to(self.device)

        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])  # sample pixels every round(default: 2048)
            indice_h, indice_w = torch.remainder(indice, self.dataset.H), torch.div(indice, self.dataset.H, rounding_mode="floor")  # selected col_Id, row_Id

            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)  # dir of sampled rays in Camera Coords, Tensor(sample_num, 3), device=cuda:0
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)  # gt RGB of sampled rays in Camera Coords, Tensor(sample_num, 3), device=cuda:0
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)  # gt depth of sampled rays in Camera Coords, Tensor(sample_num, 1), device=cuda:0

            rays_o = c2w_local[None, :3, -1].repeat(self.config['mapping']['sample'], 1)  # Tensor(N, 3)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_local[:3, :3], -1)  # Tensor(N, 3)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        print(printCurrentDatetime() + "(ActiveMap) Finished initializing localMLP_%d !!!" % self.active_localMLP_Id[0])


    # @brief: freeze the model parameters
    def freeze_model(self):
        for param in self.model.embed_fn.parameters():
            param.require_grad = False
        for param in self.model.decoder.parameters():
            param.require_grad = False


    # @brief: contsruct optimizable pose
    # @param poses: Tensor(n, 4, 4)
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])  # Parameter(n, 3)
        cur_rot = torch.nn.parameter.Parameter( self.matrix_to_tensor( poses[:, :3, :3] ) )  # rot mat --> quat, Parameter(n, 4)
        pose_optimizer = torch.optim.Adam( [ { "params": cur_rot, "lr": self.config[task]['lr_rot'] },
                                             { "params": cur_trans, "lr": self.config[task]['lr_trans'] } ] )
        return cur_rot, cur_trans, pose_optimizer

    # @brief: contsruct optimizable pose
    # @param poses: Tensor(n, 4, 4)
    def get_pose_param_optim_switch(self, poses):
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])  # Parameter(n, 3)
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))  # rot mat --> quat, Parameter(n, 4)
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config["tracking"]["switch"]["lr_rot"]},
                                           {"params": cur_trans, "lr": self.config["tracking"]["switch"]["lr_trans"]}])
        return cur_rot, cur_trans, pose_optimizer


    # @brief: do local BA in active submap
    # @param batch['c2w']: ground truth camera pose, Tensor(1, 4, 4);
    #        batch['rgb']: rgb image, Tensor(1, H, W, 3);
    #        batch['depth']: depth image, Tensor(1, H, W);
    #        batch['direction']: view direction, Tensor(1, H, W, 3);
    # @param cur_frame_id: current frame id, int.
    def local_BA(self, batch, cur_frame_id):
        pose_optimizer = None

        # Step 1: select all related keyframes' local poses / kf_Ids / frame_Ids
        first_kf_pose, first_kf_Id, poses, kf_ids_all, frame_ids_all, related_kf_ref, related_ov_kf_idx, related_ov_kf_Ids \
            = self.kfSet.extract_localMLP_vars(self.active_localMLP_Id[0], self.kf_c2w, self.est_c2w_data, self.keyframe_ref, self.process_flag)

        # Step 2: construct optimizer for fixed kf_poses and optimizable kf_poses (current_pose will bot be optimized)
        if len(kf_ids_all) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)  # pose of first kf is always fixed
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]

            if self.optim_cur:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim( torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # related keyframes' poses (except first keyframe pose)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # related keyframes' poses (except first keyframe pose)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']
        current_rays_raw = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)  # Tensor(1, H, W, 7), device=cpu
        current_rays = current_rays_raw.reshape(-1, current_rays_raw.shape[-1])  # Tensor(H * W, 7), device=cpu

        # Step 3: perform n_iters BA
        for i in range(self.config['mapping']['iters']):
            # 3.1: sampling pixels from stored keyframes
            rays, kf_ids, kf_indices = self.kfSet.sample_rays_in_submap(first_kf_Id, kf_ids_all, self.config['mapping']['sample'])

            # # 3.2: sample pixels from current frame (random sampling + uniform sampling)
            if self.config["tracking"]["iter_RO"] == 0:
                # (1) random sampling
                pixel_num_cur = max( self.config['mapping']['sample'] // kf_ids_all.shape[0], 50 )  # number of sampled pixels from current frame
                # idx_cur = random.sample( range(0, self.dataset.H * self.dataset.W), pixel_num_cur )
                idx_cur = sample_valid_pixels_random(batch["depth"][0], pixel_num_cur)
                current_rays_batch = current_rays[idx_cur, :]
            else:
                # (2) uniform sampling + random sampling
                pixel_num_cur = max( self.config['mapping']['sample'] // kf_ids_all.shape[0], self.config['mapping']['pixels_cur'] )  # number of sampled pixels from current frame
                row_indices, col_indices = sample_pixels_mix(self.dataset.H, self.dataset.W, self.config["tracking"]["RO"]["n_rows"],
                                                             self.config["tracking"]["RO"]["n_cols"], batch["depth"][0], pixel_num_cur)
                current_rays_batch = current_rays_raw.squeeze(0)[row_indices, col_indices]  # Tensor(pixel_num_cur, 7)

            # 3.3: concatenation
            rays = torch.cat([rays, current_rays_batch], dim=0)  # Tensor(N, 7), device=cpu
            indices_all = torch.cat( [ kf_indices, -torch.ones((pixel_num_cur,)) ] ).to(torch.int64)  # corresponding keyframe_Id of each sampled ray

            rays_d_cam = rays[..., :3].to(self.device)  # direction (in Camera Coords) of sampled pixels(rays), Tensor(N, 3), device=cuda
            target_s = rays[..., 3:6].to(self.device)  # gt RGB of sampled pixels(rays), Tensor(N, 3), device=cuda:0
            target_d = rays[..., 6:7].to(self.device)  # gt depth of sampled pixels(rays), Tensor(N, 1), device=cuda:0

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3) (Camera coords -> World coords)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[indices_all, None, :3, :3], -1)
            rays_o = poses_all[indices_all, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 3.4: inference
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            # loss = self.get_loss_from_ret(ret, smooth=True)
            loss = self.get_loss_from_ret(ret)
            loss.backward(retain_graph=True)

            # 3.5: update model
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            # 3.6: update poses
            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # get SE3 poses to do forward pass
                # current_pose = self.est_c2w_data[cur_frame_id][None, ...]
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)  # SE3 poses
                pose_optimizer.zero_grad()
        # END for

        # Step 4: update related keyframe's poses
        if pose_optimizer is not None and len(kf_ids_all) > 1:
            for i in range( len(kf_ids_all[1:]) ):
                pose_local = self.matrix_from_tensor(cur_rot[i: i + 1], cur_trans[i: i + 1]).detach().clone()[0]  # optimized local pose, Tensor(4, 4)
                if related_kf_ref[1:][i] >= 0:  # for ordinary keyframes
                    frame_id = frame_ids_all[1:][i]
                    self.est_c2w_data[frame_id] = pose_local
                elif related_kf_ref[1:][i] == -1:  # for first keyframes of other localMLPs
                    kf_id = kf_ids_all[1:][i]
                    pose_world = first_kf_pose @ pose_local  # Tensor(4, 4)
                    self.kf_c2w[kf_id] = pose_world
                else:  # for overlapping keyframes
                    frame_id = frame_ids_all[1:][i]
                    kf_id = kf_ids_all[1:][i]
                    if self.active_localMLP_Id[0] == self.kfSet.keyframe_localMLP[kf_id, 0]:  # if current active localMLP is its first related localMLP
                        self.est_c2w_data[frame_id] = pose_local
                    else:  # if current active localMLP is its second related localMLP
                        pose_world = first_kf_pose @ pose_local
                        first_kf_pose_another = extract_first_kf_pose(self.kfSet.keyframe_localMLP[kf_id, 0], self.kfSet.localMLP_first_kf, self.kf_c2w).detach()
                        pose_local_another = first_kf_pose_another.inverse() @ pose_world
                        self.est_c2w_data[frame_id] = pose_local_another

            if self.optim_cur:
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

        if related_ov_kf_Ids.shape[0] > 0:
            self.overlap_kf_flag[related_ov_kf_Ids] = self.process_flag


    # @brief: do local BA for loop-triggering keyframe with its nearest top K keyframes;
    # @param batch['c2w']: ground truth camera pose, Tensor(1, 4, 4);
    #        batch['rgb']: rgb image, Tensor(1, H, W, 3);
    #        batch['depth']: depth image, Tensor(1, H, W);
    #        batch['direction']: view direction, Tensor(1, H, W, 3);
    # @param cur_frame_id: current frame id, int.
    def local_BA_switch(self, batch, overlap_kf_id, overlap_frame_id):
        pose_optimizer = None

        # Step 1: select all related keyframes' local poses / kf_Ids / frame_Ids
        first_kf_pose, first_kf_Id, poses, kf_ids_all, frame_ids_all, related_kf_ref, _, _ \
            = self.kfSet.extract_localMLP_vars_given(self.active_localMLP_Id[0], self.kfSet.nearest_kf_Ids[0], self.kf_c2w, self.est_c2w_data, self.keyframe_ref)

        # Step 2: construct optimizer for fixed kf_poses and optimizable kf_poses (current_pose will bot be optimized)
        if len(kf_ids_all) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            poses_all = poses_fixed
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)  # pose of first kf is always fixed
            ovlp_kf_pose = self.est_c2w_data[overlap_frame_id].detach().unsqueeze(0)
            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim_switch(ovlp_kf_pose)
            pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

        # Set up optimizer
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        ovlp_kf_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)  # Tensor(1, H, W, 7), device=cpu
        ovlp_kf_rays = ovlp_kf_rays.reshape(-1, ovlp_kf_rays.shape[-1])  # Tensor(H * W, 7), device=cpu

        # Step 3: perform n_iters BA
        for i in range(self.config["tracking"]["switch"]["map_num"]):
            pix_num_ovlp = max(self.config['mapping']['sample'] // kf_ids_all.shape[0], self.config['mapping']['sample'] // 5)

            # 3.1: sampling pixels from stored keyframes
            rays, kf_ids, kf_indices = self.kfSet.sample_rays_in_given_kf(kf_ids_all, self.config['mapping']['sample'])

            # 3.2: sampling pixels from this overlapping keyframe
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), pix_num_ovlp)
            ovlp_rays_batch = ovlp_kf_rays[idx_cur, :]

            # 3.3: concatenation
            rays = torch.cat([rays, ovlp_rays_batch], dim=0)  # N, 7, device=cpu
            indices_all = torch.cat( [ kf_indices, -torch.ones((pix_num_ovlp, )) ] ).to(torch.int64)  # corresponding keyframe_Id of each sampled ray

            rays_d_cam = rays[..., :3].to(self.device)  # direction (in Camera Coords) of sampled pixels(rays), Tensor(N, 3), device=cuda
            target_s = rays[..., 3:6].to(self.device)  # gt RGB of sampled pixels(rays), Tensor(N, 3), device=cuda:0
            target_d = rays[..., 6:7].to(self.device)  # gt depth of sampled pixels(rays), Tensor(N, 1), device=cuda:0

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3) (Camera coords -> World coords)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[indices_all, None, :3, :3], -1)
            rays_o = poses_all[indices_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 3.2: inference
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward(retain_graph=True)

            # 3.3: update poses
            # if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)  # get SE3 poses to do forward pass

                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)  # SE3 poses
                pose_optimizer.zero_grad()
        # END for

        # Step 4: update this overlapping keyframe's pose (Local pose)
        if pose_optimizer is not None and len(kf_ids_all) > 1:
            pose_local_ovlp = self.matrix_from_tensor(cur_rot, cur_trans).detach().clone()[0]
            self.est_c2w_data[overlap_frame_id] = pose_local_ovlp


    # @brief: predict initial value of current pose from previous pose using camera motion model
    def predict_current_pose(self, frame_id, constant_speed=True):
        if constant_speed and (frame_id - self.last_switch_frame[0]) >= 2:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev
        else:
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
        
        return self.est_c2w_data[frame_id]


    # @brief: Tracking camera pose of the current frame;
    # @param batch['c2w']: ground truth camera pose, Tensor(B, 4, 4);
    #        batch['rgb']: RGB image, Tensor(B, H, W, 3);
    #        batch['depth']: depth image, Tensor(B, H, W, 1);
    #        batch['direction']: ray direction, Tensor(B, H, W, 3);
    # @param frame_id: current frame_Id (int);
    # @param n_iter_RO: iter num for RO, int;
    # @param n_iter_GO: iter num for RO, int;
    # @param switch_tracking: whether it's processing active submap switch, bool.
    def tracking_render(self, batch, frame_id, n_iter_RO, n_iter_GO, switch_tracking=False):
        if switch_tracking:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            const_speed = self.config["tracking"]["const_speed"]
            cur_c2w = self.predict_current_pose(frame_id, const_speed)  # get initial pose of this frame, Tensor(4, 4)
        self.freeze_model()

        # Step 1: do RO
        if n_iter_RO > 0:
            last_pose = self.est_c2w_data[frame_id - 1].clone().to(self.device)
            if switch_tracking:
                last_pose = self.est_c2w_data[frame_id].clone().to(self.device)
                cur_c2w = self.randomOptimizer.optimize(self.model, batch['depth'].squeeze(0), cur_c2w.clone(), last_pose, n_iter_RO).to(self.device)
            else:
                cur_c2w = self.randomOptimizer.optimize(self.model, batch['depth'].squeeze(0), cur_c2w.clone(), last_pose, n_iter_RO).to(self.device)

        indice = None
        best_sdf_loss = None
        thresh = 0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        # from initial current_pose to construct optimizable pose
        if switch_tracking:
            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim_switch(cur_c2w[None, ...])
        else:
            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None, ...], mapping=False)

        # Step 2: do GO
        for i in range(n_iter_GO):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)  # quat + trans_vec --> 4X4 mat, Tensor(1, 4, 4), device=cuda:0

            # (pixel sampling) Note here we fix the sampled points for optimization
            # only resampling pixels in the first optimization round
            if indice is None:
                if self.config["tracking"]["iter_RO"] == 0:
                    # (1) random sampling
                    indice = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2, self.config['tracking']['sample'])  # pixel sampling, default: 1000
                    # indice = sample_valid_pixels_random(batch['depth'].squeeze(0)[iH:-iH, iW:-iW], self.config['tracking']['sample'])  # # randomly sampling among all valid pixels

                    indice_h, indice_w = torch.remainder(indice, self.dataset.H-iH*2), torch.div(indice, self.dataset.H-iH*2, rounding_mode="floor")
                    rays_d_cam = batch['direction'].squeeze(0)[iH: -iH, iW: -iW, :][indice_h, indice_w, :].to(self.device)  # Tensor(N, 3), device=cuda
                else:
                    # (2) uniform + random sampling
                    rows_Id, cols_Id = sample_pixels_mix(self.dataset.H, self.dataset.W, self.config["sampling"]["n_rays_h"], self.config["sampling"]["n_rays_w"],
                                                         batch['depth'][0], self.config["tracking"]["sample"])
                    indice = pixel_rc_to_indices(rows_Id, cols_Id, self.dataset.H, self.dataset.W)
                    rays_d_cam = batch['direction'].squeeze(0)[rows_Id, cols_Id, :].to(self.device)  # Tensor(N, 3), device=cuda

            if self.config["tracking"]["iter_RO"] == 0:
                # (1) random sampling
                target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            else:
                # (2) uniform + random sampling
                target_s = batch['rgb'].squeeze(0)[rows_Id, cols_Id, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[rows_Id, cols_Id].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)  # apply translation(camera --> world)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)  # apply rotation(camera --> world): rotate direction of sampled rays

            ret = self.model.forward(rays_o, rays_d, target_s, target_d, EMD_w=0.)
            loss = self.get_loss_from_ret(ret)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)
                # print("TEST_2: \n", c2w_est)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()

        if self.config['tracking']['best']:  # default
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        # Step 3: Save relative pose for non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:  # case 1: for non-keyframe
            kf_id = frame_id // self.config['mapping']['keyframe_every']  # keyframe_Id of ref keyframe
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']  # frame_Id of ref keyframe

            kf_pose_local = self.est_c2w_data[kf_frame_id]
            delta = kf_pose_local.inverse() @ self.est_c2w_data[frame_id]
            self.est_c2w_data_rel[frame_id] = delta
        else:  # case 2: for keyframe
            if switch_tracking == False:
                kf_id = frame_id // self.config['mapping']['keyframe_every']  # keyframe_Id of this keyframe
                self.keyframe_ref[kf_id] = self.active_first_kf[0]


    # @brief: Create optimizer for mapping (embedding + MLP)
    def create_optimizer(self):
        trainable_parameters = [ { 'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder'] },
                                 { 'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed'] }]
    
        if not self.config['grid']['oneGrid']:  # default: skip if
            trainable_parameters.append( { 'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color'] } )
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))


    #-@return: local pose in current active localMLP's Local CS, Tensor(4, 4);
    #-@return: local pose in previous active localMLP's CS, Tensor(4, 4).
    def current_pose_switch_submap(self, frame_Id, keyframe_Id, prev_active_localMLP_Id=None, active_localMLP_Id=None):
        if prev_active_localMLP_Id is None:
            prev_active_localMLP_Id = self.prev_active_localMLP_Id[0]
        if active_localMLP_Id is None:
            active_localMLP_Id = self.active_localMLP_Id[0]

        # Step 1: find first keyframe of new active localMLP
        first_kf_pose_pev = extract_first_kf_pose(prev_active_localMLP_Id, self.kfSet.localMLP_first_kf, self.kf_c2w)  # Tensor(4, 4)
        first_kf_pose_aft = extract_first_kf_pose(active_localMLP_Id, self.kfSet.localMLP_first_kf, self.kf_c2w)  # Tensor(4, 4)
        # pose_world = self.kf_c2w[keyframe_Id]  # world pose of this keyframe
        pose_world = first_kf_pose_pev @ self.est_c2w_data[frame_Id]  # world pose of this keyframe

        # Step 2: convert to local pose
        cur_pose_local = first_kf_pose_aft.inverse() @ pose_world
        return cur_pose_local, self.est_c2w_data[frame_Id].clone()


    # @brief: processing function when active_localMLP is switched to a previous localMLP
    def active_submap_switch(self, frame_Id, keyframe_Id, batch):
        self.last_ovlp_kf_Id[0] = keyframe_Id

        # Step 1: modify active localMLP-related vars
        self.active_first_kf[0] = self.kfSet.localMLP_first_kf[self.active_localMLP_Id[0]]
        self.last_switch_frame[0] = frame_Id

        # Step 2: send current localMLP parameters to InactiveMap process
        self.localMLP_Id_a2i[0] = self.prev_active_localMLP_Id[0]
        self.shared_model.load_state_dict(self.model.state_dict())
        self.localMLP_Id_asked[0] = self.active_localMLP_Id[0]
        self.shared_flag[0] = 2

        # Step 3: wait for parameters of asked localMLP from InactiveMap process
        while True:
            if self.shared_flag[0] == -1:
                break
            time.sleep(0.1)

        # *** Step 4: set pose in Local Coordinate System of this keyframe
        # pose_local_ini = self.current_pose_switch_submap(frame_Id, keyframe_Id)
        # rectify_flag, pos_local_final = self.poseCorrector.switch_pose_rectifying_yuan(batch, pose_local_ini)

        self.temp_local_pose[0] = self.est_c2w_data[frame_Id]  # local pose of this overlapping keyframe in previous active_localMLP's CS
        self.est_c2w_data[frame_Id] = self.rectified_local_pose[0].clone().to(self.device)
        self.model.load_state_dict(self.shared_model.state_dict())
        self.shared_flag[0] = 0
        self.optim_cur = True
        print(printCurrentDatetime() + "(ActiveMap) Active submap switched to localMLP_%d" % (self.active_localMLP_Id[0]))


    # @brief: processing function when active_localMLP is switched to a new localMLP
    def active_submap_switch_new(self, frame_Id, keyframe_Id):
        # Step 1: send current localMLP parameters to InactiveMap process
        self.localMLP_Id_a2i[0] = self.prev_active_localMLP_Id[0]
        self.shared_model.load_state_dict(self.model.state_dict())
        self.shared_flag[0] = 1
        if self.inactive_start[0] == 0:
            self.inactive_start[0] = 1  # enable InactiveMap process

        # Step 2: set active localMLP new (recover to initial parameters)
        self.model.recover_initial_param()
        self.active_first_kf[0] = keyframe_Id
        self.last_switch_frame[0] = frame_Id

        # Step 3: set pose in Local Coordinate System of this keyframe
        self.est_c2w_data[frame_Id] = torch.eye(4).to(self.device)


    def inactive_map_start(self):
        self.inactive_map.run()


    # entry function
    def run(self):
        # Create InactiveMap process
        processes = []
        for rank in range(1):
            p = mp.Process(target=self.inactive_map_start, args=())
            p.start()
            processes.append(p)

        print(printCurrentDatetime() + "(Active Mapping process) Process starts!!! (PID=%d)" % os.getpid())

        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        for i, batch in tqdm( enumerate(data_loader) ):
            if i == 0:  # First frame mapping
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                self.logger.img_render_save(self.model, self.est_c2w_data[i], batch["rgb"].squeeze(0), batch["depth"].squeeze(0), 0)
            else:
                self.tracking_render(batch, i, self.config["tracking"]["iter_RO"], self.config["tracking"]["iter"])  # *** do tracking for each frame
    
                if i % self.config['mapping']['map_every'] == 0:  # *** do mapping every 5 frames
                    self.local_BA(batch, i)
                    self.inactive_map.active_model_copy.load_state_dict(self.model.state_dict())
                    self.inactive_map.active_model_copy_Id[0] = self.active_localMLP_Id[0]

                # Add keyframe (default: add a keyframe for every 15 frames)
                if i % self.config['mapping']['keyframe_every'] == 0:
                    kf_id = i // self.config['mapping']['keyframe_every']  # keyframe_Id of this keyframe
                    self.kfSet.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])

                    if (i - self.last_switch_frame[0]) <= self.config["tracking"]["switch_interval"]:
                        return_flag = self.manager.process_keyframe(batch, self.active_localMLP_Id[0], self.est_c2w_data[i], i, kf_id, force=True)
                    else:
                        return_flag = self.manager.process_keyframe(batch, self.active_localMLP_Id[0], self.est_c2w_data[i], i, kf_id)

                    # for active submap switch
                    if return_flag == 3:  # case 1: create a new localMLP
                        self.active_submap_switch_new(i, kf_id)
                        self.initialize_new_localMLP(batch, self.config['mapping']['first_iters'])
                    elif return_flag == 1:  # case 2: switch a previous localMLP
                        self.inactive_map.inactive_pause[0] = 1
                        self.active_submap_switch(i, kf_id, batch)
                        self.local_BA_switch(batch, kf_id, i)

                        self.key_keyframe_Id[0] = kf_id
                        self.do_globalBA[0] = True
                        self.inactive_map.inactive_pause[0] = 0

                    self.kfSet.collected_kf_num[0] = self.kfSet.collected_kf_num[0] + 1
                self.tracked_frame_Id[0] = i

                if i % self.config["mesh"]["vis"] == 0:
                    pose_relative = self.logger.convert_relative_pose(i)
                    pose_world = self.logger.convert_world_pose(pose_relative)
                    pose_evaluation(self.pose_gt, pose_world, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')
                    self.logger.save_traj_tum(pose_world, os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "traj_%d.txt" % i) )

                if self.config["mesh"]["ckpt_freq"] > 0 and i % self.config["mesh"]["ckpt_freq"] == 0:
                    self.logger.save_ckpt_active(self.tracked_frame_Id[0], self.model, self.active_localMLP_Id[0])
                    self.ckpt_frame_Id[0] = i


        # model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i))
        # self.logger.save_ckpt(model_savepath)
        # self.logger.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
        
        pose_relative = self.logger.convert_relative_pose(self.dataset.num_frames-1)
        pose_world = self.logger.convert_world_pose(pose_relative)

        pose_evaluation(self.pose_gt, pose_world, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')
        self.logger.save_traj_tum(pose_world, os.path.join(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "traj_%d.txt" % i)))

        self.logger.save_ckpt_active(self.tracked_frame_Id[0], self.model, self.active_localMLP_Id[0])
        self.seq_end[0] = 1
        print(printCurrentDatetime() + "seq end")

        for p in processes:
            p.join()

        print(printCurrentDatetime() + "ActiveMap Process ends!!!! PID=", os.getpid())
