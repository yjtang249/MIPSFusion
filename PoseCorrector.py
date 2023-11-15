import numpy as np
import copy
import torch
import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
import open3d as o3d

from model.poseGraph import PoseGraph
from model.CorrespondFinder import CorrespondFinder
from helper_functions.geometry_helper import extract_first_kf_pose
from helper_functions.sampling_helper import sample_pixels_uniformly
from testing.testing_func import vis_pc, save_pc, draw_registration_result


class PoseCorrector():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = SLAM.device
        self.dataset = self.slam.dataset
        self.kfSet = self.slam.kfSet

        self.kf_c2w = self.slam.kf_c2w
        self.est_c2w_data = self.slam.est_c2w_data
        self.keyframe_ref = self.slam.keyframe_ref

        self.poseGraph = None


    # @brief: updating pose graph according to keyframes collected so far
    def update_pose_graph(self, first_kf_Ids):
        first_kf_pose = self.kf_c2w[first_kf_Ids]  # first keyframes' poses of each localMLP
        if self.poseGraph is None:
            self.poseGraph = PoseGraph(first_kf_pose, self.device)
        else:
            self.poseGraph.update_param(first_kf_pose)


    # @brief: construct pointcloud from a given frame (downsampling first)
    # @param pose: Tensor(4, 4)
    def construct_pc(self, batch, pose):
        rays_d_cam = batch["direction"].squeeze(0)[self.kfSet.row_indices, self.kfSet.col_indices]  # Tensor(N, 3)
        target_depth = batch['depth'].squeeze(0)[self.kfSet.row_indices, self.kfSet.col_indices][..., None]  # Tensor(N, 1)

        # Camera coords -> World coords
        rays_d = torch.sum(rays_d_cam[:, None, :] * pose[None, :3, :3], -1)  # Tensor(N, 3)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = pose[None, :3, -1]  # Tensor(1, 3)
        pts_local = rays_o + rays_d * target_depth  # all points in local CS, Tensor(N, 3)

        depth_mask = (target_depth.squeeze(-1) > 0)
        pts_local_valid = pts_local[depth_mask].numpy()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_local_valid))
        pc.estimate_normals()
        return pc


    def merge_pc(self, pc1, pc2):
        merged_points = pc1.points
        merged_points.extend(pc2.points)
        merged_pc = o3d.geometry.PointCloud(merged_points)
        merged_pc.estimate_normals()
        return merged_pc


    # @brief: construct pointcloud from multiple keyframes, and convert them to the same Coordinate System;
    # @param kf_Ids: Tensor(n, );
    # @param poses: Tensor(n, 4, 4).
    def construct_pc_given_kfs(self, kf_Ids, poses):
        rays = self.kfSet.rays[kf_Ids]  # Tensor(n, num_rays_to_save, 7)
        rays_d_cam = rays[..., :3]  # Tensor(n, num_rays_to_save, 3)
        target_depth = rays[..., 6:7].reshape((-1, 1))  # Tensor(n, num_rays_to_save, 1)
        indices_all = torch.arange(kf_Ids.shape[0] * self.kfSet.num_rays_to_save) // self.kfSet.num_rays_to_save  # Tensor(n, num_rays_to_save)

        # Camera coords -> World coords
        # (n, num_rays_to_save, 1, 3) * (n, 1, 3, 3) --> (n, num_rays_to_save, 3)
        rays_d = torch.sum(rays_d_cam[:, :, None, :] * poses[:, None, :3, :3], -1)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = poses[indices_all, :3, -1].reshape(-1, 3)
        pts_local = rays_o + rays_d * target_depth  # all points in local CS, Tensor(n * num_rays_to_save, 3)

        depth_mask = ( target_depth.squeeze(-1) > 0 )
        pts_local_valid = pts_local[depth_mask].numpy()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_local_valid))
        pc.estimate_normals()
        return pc


    # @brief: rectify local pose of overlapping keyframe which triggers active submap switch;
    # @param pose_local_this: local pose in localMLP after switch, Tensor(4, 4);
    # @param pose_local_bf: local pose in localMLP before switch, Tensor(4, 4);
    # @param localMLP_Id_aft: active localMLP_Id (after switch), Tensor(, );
    # @param localMLP_Id_prev: active localMLP_Id (before switch), Tensor(, );
    # @param nearest_kf_Ids
    # @param nearest_kf_mask
    #-@return return_flag: whether the pose can be rectified, bool;
    #-@return pose_local_final: rectified local pose, Tensor(4, 4).
    def switch_pose_rectifying(self, batch, pose_local_this, pose_local_bf, localMLP_Id_aft, localMLP_Id_prev, nearest_kf_Ids, nearest_kf_mask):
        frame_Id = batch["frame_id"]
        kf_Id = frame_Id // self.config["mapping"]["keyframe_every"]
        collect_kf_num = self.kfSet.collected_kf_num[0]
        first_kf_pose, first_kf_Id = self.kfSet.extract_first_kf_pose(localMLP_Id_aft, self.kf_c2w)  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )

        # Step 1: convert local pose in prev_active_localMLP to local pose in current active_localMLP
        # 1.1: find first keyframe of new active localMLP
        first_kf_pose_pev = extract_first_kf_pose(localMLP_Id_prev, self.kfSet.localMLP_first_kf, self.kf_c2w.clone())  # Tensor(4, 4)
        first_kf_pose_aft = extract_first_kf_pose(localMLP_Id_aft, self.kfSet.localMLP_first_kf, self.kf_c2w.clone())  # Tensor(4, 4)

        # 1.2: convert to local pose in localMLP which will be switched to
        pose_world = first_kf_pose_pev @ self.est_c2w_data[frame_Id]  # world pose of this keyframe
        cur_pose_local = first_kf_pose_aft.inverse() @ pose_world

        # Step 2: construct pointcloud from selected points in nearest keyframes
        # 2.1: select keyframes and pts
        nearest_kf_Ids = nearest_kf_Ids.clone()  # Tensor(k, )
        kf_pts_mask = nearest_kf_mask.clone()  # Tensor(k, pix_num)
        kf_valid_pts_num = torch.count_nonzero(kf_pts_mask, -1)  # Tensor(k, )
        valid_kf_indices = torch.where(kf_valid_pts_num > 200)[0]
        if valid_kf_indices.shape[0] == 0:
            selected_kf_Ids = nearest_kf_Ids  # Tensor(k', )
        else:
            selected_kf_Ids = nearest_kf_Ids[valid_kf_indices]  # Tensor(k', )
        selected_frame_Ids = selected_kf_Ids * self.config["mapping"]["keyframe_every"]  # frame_Ids of all selected keyframes, Tensor(k', )

        # 2.2: construct sparse pointcloud from selected valid keyframes
        pose_local = self.est_c2w_data[selected_frame_Ids]
        selected_kf_localMLP = self.kfSet.keyframe_localMLP[selected_kf_Ids]  # keyframe-localMLP relationships of selected keyframes, Tensor(k', )
        localMLP_hit_dix = self.kfSet.get_related_localMLP_index(selected_kf_localMLP, localMLP_Id_aft)  # Tensor(k', )
        pose_local_given = self.kfSet.convert_given_local_pose(selected_kf_localMLP, localMLP_hit_dix, self.kf_c2w[:collect_kf_num, ...], first_kf_pose, pose_local)  # local pose in given localMLP's Coord system, Tensor(k', )
        pc_seletced_kf = self.construct_pc_given_kfs(selected_kf_Ids, pose_local_given.cpu())

        # 2.3: construct sparse pointcloud from this keyframe and last n keyframes
        pc_this_kf = self.construct_pc(batch, pose_local_this.cpu())

        if self.config["tracking"]["switch"]["including_last"] > 0:
            inclued_kf = [ kf_Id-i for i in range(1, self.config["tracking"]["switch"]["including_last"]+1) ]
            selected_kf_Ids_this = torch.tensor(inclued_kf, dtype=torch.int64)
            selected_frame_Ids_this = selected_kf_Ids_this * self.config["mapping"]["keyframe_every"]  # frame_Ids of all selected keyframes, Tensor(k', )
            pose_local = self.est_c2w_data[selected_frame_Ids_this]
            selected_kf_localMLP_this = self.kfSet.keyframe_localMLP[selected_kf_Ids_this]
            localMLP_hit_dix_this = self.kfSet.get_related_localMLP_index(selected_kf_localMLP_this, localMLP_Id_prev)
            pose_local_given_bf = self.kfSet.convert_given_local_pose(selected_kf_localMLP_this, localMLP_hit_dix_this, self.kf_c2w[:collect_kf_num, ...], first_kf_pose_pev, pose_local)
            pose_local_given_aft = first_kf_pose_aft.inverse() @ first_kf_pose_pev @ pose_local_given_bf
            pc_this_kf1 = self.construct_pc_given_kfs(selected_kf_Ids_this, pose_local_given_aft.cpu())
            pc_this_kf = self.merge_pc(pc_this_kf1, pc_this_kf)

        # Step 3: compute relative pose and align
        threshold = self.config["tracking"]["switch"]["align_threshold"]
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(pc_this_kf, pc_seletced_kf, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())  # point-to-plane
        rel_pose = torch.from_numpy(reg_p2p.transformation.astype(np.float32)).to(self.device)
        correspondences = np.asarray(reg_p2p.correspondence_set)  # ndarray(n, 2)

        if correspondences.shape[0] >= self.config["tracking"]["switch"]["min_correspondence"]:
            if torch.norm(rel_pose[:3, 3]) >= self.config["tracking"]["switch"]["min_trans_dist"]:
                rel_pose = torch.eye(4).to(rel_pose)
            pose_local_final = rel_pose @ pose_local_this
            return_flag = True
        else:
            pose_local_final = pose_local_this
            return_flag = False
        return return_flag, correspondences.shape[0], pose_local_final


    # @brief: jointly adjust first keyframes' poses of each localMLP;
    # @param kf_num:
    # @param adja_pairs:
    # @param local_pose_prev: pose of overlapping keyframe in previous active localMLP's coordinate system, Tensor(4, 4);
    # @param local_pose_after: pose of overlapping keyframe in current active localMLP's coordinate system, Tensor(4, 4);
    # @param localMLP_Id_prev: localMLP_Id of previous localMLP (before loop), Tensor(, );
    # @param localMLP_Id_aft: localMLP_Id of current localMLP (after loop), Tensor(, ).
    def pose_graph_optimize(self, kf_num, adja_pairs, local_pose_prev, local_pose_after, localMLP_Id_prev, localMLP_Id_aft):
        # Step 1: construct pose graph
        edges = []  # related nodes of this edge (constrain)
        poses = []  # observed relative pose of these two nodes

        # 1.1: update nodes of pose graph
        localMLP_first_kf = self.kfSet.localMLP_first_kf.detach()
        keyframe_ref = self.slam.keyframe_ref[:kf_num].detach()
        first_kf_Ids = torch.where(keyframe_ref == -1)[0]  # keyframe_Id of each localMLP's first keyframe
        first_kf_pose = self.slam.kf_c2w[first_kf_Ids].detach()

        self.update_pose_graph(first_kf_Ids)

        # 1.2: add edges for each adjacent localMLPs pair
        for adja_pair in adja_pairs:
            localMLP_Id1 = adja_pair[0]
            localMLP_Id2 = adja_pair[1]
            edges.append(adja_pair)

            # TODO: get observed relative pose by finding correspondences + ICP
            first_kf_pose1 = first_kf_pose[localMLP_Id1]
            first_kf_pose2 = first_kf_pose[localMLP_Id2]
            pose_21 = first_kf_pose2.inverse() @ first_kf_pose1  # pose convert coords in first localMLP to coords in second localMLP
            poses.append(pose_21)

        # 1.3: add edge for the key overlapping keyframe
        edges.append( torch.stack([localMLP_Id_aft, localMLP_Id_prev], 0) )
        rel_observed_pose = local_pose_prev @ local_pose_after.inverse()
        poses.append(rel_observed_pose)

        # edges = torch.stack(edges).to(self.device)
        # poses = torch.stack(poses).to(self.device)
        edges = torch.stack(edges).cpu()
        poses = torch.stack(poses).cpu()

        # Step 2: do optimization and get results
        solver = ppos.Cholesky()
        strategy = ppost.TrustRegion(radius=1e4)
        optimizer = pp.optim.LM(self.poseGraph, solver=solver, strategy=strategy, min=1e-6, vectorize=False)
        scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

        scheduler.optimize(input=(edges, poses, self.config["mapping"]["global_BA"]["key_edge_weight"]))
        optimized_first_kf_pose = self.poseGraph.get_pose_mat()
        self.slam.kf_c2w[first_kf_Ids] = optimized_first_kf_pose.to(self.device)

