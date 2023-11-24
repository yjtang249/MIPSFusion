import os
import time
import torch
from pytorch3d.transforms import matrix_to_quaternion
import numpy as np
import matplotlib.pyplot as plt

from model.Mesher import Mesher
from utils.utils import extract_mesh, extract_mesh2
from helper_functions.geometry_helper import extract_first_kf_pose, transform_points, rays_camera_to_world
from datasets.utils import get_camera_rays


# @brief: class for checkpoint/trajectory saving, recovering and mesh extracting
class Logger():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = self.slam.device
        self.dataset = self.slam.dataset
        self.kfSet = self.slam.kfSet
        self.bounding_box = self.slam.bounding_box.clone()
        self.marching_cube_bound = self.slam.marching_cube_bound.clone()
        self.kf_ref = self.slam.keyframe_ref  # Tensor(num_kf, ), dtype=torch.int
        self.kf_c2w = self.slam.kf_c2w  # Tensor(num_kf, 4, 4)
        self.est_c2w_data = self.slam.est_c2w_data  # Tensor(num_frame, 4, 4)
        self.est_c2w_data_rel = self.slam.est_c2w_data_rel  # Tensor(num_frame, 4, 4)

        self.mesher = Mesher(config, SLAM)
        self.rays_d = get_camera_rays(self.dataset.H, self.dataset.W, self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy).to(self.device)  # dir vector of each pixel (in Camera Frame), Tensor(H, W, 3)


    def save_state_dict(self, model, save_path):
        torch.save(model.state_dict(), save_path)


    def load_state_dict(self, model, load_path):
        model.load_state_dict(torch.load(load_path))


    # @brief: save state tensors
    def save_ckpt(self, frame_id, save_path):
        tensor_dict = {
            # pose-related tensors
            "kf_c2w": self.slam.kf_c2w,
            "est_c2w_data": self.slam.est_c2w_data,
            "est_c2w_data_rel": self.slam.est_c2w_data_rel,

            # keyframe-related tensors
            "keyframe_ref": self.slam.keyframe_ref,
            "keyframe_localMLP": self.kfSet.keyframe_localMLP,
            "keyframe_mutex_mask": self.kfSet.keyframe_mutex_mask,

            # localMLP-related tensors
            "localMLP_info": self.kfSet.localMLP_info,
            "localMLP_first_kf": self.kfSet.localMLP_first_kf,
            "localMLP_max_len": self.kfSet.localMLP_max_len,
            "localMLP_adjacent": self.kfSet.localMLP_adjacent,

            # active localMLP stats
            "active_localMLP_Id": self.slam.active_localMLP_Id,
            "prev_active_localMLP_Id": self.slam.prev_active_localMLP_Id,
            "active_first_kf": self.slam.active_first_kf,
            "last_switch_frame": self.slam.last_switch_frame,

            "do_globalBA": self.slam.do_globalBA
        }
        torch.save(tensor_dict, save_path)
        print("Checkpoint for frame_%d was saved." % frame_id)


    # @brief: Load the model parameters and the estimated pose
    def load_ckpt(self, load_path):
        dict = torch.load(load_path)
        # model.load_state_dict(dict['model'])
        # self.est_c2w_data = dict['pose']
        # self.est_c2w_data_rel = dict['pose_rel']
        return dict


    # @brief: get absolute pose in corresponding Local Coordinate System of each frame;
    # @param idx: latest frame_Id of all collected frames;
    #-@return: Tensor(idx+1, 4, 4).
    def convert_relative_pose(self, idx):
        poses = []
        for i in range(idx + 1):
            if i % self.config['mapping']['keyframe_every'] == 0:  # case 1: for keyframes
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_ref = self.kf_ref[kf_id]
                if kf_ref >= 0:  # case 1.1: for ordinary keyframes
                    kf_pose_local = self.est_c2w_data[i]
                elif kf_ref == -1:  # case 1.2: for first keyframes
                    kf_pose_local = torch.eye(4).to(self.slam.device)
                else:  # case 1.3: overlapping keyframes
                    kf_pose_local = self.est_c2w_data[i]
                poses.append(kf_pose_local)
            else:  # case 2: for non-keyframes
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                # c2w_key = poses[kf_frame_id]  # pose of ref keyframe (in Local Coordinate System)
                c2w_key = self.est_c2w_data[kf_frame_id]  # pose of ref keyframe (in Local Coordinate System)
                delta = self.est_c2w_data_rel[i]  # relative pose of current frame w.r.t. ref keyframe
                # poses.append(delta @ c2w_key)  # absolute pose of current keyframe
                poses.append(c2w_key @ delta)  # absolute pose of current keyframe
        poses = torch.stack(poses, dim=0)
        return poses


    # @brief: convert all absolute poses from Local Coordinate System to World Coordinate System;
    # @param poses_local: poses in Local Coordinate System, Tensor(frame_num, 4, 4);
    #-@return: Tensor(frame_num, 4, 4).
    def convert_world_pose(self, poses_local):
        idx = len(poses_local)  # frame_num

        # Step 1: find ref keyframe of each frame
        ref_frame_kfIds = torch.div(torch.arange(idx), self.config["mapping"]["keyframe_every"], rounding_mode="floor")  # Tensor(frame_num, )
        # kf_localMLP_Ids = self.kfSet.get_kf_localMLP_Id()  # corresponding localMLP_Id of each keyframe, Tensor(keyframe_num, )
        kf_localMLP_Ids = self.kfSet.keyframe_localMLP[:, 0]  # corresponding localMLP_Id of each keyframe, Tensor(keyframe_num, )

        # Step 2: get first keyframe's pose (in World Coordinate System) of each localMLP which each frame belongs to
        kf_localMLP_first_kf, _ = self.kfSet.extract_first_kf_pose(kf_localMLP_Ids, self.slam.kf_c2w)
        traj_localMLP_first_kf = kf_localMLP_first_kf[ref_frame_kfIds]

        # Step 3: get World pose of each frame
        poses_world = (traj_localMLP_first_kf @ poses_local).detach()
        return poses_world


    # @param pose_world: Tensor(n, 4, 4);
    # @param output_file: str
    def save_traj_tum(self, pose_world, output_file):
        frame_num = pose_world.shape[0]
        pose_trans = pose_world[:, :3, 3].cpu()  # Tensor(n, 3)
        pose_quat_r = matrix_to_quaternion(pose_world[:, :3, :3]).cpu()  # quaternions with real parts first, Tensor(n, 4)

        # convert quaternions with real parts first to quaternions with real parts last
        pose_quat = torch.zeros_like(pose_quat_r)
        pose_quat[:, :3] = pose_quat_r[:, 1:4]
        pose_quat[:, -1] = pose_quat_r[:, 0]  # Tensor(n, 4)

        with open(output_file, "w") as f_out:
            for i in range(frame_num):
                f_out.write("%.4f " % i)  # timestamp
                f_out.write(' '.join( pose_trans[i].numpy().astype('str').tolist() ) + ' ')  # translation vector
                f_out.write(' '.join( pose_quat[i].numpy().astype('str').tolist() ) + '\n')  # quaternion


    def extract_a_mesh(self, frame_id, localMLP_Id, model):
        mesh_save_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "%d" % frame_id)
        if not os.path.exists(mesh_save_dir):
            os.makedirs(mesh_save_dir)
        mesh_save_path = os.path.join(mesh_save_dir, "localMLP_%d.ply" % int(localMLP_Id))
        self.mesher.extract_single_mesh(model, localMLP_Id, save_path=mesh_save_path)


    def extract_a_mesh_offline(self, localMLP_Id, model, kf_num, mesh_save_path):
        return self.mesher.extract_single_mesh(model, localMLP_Id, kf_num=kf_num, save_path=mesh_save_path)


    # @brief: jointly rendering (large amount of GPU memory is required).
    def extract_mesh_jointly(self, model_list, bounding_geo_list, submesh_list, using_obbox, save_path=None):
        self.mesher.extract_mesh_jointly(model_list, bounding_geo_list, submesh_list, using_obbox, save_path=save_path)
        if save_path is not None:
            print("Whole mesh was saved to: %s" % save_path)


    def extract_mesh_jointly_simple(self, submesh_list, save_path=None):
        concat_mesh = self.mesher.trimesh_concat(submesh_list)
        if save_path is not None:
            concat_mesh.export(save_path)
            print("Whole mesh was saved to: %s" % save_path)


    def extract_all_mesh(self, frame_id, model_list, first_kf_poses, voxel_size=0.05):
        mesh_save_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "%d" % frame_id)
        if not os.path.exists(mesh_save_dir):
            os.makedirs(mesh_save_dir)
        model_num = len(model_list)

        for i in range(model_num):
            model = model_list[i]
            mesh_save_path = os.path.join(mesh_save_dir, "localMLP_%d.ply" % i)
            first_kf_c2w = first_kf_poses[i]
            extract_mesh2(model.query_sdf,
                          first_kf_c2w,
                          self.config,
                          self.bounding_box,
                          color_func=model.query_color,
                          marching_cube_bound=self.marching_cube_bound,
                          voxel_size=voxel_size,
                          mesh_savepath=mesh_save_path)


    # @brief: render a full image, getting its RGB map and depth map (for each pixel)
    def render_full_img(self, model, pose_local, gt_depth, ray_batch_size=10000):
        with torch.no_grad():
            gt_depth = torch.reshape(gt_depth, (-1, 1))
            rays_d, rays_o = rays_camera_to_world(torch.reshape(self.rays_d, (-1, 3)), pose_local)  # Camera coordinate system --> Local coordinate system

            rgb_list = []
            depth_list = []
            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size].to(self.device)
                rays_o_batch = rays_o[i:i + ray_batch_size].to(self.device)
                gt_depth_batch = gt_depth[i:i + ray_batch_size].to(self.device)

                render_dict = model.render_rays(rays_o_batch, rays_d_batch, gt_depth_batch)
                rgb_list.append(render_dict["rgb"])
                depth_list.append(render_dict["depth"])

            rgb = torch.cat(rgb_list, 0)
            depth = torch.cat(depth_list, 0)
            rgb = rgb.reshape((self.dataset.H, self.dataset.W, 3))
            depth = depth.reshape((self.dataset.H, self.dataset.W))
            torch.cuda.empty_cache()
            return rgb, depth


    # @brief: render a keyframe using given localMLP and given local pose;
    # @param pose_local: Tensor(4, 4);
    # @param gt_color: Tensor(H, W, 3);
    # @param gt_depth: Tensor(H, W).
    def img_render_save(self, model, pose_local, gt_color, gt_depth, i):
        save_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "keyframe")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "frame_%d.png" % i)

        with torch.no_grad():
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            valid_mask = (gt_depth.squeeze() > self.config["cam"]["near"]) * (gt_depth.squeeze() < self.config["cam"]["far"])  # Tensor(H, W), dtype=torch.bool

            rendered_color, rendered_depth = self.render_full_img(model, pose_local, gt_depth)  # Tensor(H, W, 3) / Tensor(H, W)
            rendered_color = rendered_color.cpu().numpy()
            rendered_depth = rendered_depth.cpu().numpy()

            loss_rgb = torch.mean( torch.abs(gt_color[valid_mask] - rendered_color[valid_mask]) )
            loss_depth = torch.mean( torch.abs(gt_depth[valid_mask] - rendered_depth[valid_mask]) )
            title_text = "RGB_loss = %.4f; depth_loss=%.4f" % (float(loss_rgb.numpy()), float(loss_depth.numpy()))

            fig, axs = plt.subplots(2, 2, figsize=(10, 9))
            fig.tight_layout()
            max_depth = np.max(gt_depth_np)
            fig.suptitle(title_text)
            axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(rendered_depth, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Generated Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[1, 0].imshow(gt_color_np, cmap="plasma")
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(rendered_color, cmap="plasma")
            axs[1, 1].set_title('Generated RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
            plt.clf()


    # @brief: save checkpoint until frame_id, called by ActiveMap process
    def save_ckpt_active(self, frame_id, model, active_localMLP_Id, final=False):
        if final == False:
            ckpt_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "ckpt_%d" % frame_id)
        else:
            ckpt_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "ckpt_final")

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        model_weights_path = os.path.join(ckpt_dir, "model_%d.pth" % int(active_localMLP_Id.cpu().numpy()))
        self.save_state_dict(model, model_weights_path)


    # @brief: save checkpoint until frame_id, called by InactiveMap process
    def save_ckpt_inactive(self, frame_id, model_list, active_localMLP_Id):
        ckpt_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "ckpt_%d" % frame_id)
        time.sleep(0.5)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # Step 1: save each inactive localMLP's parameters
        for i in range(len(model_list)):
            if i == active_localMLP_Id:
                continue
            model_weights_path = os.path.join(ckpt_dir, "model_%d.pth" % i)
            self.save_state_dict(model_list[i], model_weights_path)

        # Step 2: save other tensors
        ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
        self.save_ckpt(frame_id, ckpt_path)

