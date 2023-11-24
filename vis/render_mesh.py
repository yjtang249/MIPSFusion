import os
from utils import config
import torch
from torch.utils.data import DataLoader
import argparse

from mipsfusion import MIPSFusion
from model.scene_rep import JointEncoding


def fill_members(slam, kfSet, tensor_dict):
    slam.kf_c2w = tensor_dict["kf_c2w"]
    slam.est_c2w_data = tensor_dict["est_c2w_data"]
    slam.est_c2w_data_rel = tensor_dict["est_c2w_data_rel"]
    
    slam.keyframe_ref = tensor_dict["keyframe_ref"]
    kfSet.keyframe_localMLP = tensor_dict["keyframe_localMLP"]
    kfSet.keyframe_mutex_mask = tensor_dict["keyframe_mutex_mask"]

    kfSet.localMLP_info = tensor_dict["localMLP_info"]
    kfSet.localMLP_first_kf = tensor_dict["localMLP_first_kf"]
    kfSet.localMLP_max_len = tensor_dict["localMLP_max_len"]
    kfSet.localMLP_adjacent = tensor_dict["localMLP_adjacent"]

    slam.active_localMLP_Id = tensor_dict["active_localMLP_Id"]
    slam.prev_active_localMLP_Id = tensor_dict["prev_active_localMLP_Id"]
    slam.active_first_kf = tensor_dict["active_first_kf"]
    slam.last_switch_frame = tensor_dict["last_switch_frame"]


def fill_rays(slam, kfSet):
    for i in slam.dataset.frame_ids:
        if i == 0 or i % slam.config['mapping']['keyframe_every'] == 0:
            batch = slam.dataset.__getitem__(i)
            kfSet.add_keyframe(batch, filter_depth=slam.config['mapping']['filter_depth'])
    print("Added kf_rays.")


if __name__ == '__main__':
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--seq_result', type=str, help='Output path of running this sequence.')
    parser.add_argument('--ckpt', type=str, help='Path of selected checkpoint.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    result_path = os.path.join(args.seq_result, "result")  # dir to save result of this running
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    slam = MIPSFusion(cfg)
    kfSet = slam.kfSet
    logger = slam.logger
    mesher = logger.mesher
    fill_rays(slam, kfSet)

    # Step 1: create models and load params
    ckpt_dir = os.path.join(args.seq_result, "ckpt_"+args.ckpt)
    files = os.listdir(ckpt_dir)
    model_paths = []
    for file in files:
        if "model" in file:
            model_paths.append(os.path.join(ckpt_dir, file))
    model_paths.sort(key=lambda x: int(os.path.basename(x)[6:-4]))

    model_list = []
    for model_path in model_paths:
        model = JointEncoding(slam.config, slam.bounding_box, slam.coords_norm_factor).to(slam.device)
        logger.load_state_dict(model, model_path)
        model_list.append(model)


    # Step 2: load ckpt
    dict_tensors = logger.load_ckpt(os.path.join(ckpt_dir, "ckpt.pt"))
    fill_members(slam, kfSet, dict_tensors)
    kf_num = torch.where(slam.keyframe_ref != -3)[0].shape[0]

    # Step 3: render each mesh
    bound_geo_list = []  # list of open3d.geometry.VoxelGrid / open3d.geometry.OrientedBoundingBox obj;
    submesh_list = []  # list trimesh.Trimesh.
    for i in range(len(model_list)):
        mesh_save_path = os.path.join(result_path, "%d.ply" % i)
        bounding_geometry, submesh, using_obbox = logger.extract_a_mesh_offline(i, model_list[i], kf_num, mesh_save_path)
        bound_geo_list.append(bounding_geometry)
        submesh_list.append(submesh)

    # Step 4: jointly render to get final mesh
    mesh_save_path = os.path.join(result_path, "final_mesh.ply")

    if cfg["mesh"]["simply_joint"]:
        logger.extract_mesh_jointly_simple(submesh_list, save_path=mesh_save_path)
    else:
        logger.extract_mesh_jointly(model_list, bound_geo_list, submesh_list, using_obbox, save_path=mesh_save_path)

