import numpy as np
import torch
import pytorch3d.transforms as transforms
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import torch.nn.functional as F


# @param rot: Tensor/Parameter(n, 4);
# @param trans: Tensor/Parameter(n, 3);
#-@return: Tensor(b, 4, 4).
def qt_to_transform_matrix(rot, trans):
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = quaternion_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T


# @param quat: Tensor(4, )
def get_unit_quaternion(quat):
    quat_uni = F.normalize(quat, p=2, dim=0)
    # quat_uni = quat / (quat.norm() + 1e-5)
    quat_uni = transforms.standardize_quaternion(quat_uni)
    return quat_uni


# @brief: convert to 7D poses to 4X4 matrices;
# @param quats: batch of [qw, qx, qy, qz], Tensor(n, 4);
# @param trans: batch of [rx, ry, rz], Tensor(n, 3);
#-@return: Tensor(n, 4, 4).
def pose_7d_to_mat(quats, trans):
    rot_mats = transforms.quaternion_to_matrix(quats)  # Tensor(n, 3, 3)
    trans_mats = torch.eye(4)[None, ...].to(quats).repeat( (quats.shape[0], 1, 1) )
    trans_mats[:, :3, :3] = rot_mats
    trans_mats[:, :3, 3] = trans
    return trans_mats


# @param rot_mat: Tensor(3, 3);
# @param trans_vec: Tensor(3, 1);
#-@return: Tensor(n, 4, 4).
def pose_compose(rot_mat, trans_vec):
    trans_mat = torch.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = trans_vec.squeeze()
    return trans_mat


# @brief: inverse a transformation matrix;
# @param trans_mat: Tensor(4, 4);
#-@return: Tensor(4, 4)
def trans_mat_inv(trans_mat):
    R_T = trans_mat[:3, :3].T
    inv_mat = torch.eye(4).to(trans_mat)
    inv_mat[:3, :3] = R_T
    inv_mat[:3, 3] = -1. * R_T @ trans_mat[:3, 3]
    return inv_mat


# @brief: inverse a transformation matrix;
# @param trans_mat: Tensor(n, 4, 4);
#-@return: Tensor(n, 4, 4)
def trans_mats_inv(trans_mats):
    R_T = torch.transpose(trans_mats[:, :3, :3], 1, 2)  # Tensor(n, 3, 3)
    inv_mats = torch.eye(4).to(trans_mats).repeat((trans_mats.shape[0], 1, 1))
    inv_mats[:, :3, :3] = R_T
    inv_mats[:, :3, 3:] = -1. * R_T @ trans_mats[:, :3, 3:]
    return inv_mats


# @brief: apply rotation and translation to a batch of 3D points;
# @param pts: Tensor(m, 3);
# @param pose_rot:Tensor(4, 4);
#-@return: Tensor(m, 3);
def transform_points(pts, mats):
    pose_rot = mats[:3, :3]  # Tensor(3, 3);
    pose_trans = mats[:3, 3:]  # Tensor(3, 1)
    transed_pts = pose_rot @ torch.transpose(pts, 0, 1)  # EagerTensor(3, m)
    transed_pts = transed_pts + pose_trans  # EagerTensor(3, m)
    transed_pts = torch.transpose(transed_pts, 0, 1)  # EagerTensor(N, m, 3)
    return transed_pts


def convert_to_local_pts(wrld_pts, first_kf_pose):
    pose_inv = trans_mat_inv(first_kf_pose)
    rot_inv, trans_inv = pose_inv[:3, :3].to(wrld_pts), pose_inv[:3, 3].to(wrld_pts)
    transd_pts = rot_inv @ wrld_pts.T + trans_inv[..., None]
    transd_pts = transd_pts.T
    return transd_pts


def convert_to_local_pts2(wrld_pts, first_kf_pose):
    pose_w2c = torch.inverse(first_kf_pose)
    rot_w2c = pose_w2c[:3, :3]  # rot mat w2c, Tensor(3, 3)
    trans_w2c = pose_w2c[:3, 3]  # trans vec w2c, Tensor(3)
    rotated_pts = torch.sum(wrld_pts[:, None, :] * rot_w2c[None, :, :], -1)  # Tensor(pixel_num, 3)
    transed_pts = rotated_pts + trans_w2c[None, :]  # Tensor(k, pixel_num, 3)
    return transed_pts


# @brief: convert rays from Camera to World (using same pose);
# @param rays_d_cam: ray directions of selection pixels in Camera Coordinate System, Tensor(N, 3);
# @param c2w_mat: Tensor(4, 4);
#-@return rays_d: ray directions of selection pixels in World Coordinate System, Tensor(N, 3);
#-@return rays_o: ray origins of selection pixels in World Coordinate System, Tensor(N, 3).
def rays_camera_to_world(rays_d_cam, c2w_mat):
    rays_num = rays_d_cam.shape[0]
    rays_o = c2w_mat[None, :3, -1].repeat(rays_num, 1)  # Tensor(N, 3)
    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_mat[:3, :3], -1)  # Tensor(N, 3)
    return rays_d, rays_o


# @brief: convert rays from Camera to World (using different poses);
# @param rays_d_cam: ray directions of selection pixels in Camera Coordinate System, Tensor(N, 3);
# @param c2w_mats: M different poses, Tensor(M, 4, 4);
# @param pose_indices: indices of each used pose, Tensor(N, );
#-@return rays_d: ray directions of selection pixels in World Coordinate System, Tensor(N, 3);
#-@return rays_o: ray origins of selection pixels in World Coordinate System, Tensor(N, 3).
def rays_camera_to_world2(rays_d_cam, c2w_mats, pose_indices):
    rays_o = c2w_mats[pose_indices, :3, -1]  # Tensor(N, 3)
    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_mats[pose_indices, :3, :3], -1)  # Tensor(N, 3)
    return rays_d, rays_o


# @brief: giving a keyframe's depth image and its pose in World Coordinate System, compute the surface bounding box;
# @param frame_pose: pose of given frame (c2w, in World Coordinate System), Tensor(4, 4);
# @param frame_depth: depth image of the frame, Tensor(H, W);
# @param rays_d: ray directions of all pixels in Camera Coordinate System, Tensor(H, W, 3);
#-@return xyz_center: Tensor(3, );
#-@return xyz_len: Tensor(3, ).
@torch.no_grad()
def get_frame_surface_bbox(frame_pose, frame_depth, rays_d, dist_near, dist_far):
    rays_d_w, rays_o_w = rays_camera_to_world(rays_d.reshape((-1, 3)), frame_pose)  # Tensor(N, 3) / Tensor(N, 3)

    # filter the pixels invalid depth values
    frame_depth = frame_depth.reshape((-1, 1))  # Tensor(N, 1)
    valid_mask = (frame_depth.squeeze() > dist_near) * (frame_depth.squeeze() < dist_far)  # Tensor(N, ), dtype=torch.bool

    pts_world = rays_o_w + rays_d_w * frame_depth
    valid_pts_world = pts_world[valid_mask]  # Tensor(N', 3)

    xyz_max, _ = torch.max(valid_pts_world, 0)
    xyz_min, _ = torch.min(valid_pts_world, 0)
    xyz_len = xyz_max - xyz_min
    xyz_center = xyz_min + 0.5 * xyz_len
    return xyz_center, xyz_len


# @brief: get first keyframe's pose (in World Coordinate System) of given localMLP;
# @param kf_localMLP_Ids: corresponding localMLP_Id of each keyframe, Tensor(m, )/Tensor(, );
# @param localMLP_first_kf: Tensor(m, );
# @param kf_poses: Tensor(n, 4, 4);
#-@return: Tensor(m, 4, 4)/Tensor(4, 4).
def extract_first_kf_pose(kf_localMLP_Ids, localMLP_first_kf, kf_poses):
    first_kf_Ids = localMLP_first_kf[kf_localMLP_Ids]  # first keyframe's kf_Id of each localMLP, Tensor(m, )/Tensor(, )
    first_kf_pose = kf_poses[first_kf_Ids]  # first keyframe's pose(c2w, in World Coordinate System) of each localMLP, Tensor(m, 4, 4)/Tensor(4, 4)
    return first_kf_pose


# @param keyframe_localMLP: Tensor(kf_num, 2)
# @param localMLP_Id1: Tensor(, );
# @param localMLP_Id2: Tensor(, );
#-@return: Tensor(kf_num, ).
def find_related_localMLPs(keyframe_localMLP, localMLP_Id1, localMLP_Id2):
    perm1 = torch.stack([localMLP_Id1, localMLP_Id2], 0)
    perm2 = torch.stack([localMLP_Id2, localMLP_Id1], 0)
    mask1 = torch.where(keyframe_localMLP == perm1, torch.ones_like(keyframe_localMLP[:, 0:1]), torch.zeros_like(keyframe_localMLP[:, 0:1])).all(-1)  # Tensor(kf_num, )
    mask2 = torch.where(keyframe_localMLP == perm2, torch.ones_like(keyframe_localMLP[:, 0:1]), torch.zeros_like(keyframe_localMLP[:, 0:1])).all(-1)  # Tensor(kf_num, )
    mask = torch.logical_or(mask1, mask2)
    related_kf_Ids = torch.where(mask)[0]
    return related_kf_Ids


# @param kf_poses: Tensor(n, 4, 4)
# @param kf_ref: Tensor(n, )
# @param localMLP_first_kf: Tensor(m, )
# @param kf_localMLP_Ids: corresponding localMLP_Id of each keyframe, Tensor(n, )
@torch.no_grad()
def get_local_kf_pose(kf_poses, kf_ref, localMLP_first_kf, kf_localMLP_Ids):
    first_kf_Ids = localMLP_first_kf[kf_localMLP_Ids]  # first keyframe's kf_Id of each localMLP, Tensor(n, )
    first_kf_pose = kf_poses[first_kf_Ids]  # first keyframe's pose(c2w, in World Coordinate System) of each localMLP, Tensor(n, 4, 4)
    first_kf_pose_w2c = trans_mats_inv(first_kf_pose)  # Tensor(n, 4, 4)
    kf_pose_local = torch.where(kf_ref[..., None, None] >= 0, kf_poses, first_kf_pose_w2c @ kf_poses)
    return kf_pose_local


# @brief: give n points and m localMLPs, judge the containing relationships between them;
# @param pts: Tensor(n, 3);
# @param xyz_min: Tensor(m, 3);
# @param xyz_max: Tensor(m, 3);
#-@return: Tensor(n, m)
def pts_in_bbox(pts, xyz_min, xyz_max):
    mask = []
    for i in range(xyz_min.shape[0]):
        condition1 = (pts > xyz_min[i]).all(dim=-1 )  # Tensor(n, )
        condition2 = (pts < xyz_max[i]).all(dim=-1 )  # Tensor(n, )
        condition = torch.logical_and(condition1, condition2)  # Tensor(n, )
        mask.append(condition)
    mask_tensor = torch.stack(mask, dim=-1)  # Tensor(n, m)
    return mask_tensor


# @brief: giving m keyframes' surface information, compute their centers;
# @param rays: Tensor(m, n_rays, 7), [direction, RGB, depth];
#-@return: mean point of each given keyframe's surface.
def compute_surface_center(rays):
    pts = rays[:, :, :3] * rays[:, :, -1:]  # get each surface point, Tensor(m, n_rays, 3)
    mean_pts = pts.mean(dim=1, dtype=torch.float32)  # Tensor(m, 3)
    return mean_pts


# @param K: Tensor(3, 3);
# @para pts: Tensor(n, 3, 1);
#-@return: pixel coordinates, Tensor(n, 2).
def project_to_pixel(K, pts):
    pts[:, 0] *= -1
    uv = (K @ pts).squeeze(-1)
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float()
    return uv


def compute_avg_SDF_difference(pred_SDF1, pred_SDF2, mask):
    loss = torch.sum( torch.square(pred_SDF1 * mask - pred_SDF2 * mask) )
    valid_pixel_num = torch.count_nonzero(mask) + 0.001
    loss_avg = loss / valid_pixel_num
    return loss_avg


def compute_avg_RGB_difference(pred_RGB1, pred_RGB2, mask):
    losses_rgb = torch.where(mask.squeeze(-1) > 0., torch.sum(torch.abs(pred_RGB1 - pred_RGB2), dim=1), 0.)  # each valid sampled pixel's Photometric Loss ( Sec 3.4, 式(3) ), EagerTensor(n, )
    valid_pixel_num = torch.count_nonzero(mask) + 0.001
    loss_rgb_avg = torch.sum( torch.square(losses_rgb) ) / valid_pixel_num  # valid sampled pixel的 rgb loss
    return loss_rgb_avg
