import torch
import torch.nn as nn
import pypose as pp

from external.Pypose_external.convert import mat2SE3


class PoseGraph(nn.Module):
    # @param first_kf_pose: Tensor(n, 4, 4)
    def __init__(self, first_kf_pose, device):
        super().__init__()
        self.device = "cpu"
        # self.device = device
        self.poses_SE3 = mat2SE3(first_kf_pose).to(self.device)
        self.nodes = pp.Parameter(self.poses_SE3).to(self.device)

    def update_param(self, first_kf_pose):
        self.poses_SE3 = mat2SE3(first_kf_pose).to(self.device)
        self.nodes = pp.Parameter(self.poses_SE3).to(self.device)

    def get_pose_mat(self):
        return self.nodes.matrix().detach()


    # @param edges: each edge indicates 2 related nodes(localMLP_Id), Tensor(n, 2);
    # @param poses: each pose represents from first local coords to second local coords, Tensor(n, 4, 4);
    #-@return: Tensor(n, 6).
    def forward(self, edges, poses, key_edge_weight=0.1):
        nodes_all = torch.cat([self.poses_SE3[0:1], self.nodes[1:]])  # first keyframe's pose will never be updated

        # Step 1: compute error for all other edges
        node1 = nodes_all[edges[:-1, 0]]  # LiTensor(n-1, 7)
        node2 = nodes_all[edges[:-1, 1]]  # LiTensor(n-1, 7)
        poses_SE3 = mat2SE3(poses[:-1, ...])  # LiTensor(n-1, 7)
        error = poses_SE3 @ (node1.Inv() @ node2)  # LiTensor(n-1, 7)
        error1 = error.Log().tensor()

        # Step 2: compute error for the key overlapping keyframe (key edge)
        node1_key = nodes_all[edges[-1, 0]]  # LiTensor(7, )
        node2_key = nodes_all[edges[-1, 1]]  # LiTensor(7, )
        poses_SE3_key = mat2SE3(poses[-1, ...])  # LiTensor(7, )
        error_key = poses_SE3_key @ (node1_key.Inv() @ node2_key)  # LiTensor(7, )
        error2 = key_edge_weight * error_key.Log().tensor()[None, ...]

        error_final = torch.cat([error1, error2], 0)
        return error_final

