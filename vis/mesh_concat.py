import os
import open3d as o3d
import trimesh
import numpy as np

mesh_dir = "/home/javens/git_repos/MIPSFusion_torch/output/FastCaMo-large/floor2/0/result"
submesh_num = 11
mesh_basename_list = ["clip_%d.ply" % i for i in range(submesh_num)]

if __name__ == '__main__':
    submesh_num = len(mesh_basename_list)
    mesh_rgb_list = []
    for i in range(submesh_num):
        submesh_path = os.path.join(mesh_dir, mesh_basename_list[i])
        sub_mesh = trimesh.load_mesh(submesh_path)
        mesh_rgb_list.append(sub_mesh)

    mesh_rgb = trimesh.util.concatenate(mesh_rgb_list)
    mesh_concat_path = os.path.join(mesh_dir, "color_whole_mesh.ply")
    mesh_rgb.export(mesh_concat_path)


    mesh = o3d.io.read_triangle_mesh(mesh_concat_path)
    vertices = np.asarray(mesh.vertices)
    xyz_min = np.min(vertices, axis=0)
    xyz_max = np.max(vertices, axis=0)
    print("xyz_min:", xyz_min.tolist())
    print("xyz_max:", xyz_max.tolist())


