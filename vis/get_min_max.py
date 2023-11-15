import os
import open3d as o3d
import numpy as np

mesh_concat_path = os.path.join("/home/javens/NeuROFusion_test_1/output/IndoorLarge/stairs2/2023-04-24_21-52-20/final/mesh", "color_whole_mesh.ply")

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh(mesh_concat_path)
    vertices = np.asarray(mesh.vertices)
    xyz_min = np.min(vertices, axis=0)
    xyz_max = np.max(vertices, axis=0)
    print("xyz_min:", xyz_min.tolist())
    print("xyz_max:", xyz_max.tolist())