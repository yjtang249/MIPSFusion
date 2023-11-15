import numpy as np
import torch
import os
import open3d as o3d
from skimage import measure
import trimesh

from helper_functions.geometry_helper import extract_first_kf_pose, transform_points, convert_to_local_pts, convert_to_local_pts2, project_to_pixel
from vis.math_helper import compute_dist_to_center, convert_dist_to_weight, reduce_or, reduce_and, compute_weights, compute_weights2


def get_batch_query_fn(query_fn, args_num=1):
    if args_num == 1:
        fn = lambda f, i0, i1: query_fn(f[i0:i1, ...])
    else:
        fn = lambda f, i0, i1, v: query_fn(f[i0:i1, ...], v)
    return fn


class Mesher(object):
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.dataset = self.slam.dataset
        self.device = self.slam.device
        self.kfSet = self.slam.kfSet
        self.batch_size = 1024 * 16

        self.create_bounds()
        self.K = torch.tensor([[self.dataset.fx, 0., self.dataset.cx],
                               [0., self.dataset.fy, self.dataset.cy],
                               [0., 0., 1.]]).to(self.device)

    # @brief: get axis-aligned bounds
    def create_bounds(self):
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)
        self.coords_norm_factor = torch.from_numpy(np.array(self.config["mapping"]["localMLP_max_len"])).to(self.device)


    # @param bound: Tensor(3, 2)/ndarray(3, 2);
    #-@return grid_points: ndarray(r_x * r_y * r_z, 3)
    def get_grid_uniform(self, xyz_min, xyz_max, padding=0.05, voxel_size=0.05):
        resolution_x = ((xyz_max[0] + padding) - (xyz_min[0] - padding)) // voxel_size
        resolution_y = ((xyz_max[1] + padding) - (xyz_min[1] - padding)) // voxel_size
        resolution_z = ((xyz_max[2] + padding) - (xyz_min[2] - padding)) // voxel_size

        x = np.linspace(xyz_min[0] - padding, xyz_max[0] + padding, int(resolution_x))  # ndarray(resolution, )
        y = np.linspace(xyz_min[1] - padding, xyz_max[1] + padding, int(resolution_y))  # ndarray(resolution, )
        z = np.linspace(xyz_min[2] - padding, xyz_max[2] + padding, int(resolution_z))  # ndarray(resolution, )

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        return grid_points, [x, y, z]

    # TEST
    def draw_geometry(self, geo_list):
        o3d.visualization.draw_geometries(geo_list, zoom=0.4459,
                                          front=[0.9288, -0.2951, -0.2242],
                                          lookat=[1.6784, 2.0612, 1.4451],
                                          up=[-0.3402, -0.9189, -0.1996])
    # END TEST


    # @brief: select points by mask;
    # @param points: ndarray(n, 3)
    # @param mask: ndarray(n, ), dtype=np.bool
    #-@return selected_indices: indices of points with mask==True, ndarray(m, ), dtype=np.int32;
    #-@return selected_pts: points with mask==True, ndarray(m, 3).
    def select_points(self, points, mask):
        selected_indices = np.where(mask)[0]
        selected_pts = points[mask]
        return selected_indices, selected_pts


    # @brief: giving a pointcloud, create a VoxelGrid which can contains all points;
    # @param input_geometry: trimesh.Trimesh / open3d.io.geometry.PointCloud obj;
    # @param input_trimesh: whether input_geometry is Trimesh or PointCloud;
    #-@return: open3d.geometry.VoxelGrid obj.
    def create_voxelgrids_from_pointcloud(self, input_geometry, input_trimesh=False, expand_scale=1.2, shrink_scale=0.8, vox_size=0.5):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_geometry.vertices)) if input_trimesh else input_geometry
        all_points = [ np.asarray(pc.points) ]
        if expand_scale is not None:  # Step 1: expanded pointcloud
            pc_expand = o3d.geometry.PointCloud(pc)
            pc_expand = pc_expand.scale(expand_scale, pc_expand.get_center())
            all_points.append(np.asarray(pc_expand.points))
        if shrink_scale is not None:  # Step 2: shrunken pointcloud
            pc_shrink = o3d.geometry.PointCloud(pc)
            pc_shrink = pc_shrink.scale(shrink_scale, pc_shrink.get_center())
            all_points.append(np.asarray(pc_shrink.points))

        all_points = np.concatenate(all_points, axis=0)
        pc_whole = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc_whole, vox_size)
        return voxel_grid


    # @brief: giving a pointcloud, create an Oriented bounding box which can contains all points;
    # @param input_geometry: trimesh.Trimesh / open3d.io.geometry.PointCloud obj;
    # @param input_trimesh: whether input_geometry is Trimesh or PointCloud;
    #-@return: open3d.geometry.OrientedBoundingBox obj.
    def create_obbox_from_pointcloud(self, input_geometry, input_trimesh=False, expand_scale=1.1, shrink_scale=0.9):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_geometry.vertices)) if input_trimesh else input_geometry
        all_points = [np.asarray(pc.points)]
        if expand_scale is not None:  # Step 1: expanded pointcloud
            pc_expand = o3d.geometry.PointCloud(pc)
            pc_expand = pc_expand.scale(expand_scale, pc_expand.get_center())
            all_points.append(np.asarray(pc_expand.points))
        if shrink_scale is not None:  # Step 2: shrunken pointcloud
            pc_shrink = o3d.geometry.PointCloud(pc)
            pc_shrink = pc_shrink.scale(shrink_scale, pc_shrink.get_center())
            all_points.append(np.asarray(pc_shrink.points))

        all_points = np.concatenate(all_points, axis=0)
        pc_whole = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
        obbox = pc_whole.get_oriented_bounding_box()
        return obbox


    # @param kf_Ids: keyframe_Ids of each given keyframe, Tensor(n, );
    # @param kf_pose_c2w: c2w pose of each given keyframe, Tensor(n, 4, 4);
    # @param xyz_min: Tensor(3, 1);
    # @param xyz_max: Tensor(3, 1);
    #-@return bounding_gemetry: open3d.geometry.VoxelGrid / open3d.geometry.OrientedBoundingBox obj;
    #-@return point_cloud: open3d.geometry.PointCloud obj.
    def get_bounding_geometry(self, kf_Ids, kf_poses_c2w, xyz_min, xyz_max, using_obbox=False, vox_size=0.5):
        kf_num = int(kf_Ids.shape[0])
        rays = self.kfSet.rays[kf_Ids]  # Tensor(kf_num, num_rays_to_save, 7)

        points = []
        for i in range(kf_num):
            kf_pose_c2w = kf_poses_c2w[i]  # Tensor(4, 4)
            kf_rays = rays[i]
            rays_d_cam = kf_rays[:, :3]  # Tensor(num_rays_to_save, 3)
            target_depth = kf_rays[:, 6:7]  # Tensor(num_rays_to_save, 1)

            rays_d = torch.sum(rays_d_cam[:, None, :] * kf_pose_c2w[None, :3, :3], -1)  # Tensor(num_rays_to_save, 3)
            rays_d = rays_d.reshape(-1, 3)
            rays_o = kf_pose_c2w[None, :3, -1]
            pts_world = rays_o + rays_d * target_depth  # all points in World Coordinate System, Tensor(N, 3)

            depth_mask = (target_depth.squeeze(-1) > 0)
            pts_world_valid = pts_world[depth_mask].numpy()
            points.append(pts_world_valid)

        points = np.concatenate(points, 0)
        min_bound, max_bound = xyz_min.numpy().astype(np.float64), xyz_max.numpy().astype(np.float64)
        aa_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # filter out points lying out of bbox
        valid_indices = aa_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
        try:
            valid_points = points[np.array(valid_indices)]
        except IndexError as ie:
            print(3)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))

        if using_obbox:
            bounding_geometry = self.create_obbox_from_pointcloud(point_cloud)
        else:
            bounding_geometry = self.create_voxelgrids_from_pointcloud(point_cloud, vox_size=vox_size)
        return bounding_geometry, using_obbox, point_cloud


    # @brief: judge whether each given 3D point lies within given Axis_aligned box;
    # @param points: ndarray(n, 3);
    #-@return: ndarray(n, ), dtype=np.bool.
    def pts_in_abbox(self, points, axis_aligned_bbox):
        pts_3d = o3d.utility.Vector3dVector(points.astype(np.float64))
        valid_indices = axis_aligned_bbox.get_point_indices_within_bounding_box(pts_3d)  # list(int)
        containing_mask = np.zeros(shape=(points.shape[0]), dtype=np.bool)
        containing_mask[valid_indices] = True
        return containing_mask


    # @brief: judge whether each given 3D point lies within given Oriented boudning box;
    # @param points: ndarray(n, 3);
    #-@return: ndarray(n, ), dtype=np.bool.
    def pts_in_obbox(self, points, oriented_bbox):
        pts_3d = o3d.utility.Vector3dVector(points.astype(np.float64))
        valid_indices = oriented_bbox.get_point_indices_within_bounding_box(pts_3d)  # list(int)
        containing_mask = np.zeros(shape=(points.shape[0]), dtype=np.bool)
        containing_mask[valid_indices] = True
        return containing_mask


    # @brief: judge whether each given 3D point lies within given VoxelGrid;
    # @param points: ndarray(n, 3);
    #-@return: ndarray(n, ), dtype=np.bool.
    def pts_in_voxelgrid(self, points, voxelgrid):
        pts_3d = o3d.utility.Vector3dVector(points.astype(np.float64))
        containing_mask = voxelgrid.check_if_included(pts_3d)  # list(bool)
        containing_mask = np.array(containing_mask, dtype=np.bool)
        return containing_mask


    # @brief: giving a batch of 3D points and bounding_geometry, judge whether each point lies within the bounding_geometry;
    # @param points: ndarray(n, 3);
    # @param bounidng_geometry: open3d.geometry.VoxelGrid / open3d.geometry.OrientedBoundingBox obj;
    # @param using_obbox: whether the bounidng_geometry is Oriented bounding box;
    #-@return: containing mask of each point, ndarray(n, ), dtype=np.bool.
    def get_containing_mask(self, points, bounidng_geometry, using_obbox):
        point_num = points.shape[0]
        containing_function = self.pts_in_obbox if using_obbox else self.pts_in_voxelgrid
        fn_contain = get_batch_query_fn(containing_function, args_num=2)
        raw_mask = [fn_contain(points, i, i + self.batch_size, bounidng_geometry) for i in range(0, point_num, self.batch_size)]
        containing_mask = np.concatenate(raw_mask, axis=0)
        return containing_mask


    def get_containing_mask_abbox(self, points, abbox):
        point_num = points.shape[0]
        fn_contain = get_batch_query_fn(self.pts_in_abbox, args_num=2)
        raw_mask = [fn_contain(points, i, i + self.batch_size, abbox) for i in range(0, point_num, self.batch_size)]
        containing_mask = np.concatenate(raw_mask, axis=0)
        return containing_mask


    # @brief: get visibility mask of face according to visibility mask of vertices;
    # @param vert_mask: whether each vertex is seen, ndarray(V, 3);
    # @param faces: related vertices_Ids of each face, ndarray(F, 3);
    #-@return face_flag_seen: ndarray(F, ), dtype=bool.
    def get_face_mask(self, vert_mask, faces):
        unseen_mask = np.logical_not(vert_mask)
        face_vertices_flag = unseen_mask[faces]

        # face_flag_unseen = reduce_or(face_vertices_flag)  # (strict)whether each face in mesh is unseen, ndarray(F, ), dtype=bool
        face_flag_unseen = reduce_and(face_vertices_flag)  # (loose)whether each face in mesh is unseen, ndarray(F, ), dtype=bool

        face_flag_seen = ~face_flag_unseen
        return face_flag_seen


    # @brief: concatenate several meshes into a whole mesh;
    # @param trimesh_list: list of trimesh.Trimesh obj;
    #-@return: trimesh.Trimesh obj.
    def trimesh_concat(self, trimesh_list):
        concat_mesh = trimesh.util.concatenate(trimesh_list)
        return concat_mesh


    # @brief: judge whether each given point is seen to at least one keyframe;
    # @param points: given points in World Coordinate System, Tensor(n, 3);
    # @param kf_Ids: Tensor(k, );
    # @param first_kf_pose: first keyframe's pose (in World CS) if given localMLP, Tensor(4, 4);
    #-@return: Tensor(n, ), dtype=torch.bool.
    def point_mask(self, points, kf_Ids, kf_pose_c2w):
        points = points.to(torch.float32).to(self.device)
        seen_mask = torch.zeros_like(points[:, 0], dtype=torch.bool)
        kf_num = kf_Ids.shape[0]
        kf_pose_w2c = kf_pose_c2w.inverse()  # Tensor(n, 4, 4)

        # Step 1: for each selected keyframe, convert all given 3D points to its Camera Coordinate System
        kf_rot_w2c = kf_pose_w2c[:, :3, :3]  # rot mat w2c, Tensor(k, 3, 3)
        kf_trans_w2c = kf_pose_w2c[:, :3, 3]  # trans vec w2c, Tensor(k, 3)

        rotated_pts = torch.sum(points[None, :, None, :] * kf_rot_w2c[:, None, :, :], -1)  # Tensor(k, n, 3)
        transed_pts = rotated_pts + kf_trans_w2c[:, None, :]  # points in each keyframe's Camera Coordinate System, Tensor(k, n, 3)

        # Step 2: for each keyframe, compute its visibility mask
        for i in range(kf_num):
            kf_Id = int(kf_Ids[i].cpu().numpy())
            kf_depth = self.kfSet.rays[kf_Id, :, -1]  # Tensor(n_rays_h, n_rays_w)

            # 2.1: cam coords --> pixel coords, judge whether each point
            camera_points = transed_pts[i]  # Tensor(n, 3)
            uv = project_to_pixel(self.K, camera_points.unsqueeze(-1))  # Tensor(n, 2)
            edge = 20
            mask1 = (uv[:, 0] < self.config['cam']['W'] - edge) * (uv[:, 0] > edge) * (uv[:, 1] < self.config['cam']['H'] - edge) * (uv[:, 1] > edge)
            mask1 = mask1 & (camera_points[..., -1] < 0)  # camera coordinates with z < 0 means lying in front of the camera, Tensor(n, )

            # 2.2:
            max_depth = torch.max(kf_depth)  # max value in this depth image(metrics: m)
            min_depth = torch.min(kf_depth)  # min value in this depth image(metrics: m)
            mean_depth = torch.mean(kf_depth)  # mean value of this depth image(metrics: m)
            camera_points_z = torch.abs(camera_points[:, -1])
            mask2 = (camera_points_z > 0) * (camera_points_z < max_depth)

            mask = mask1 & mask2
            seen_mask = torch.logical_or(seen_mask, mask)
        return seen_mask.cpu()


    # @brief: get mesh of a given localMLP;
    #-@return bounding_geometry: geometry that bounds the submesh, open3d.geometry.VoxelGrid / open3d.geometry.OrientedBoundingBox obj;
    #-@return sub_mesh: RGB mesh of this submap, trimesh.Trimesh obj;
    #-@return using_obbox: whether the bounding geometry is oriented-bbox all voxelgrid, bool.
    @torch.no_grad()
    def extract_single_mesh(self, model, localMLP_Id, kf_num=None, save_path=None, render_color=True):
        if kf_num is None:
            kf_num = self.kfSet.collected_kf_num[0]

        # Step 1: find all related keyframes and their poses
        related_kf_mask = self.kfSet.get_related_keyframes(localMLP_Id, kf_num)
        related_kf_Ids = torch.where(related_kf_mask > 0)[0]
        if related_kf_Ids.shape[0] == 0:
            return

        first_kf_pose, first_kf_Id, poses_local, _, _, _, _, _ = self.kfSet.extract_localMLP_vars_given(localMLP_Id, related_kf_Ids, self.slam.kf_c2w[:kf_num],
                                                                                                        self.slam.est_c2w_data, self.slam.keyframe_ref[:kf_num])
        poses_world = first_kf_pose @ poses_local  # world poses of all selected keyframes, Tensor(n, 4, 4)

        # Step 2: construct meshgrids
        # 2.1: get min and max bound from localMLP center and axis-aligned length
        xyz_len = self.kfSet.localMLP_info[localMLP_Id][1:]  # Tensor(6, )
        xyz_center, xyz_len = xyz_len[:3], xyz_len[3:]
        xyz_min, xyz_max = xyz_center - 0.5 * xyz_len, xyz_center + 0.5 * xyz_len

        if self.marching_cube_bound is not None:
            xyz_min_mc = self.marching_cube_bound[:, 0].cpu()
            xyz_max_mc = self.marching_cube_bound[:, 1].cpu()
            xyz_min, _ = torch.max( torch.stack([xyz_min, xyz_min_mc], -1), -1 )
            xyz_max, _ = torch.min( torch.stack([xyz_max, xyz_max_mc], -1), -1 )

        # 2.2: construct a coarse bounding geometry structure (Oriented bounding box / VoxelGrid)
        kf_Ids_uniq = torch.where(self.kfSet.keyframe_localMLP[:, 0] == localMLP_Id)[0]
        kf_frame_Ids = kf_Ids_uniq * self.config["mapping"]["keyframe_every"]  # Tensor(n, )
        kf_pose_local = self.slam.est_c2w_data[kf_frame_Ids]  # Tensor(n, 4, 4)
        kf_pose_world = first_kf_pose @ kf_pose_local
        bounding_geometry, using_obbox, _ = self.get_bounding_geometry(kf_Ids_uniq, kf_pose_world.cpu(), xyz_min[..., None], xyz_max[..., None], using_obbox=False)

        # 2.3: create all meshgrids, and compute containing mask
        grid_points, mesh_bound = self.get_grid_uniform(xyz_min.numpy(), xyz_max.numpy(), voxel_size=self.config["mesh"]["voxel_final"])  # ndarray(N, 3) / _
        containing_mask = self.get_containing_mask(grid_points, bounding_geometry, using_obbox)  # ndarray(N, ), dtype=np.bool


        # Step 3: do inference for each meshgrid and do Marching Cubes
        first_kf_pose_inv = first_kf_pose.inverse().to(self.device)
        fn_transform = get_batch_query_fn(convert_to_local_pts2, args_num=2)

        # 3.1: world coords --> local coords
        grid_points = torch.from_numpy(grid_points.astype(np.float32)).to(self.device)
        raw = [fn_transform(grid_points, i, i + self.batch_size, first_kf_pose) for i in range(0, grid_points.shape[0], self.batch_size)]
        grid_points_local = torch.cat(raw, dim=0)  # local coords of all meshgrids, Tensor(n, 3)
        if self.config['grid']['tcnn_encoding']:
            if self.config["grid"]["use_bound_normalize"]:
                grid_points_local = (grid_points_local - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            else:
                grid_points_local = (grid_points_local + self.coords_norm_factor) / (2 * self.coords_norm_factor)

        # 3.2: infer to get pred SDF of these meshgrids
        fn_sdf = get_batch_query_fn(model.query_sdf)
        raw_sdf = [fn_sdf(grid_points_local, i, i + self.batch_size) for i in range(0, grid_points_local.shape[0], self.batch_size)]
        grids_sdf = torch.cat(raw_sdf, dim=0).detach().cpu().numpy()  # Tensor(n, 1)

        # 3.3: do Marching Cubes
        SDF = grids_sdf.reshape(mesh_bound[1].shape[0], mesh_bound[0].shape[0], mesh_bound[2].shape[0]).transpose([1, 0, 2])
        final_mask = containing_mask.reshape(mesh_bound[1].shape[0], mesh_bound[0].shape[0], mesh_bound[2].shape[0]).transpose([1, 0, 2])
        vertices, faces, normals, values = measure.marching_cubes(volume=SDF, level=0.,
                                                                  spacing=(mesh_bound[0][2] - mesh_bound[0][1],
                                                                           mesh_bound[1][2] - mesh_bound[1][1],
                                                                           mesh_bound[2][2] - mesh_bound[2][1]),
                                                                  mask=final_mask)

        # 3.4: convert local coordinates back to world coordinates (metrics: m)
        vertices = vertices + np.array([mesh_bound[0][0], mesh_bound[1][0], mesh_bound[2][0]])  # local coordinates of all vertices, ndarray(v, 3)
        normals = np.array(normals)[:, [1, 0, 2]]
        sub_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)

        # 3.5: filter out small components
        components = sub_mesh.split(only_watertight=False)
        new_components = []
        for comp in components:
            if comp.area > 0.5:  # remove_small_geometry_threshold
                new_components.append(comp)
        sub_mesh = trimesh.util.concatenate(new_components)
        vertices, faces = sub_mesh.vertices, sub_mesh.faces

        # 3.6: construct returning bounding_geometry
        if using_obbox:
            bounding_geometry = self.create_obbox_from_pointcloud(sub_mesh, input_trimesh=True)
        else:
            bounding_geometry = self.create_voxelgrids_from_pointcloud(sub_mesh, input_trimesh=True)

        # Step 4: filter out vertices which is unseen to any keyframes
        seen_mask = self.point_mask(torch.from_numpy(vertices), kf_Ids_uniq, kf_pose_world)
        face_flag_seen = self.get_face_mask(seen_mask.numpy(), faces)
        sub_mesh.update_faces(face_flag_seen)

        # Step 5: query RGB value for each vertex
        if render_color:
            vertices, faces = sub_mesh.vertices, sub_mesh.faces
            vertices = torch.from_numpy(vertices.astype(np.float32)).to(self.device)
            raw = [ fn_transform(vertices, i, i + self.batch_size, first_kf_pose) for i in range(0, vertices.shape[0], self.batch_size) ]
            vertices_local = torch.cat(raw, dim=0)  # local coords of all vertices, Tensor(v, 3)
            if self.config['grid']['tcnn_encoding']:
                if self.config["grid"]["use_bound_normalize"]:
                    vertices_local = (vertices_local - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
                else:
                    vertices_local = (vertices_local + self.coords_norm_factor) / (2 * self.coords_norm_factor)

            fn_color = get_batch_query_fn(model.query_color)
            raw_rgb = [fn_color(vertices_local, i, i + self.batch_size) for i in range(0, vertices_local.shape[0], self.batch_size)]
            vertices_rgb = torch.cat(raw_rgb, dim=0).detach().cpu().numpy()  # Tensor(v, 3)
            vertices = vertices.detach().cpu().numpy()
            sub_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, vertex_colors=vertices_rgb)

        if save_path is not None:
            sub_mesh.export(save_path)
        torch.cuda.empty_cache()
        print("Finished rendering submesh_%d ..." % int(localMLP_Id))
        return bounding_geometry, sub_mesh, using_obbox


    @torch.no_grad()
    def extract_mesh_jointly(self, model_list, bounding_geo_list, submesh_list, using_obbox, save_path=None, render_color=True):
        mesh_gray, submesh_tri_list, abbox_list, obbox_list, centroids = self.extract_mesh_jointly_geometry(model_list, submesh_list, bounding_geo_list, using_obbox)

        if render_color:
            mesh_color = self.extract_mesh_jointly_color(mesh_gray, model_list, submesh_tri_list, abbox_list, obbox_list, centroids, save_path)


    # @brief: jointly render the whole mesh using all rendered submeshes;
    # @param model_list
    # @param bounding_geo_list
    # @param submesh_list
    # @param using_obbox: whether the bounding geometry is oriented-bbox all voxelgrid, bool;
    @torch.no_grad()
    def extract_mesh_jointly_geometry(self, model_list, submesh_list, bounding_geo_list, using_obbox, save_path=None):
        kf_num = self.kfSet.frame_ids.shape[0]
        submesh_num = len(submesh_list)

        # Step 1: compute axis_aligned bbox and oriented bbox for each submesh
        submesh_tri_list = []
        centroid_list = []  # list of ndarray(1, 3)
        abbox_list = []
        abbox_min_list, abbox_max_list = [], []
        obbox_list = []
        for i in range(submesh_num):
            submesh = submesh_list[i]
            submesh_tri = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(submesh.vertices), triangles=o3d.utility.Vector3iVector(submesh.faces))
            submesh_tri_list.append(submesh_tri)

            centroid_this = submesh_tri.get_center().astype(np.float32)[None, ...]  # ndarray(1, 3)
            centroid_list.append(centroid_this)

            mesh_expand = submesh_tri.scale(1.1, submesh_tri.get_center())
            oriented_bbox = mesh_expand.get_oriented_bounding_box()
            axis_aligned_bbox = mesh_expand.get_axis_aligned_bounding_box()

            abbox_list.append(axis_aligned_bbox)
            abbox_min_list.append(axis_aligned_bbox.get_min_bound())
            abbox_max_list.append(axis_aligned_bbox.get_max_bound())
            obbox_list.append(oriented_bbox)
        centroids = np.concatenate(centroid_list, axis=0)  # ndarray(localMLP_num, 3)

        # Step 2: get scene bound and construct meshgrids
        abbox_min_list = np.stack(abbox_min_list, -1)  # ndarray(3, n)
        abbox_max_list = np.stack(abbox_max_list, -1)  # ndarray(3, n)
        bound_min = np.min(abbox_min_list, -1)  # ndarray(3, )
        bound_max = np.max(abbox_max_list, -1)  # ndarray(3, )
        grid_points, mesh_bound = self.get_grid_uniform(bound_min, bound_max, voxel_size=self.config["mesh"]["voxel_final"])  # ndarray(N, 3) / _
        grid_num = grid_points.shape[0]

        # Step 3: construct and compute containing mask
        grid_mask = np.zeros(shape=(grid_num, submesh_num), dtype=np.bool)  # visibility of each meshgrid to each submap, ndarray(grid_num, submesh_num)
        grid_mask_mc = np.zeros(shape=(grid_num, submesh_num), dtype=np.bool)  # mask for Marching Cubes of each submap, ndarray(grid_num, submesh_num)
        grid_entropy = np.zeros(shape=(grid_num, submesh_num), dtype=np.float32)  # pred entropy of each submap to each meshgrid, ndarray(grid_num, submesh_num)
        grid_tsdf = np.full(shape=(grid_num, submesh_num), fill_value=-1, dtype=np.float32)
        grid_dist_weight = np.zeros(shape=(grid_num, submesh_num), dtype=np.float32)  # ndarray(grid_num, submesh_num)
        grid_weighted_tsdf = np.full(shape=(grid_num, ), fill_value=-1, dtype=np.float32)

        kf_pose_world_list, kf_Ids_list = [], []
        for i in range(submesh_num):
            localMLP_Id = i
            model = model_list[i]

            # 3.1: compute seen mask of each grid
            first_kf_pose, first_kf_Id = self.kfSet.extract_first_kf_pose(i, self.slam.kf_c2w[:kf_num])  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )
            containing_mask1 = self.get_containing_mask_abbox(grid_points, abbox_list[i])
            grid_mask_mc[:, i] = containing_mask1
            valid_indices = np.where(containing_mask1)[0]  # indices of all valid meshgrids
            valid_grids = grid_points[containing_mask1]  # all valid meshgrids to current submap, ndarray(grid_num1, 3)

            # 3.2: world coords --> local coords
            fn_transform = get_batch_query_fn(convert_to_local_pts2, args_num=2)
            valid_grids = torch.from_numpy(valid_grids.astype(np.float32)).to(self.device)
            raw = [fn_transform(valid_grids, i, i + self.batch_size, first_kf_pose) for i in range(0, valid_grids.shape[0], self.batch_size)]
            valid_grid_points_local = torch.cat(raw, dim=0)  # local coords of all meshgrids, Tensor(valid_grid_num, 3)
            if self.config['grid']['tcnn_encoding']:
                if self.config["grid"]["use_bound_normalize"]:
                    valid_grid_points_local = (valid_grid_points_local - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
                else:
                    valid_grid_points_local = (valid_grid_points_local + self.coords_norm_factor) / (2 * self.coords_norm_factor)

            # 3.3: infer to get pred SDF/entropy/prob of these meshgrids
            fn_sdf = get_batch_query_fn(model.query_sdf_entropy_prob)
            raw_sdf_vars = [fn_sdf(valid_grid_points_local, i, i + self.batch_size) for i in range(0, valid_grid_points_local.shape[0], self.batch_size)]
            grids_sdf_vars = torch.cat(raw_sdf_vars, dim=0).detach().cpu().numpy()  # ndarray(valid_grid_num, 7)
            valid_grids_tsdf = grids_sdf_vars[:, 0]
            valid_grids_entropy = grids_sdf_vars[:, 1]

            np.put(grid_tsdf[:, i], valid_indices, valid_grids_tsdf)
            np.put(grid_entropy[:, i], valid_indices, valid_grids_entropy)

            # 3.4:
            pts_dist = compute_dist_to_center(valid_grids.cpu().numpy(), centroids[i])  # ndarray(valid_grid_num, )
            pts_dist_weight = convert_dist_to_weight(pts_dist)  # ndarray(valid_grid_num, )
            np.put(grid_dist_weight[:, i], valid_indices, pts_dist_weight)

            # 3.5: finner filtering (filter out those meshgrids invisible to any keyframe)
            kf_Ids_uniq = torch.where(self.kfSet.keyframe_localMLP[:, 0] == localMLP_Id)[0]
            kf_frame_Ids = kf_Ids_uniq * self.config["mapping"]["keyframe_every"]  # Tensor(n, )
            kf_pose_local = self.slam.est_c2w_data[kf_frame_Ids]  # Tensor(n, 4, 4)
            kf_pose_world = first_kf_pose @ kf_pose_local
            kf_pose_world_list.append(kf_pose_world)
            kf_Ids_list.append(kf_Ids_uniq)

            containing_mask2_1 = self.get_containing_mask(valid_grids.cpu().numpy(), obbox_list[i], using_obbox=True)  # ndarray(grid_num1, ), dtype=np.bool
            seen_mask = self.point_mask(valid_grids, kf_Ids_uniq, kf_pose_world).numpy()
            containing_mask2 = containing_mask2_1 & seen_mask  # ndarray(grid_num1, )
            np.put(grid_mask[:, i], valid_indices, containing_mask2)
        # end for
        kf_poses_world = torch.cat(kf_pose_world_list, 0)
        kf_Ids_all = torch.cat(kf_Ids_list, 0)

        # Step 4: compute weights
        final_mask_mc = reduce_or(grid_mask_mc)  # mask of each grid for marching cubes, ndarray(grid_num, ), dtype=np.bool
        final_mask = reduce_or(grid_mask)  # visibility mask of each grid, ndarray(grid_num, ), dtype=np.bool
        final_mask_mc = final_mask_mc & final_mask
        final_mask_indices = np.where(final_mask)[0]
        grid_entropy = np.clip(grid_entropy, 0, 10000.)

        ################################# need to be modified #################################
        grid_weights = compute_weights(grid_entropy, grid_dist_weight, grid_mask)  # ndarray(grid_num, localMLP_num), dtype=np.float32
        weighted_sdf = np.sum(grid_tsdf * grid_weights, axis=-1)  # ndarray(grid_num, ), dtype=np.float32
        np.put(grid_weighted_tsdf, final_mask_indices, weighted_sdf[final_mask_indices])
        ################################# END modified #################################


        # Step 5: do marching cubes
        # 5.1:
        SDF = grid_weighted_tsdf.reshape(mesh_bound[1].shape[0], mesh_bound[0].shape[0], mesh_bound[2].shape[0]).transpose([1, 0, 2])
        final_mask_mc = final_mask_mc.reshape(mesh_bound[1].shape[0], mesh_bound[0].shape[0], mesh_bound[2].shape[0]).transpose([1, 0, 2])
        vertices, faces, normals, values = measure.marching_cubes(volume=SDF,
                                                                  level=0.,
                                                                  spacing=(mesh_bound[0][2] - mesh_bound[0][1],
                                                                           mesh_bound[1][2] - mesh_bound[1][1],
                                                                           mesh_bound[2][2] - mesh_bound[2][1]),
                                                                  mask=final_mask_mc)

        # 5.2: convert back to world coordinates (metrics: m)
        vertices = vertices + np.array([mesh_bound[0][0], mesh_bound[1][0], mesh_bound[2][0]])
        normals = np.array(normals)[:, [1, 0, 2]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)  # coarse mesh (some vertices will be filtered out)

        # Step 6: filter out vertices which is unseen to any keyframes
        seen_mask = self.point_mask(torch.from_numpy(vertices), kf_Ids_all, kf_poses_world)
        face_flag_seen = self.get_face_mask(seen_mask.numpy(), faces)
        mesh.update_faces(face_flag_seen)

        # Step 7: mesh clipping (remove some mesh vertices and corresponding faces)
        components = mesh.split(only_watertight=False)
        new_components = []
        for comp in components:
            if comp.area > 0.5:  # remove_small_geometry_threshold: 用于移除reconstructed mesh中一些小杂块时要给出的一个threshold
                new_components.append(comp)
        mesh = trimesh.util.concatenate(new_components)

        # Step 8: final filtering for mesh vertices
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        vert_num = vertices.shape[0]
        vert_mask = np.zeros(shape=(vert_num, submesh_num), dtype=np.bool)  # visibility of each vertex to each submap, ndarray(vert_num, submesh_num)
        for i in range(submesh_num):
            vert_contain_mask = self.get_containing_mask(vertices, bounding_geo_list[i], using_obbox)  # ndarray(vert_num, )
            vert_mask[:, i] = vert_contain_mask
        vert_mask_final = reduce_or(vert_mask)  # whether each vertex lies within at least 1 submesh's bounding geometry, ndarray(vert_num, )

        # concat_mesh = self.trimesh_concat(submesh_list)
        # if using_obbox:
        #     bounding_geometry = self.create_obbox_from_pointcloud(concat_mesh, input_trimesh=True)
        # else:
        #     bounding_geometry = self.create_voxelgrids_from_pointcloud(concat_mesh, input_trimesh=True)
        # vert_mask_final = self.get_containing_mask(vertices, bounding_geometry, using_obbox)  # ndarray(vert_num, )

        face_flag_seen = self.get_face_mask(vert_mask_final, faces)
        mesh.update_faces(face_flag_seen)

        if save_path is not None:
            mesh.export(save_path)
        return mesh, submesh_tri_list, abbox_list, obbox_list, centroids


    # @brief: jointly render the whole mesh using all rendered submeshes;
    # @param model_list
    # @param submesh_tri_list
    # @param abbox_list
    # @param obbox_list:
    # @param centroids:
    @torch.no_grad()
    def extract_mesh_jointly_color(self, mesh_gray, model_list, submesh_tri_list, abbox_list, obbox_list, centroids, save_path=None):
        kf_num = self.kfSet.frame_ids.shape[0]
        vertices, faces = mesh_gray.vertices, mesh_gray.faces
        submesh_num = len(submesh_tri_list)
        vertices = np.asarray(vertices)
        vert_num = vertices.shape[0]

        # Step 1: construct and compute containing mask
        vert_mask = np.zeros(shape=(vert_num, submesh_num), dtype=np.bool)  # visibility of each vertex to each submap, ndarray(vert_num, submesh_num)
        vert_entropy = np.zeros(shape=(vert_num, submesh_num), dtype=np.float32)  # pred entropy of each submap to each vertex, ndarray(vert_num, submesh_num)
        vert_dist_weight = np.zeros(shape=(vert_num, submesh_num), dtype=np.float32)  # ndarray(vert_num, submesh_num)
        vert_color_r = np.zeros(shape=(vert_num, submesh_num), dtype=np.float32)
        vert_color_g = np.zeros(shape=(vert_num, submesh_num), dtype=np.float32)
        vert_color_b = np.zeros(shape=(vert_num, submesh_num), dtype=np.float32)

        for i in range(submesh_num):
            localMLP_Id = i
            model = model_list[i]

            # 1.1: compute seen mask of each vertex
            first_kf_pose, first_kf_Id = self.kfSet.extract_first_kf_pose(i, self.slam.kf_c2w[:kf_num])  # first keyframe's pose in World Coordinate System / kf_Id of given localMLP, Tensor(4, 4)/Tensor(, )
            containing_mask1 = self.get_containing_mask_abbox(vertices, abbox_list[i])
            valid_indices = np.where(containing_mask1)[0]  # indices of all valid vertices
            valid_vertices = vertices[containing_mask1]  # all valid vertices to current submap, ndarray(vert_num1, 3)

            # 1.2: world coords --> local coords
            fn_transform = get_batch_query_fn(convert_to_local_pts2, args_num=2)
            valid_vertices = torch.from_numpy(valid_vertices.astype(np.float32)).to(self.device)
            raw = [fn_transform(valid_vertices, i, i + self.batch_size, first_kf_pose) for i in range(0, valid_vertices.shape[0], self.batch_size)]
            valid_vertices_local = torch.cat(raw, dim=0)  # local coords of all vertices, Tensor(valid_vert_num, 3)
            if self.config['grid']['tcnn_encoding']:
                if self.config["grid"]["use_bound_normalize"]:
                    valid_vertices_local = (valid_vertices_local - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
                else:
                    valid_vertices_local = (valid_vertices_local + self.coords_norm_factor) / (2 * self.coords_norm_factor)

            # 1.3: infer to get pred RGB/entropy of these vertices
            fn_sdf = get_batch_query_fn(model.query_color_sdf)
            raw_rgb_vars = [fn_sdf(valid_vertices_local, i, i + self.batch_size) for i in range(0, valid_vertices_local.shape[0], self.batch_size)]
            vertices_rgb_vars = torch.cat(raw_rgb_vars, dim=0)  # Tensor(valid_vert_num, 10)
            valid_verts_rgb = torch.sigmoid(vertices_rgb_vars[:, :3]).detach().cpu().numpy()  # ndarray(vert_num, 3)
            valid_verts_entropy = vertices_rgb_vars[:, 4].detach().cpu().numpy()

            np.put(vert_color_r[:, i], valid_indices, valid_verts_rgb[:, 0])
            np.put(vert_color_g[:, i], valid_indices, valid_verts_rgb[:, 1])
            np.put(vert_color_b[:, i], valid_indices, valid_verts_rgb[:, 2])
            np.put(vert_entropy[:, i], valid_indices, valid_verts_entropy)

            # 1.4:
            pts_dist = compute_dist_to_center(valid_vertices.cpu().numpy(), centroids[i])  # ndarray(valid_vert_num, )
            pts_dist_weight = convert_dist_to_weight(pts_dist)  # ndarray(valid_vert_num, )
            np.put(vert_dist_weight[:, i], valid_indices, pts_dist_weight)

            # 1.5: finner filtering (filter out those meshgrids invisible to any keyframe)
            kf_Ids_uniq = torch.where(self.kfSet.keyframe_localMLP[:, 0] == localMLP_Id)[0]
            kf_frame_Ids = kf_Ids_uniq * self.config["mapping"]["keyframe_every"]  # Tensor(n, )
            kf_pose_local = self.slam.est_c2w_data[kf_frame_Ids]  # Tensor(n, 4, 4)
            kf_pose_world = first_kf_pose @ kf_pose_local

            containing_mask2_1 = self.get_containing_mask(valid_vertices.cpu().numpy(), obbox_list[i], using_obbox=True)  # ndarray(vert_num1, ), dtype=np.bool
            seen_mask = self.point_mask(valid_vertices, kf_Ids_uniq, kf_pose_world).numpy()
            containing_mask2 = containing_mask2_1 & seen_mask  # ndarray(grid_num1, )
            np.put(vert_mask[:, i], valid_indices, containing_mask2)
        # end for

        # Step 2: compute weights
        vert_entropy = np.clip(vert_entropy, 0, 10000.)
        vert_weights = compute_weights(vert_entropy, vert_dist_weight, vert_mask)  # ndarray(vert_num, localMLP_num), dtype=np.float32

        weighted_color_r = np.sum(vert_color_r * vert_weights, axis=-1)  # ndarray(vertices_num, ), dtype=np.float32
        weighted_color_g = np.sum(vert_color_g * vert_weights, axis=-1)  # ndarray(vertices_num, ), dtype=np.float32
        weighted_color_b = np.sum(vert_color_b * vert_weights, axis=-1)  # ndarray(vertices_num, ), dtype=np.float32
        weighted_rgb = np.stack((weighted_color_r, weighted_color_g, weighted_color_b), axis=-1)  # ndarray(vertices_num, 3), dtype=np.float32

        # Step 3: construct color mesh
        recon_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=weighted_rgb)
        if save_path is not None:
            recon_mesh.export(save_path)
        return recon_mesh