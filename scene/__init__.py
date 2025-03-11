#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import json
import os
import random

import numpy as np
import torch
from arguments import ModelParams
from scipy.interpolate import CubicSpline
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.graphics_utils import getWorld2View2
from utils.render_utils import focus_point_fn
from utils.system_utils import searchForMaxIteration

from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel


def viewmatrix_(z, up, pos):
    vec2 = normalize_(z)
    vec1_avg = normalize_(up)
    vec0 = normalize_(np.cross(vec1_avg, vec2))
    vec1 = normalize_(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return np.concatenate([m, np.array([[0, 0, 0, 1]])], 0)


def normalize_(v):
    return v / np.linalg.norm(v)


class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]
    ):
        """
        :param path: Path to colmap scene main folder.
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        print("model_path: ", self.model_path)
        # breakpoint()
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.custom_cameras = {}
        self.fancy_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras.xml")):
            print("Metashape dataset is loaded")
            scene_info = sceneLoadTypeCallbacks["Metashape"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "Rdy2Use")):
            print("RealityCapture dataset is loaded")
            scene_info = sceneLoadTypeCallbacks["RC"](args.source_path, args.eval, args)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

            # TODO: uncomment
            self.custom_cameras[resolution_scale] = self.generate_intermidiate(
                direction=self.args.direction, resolution_scale=resolution_scale, dataset="pollen"
            )

            # self.fancy_cameras[resolution_scale] = self.generate_fancy_path(resolution_scale=resolution_scale)

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getCustomCameras(self, scale=1.0):
        return self.custom_cameras[scale]

    def getFancyCameras(self, scale=1.0):
        return self.fancy_cameras[scale]

    def generate_fancy_path(self, resolution_scale=1):

        names_intermidiate = [
            "Pollen-center-005-10",
            # "Pollen-center-005-9",
            # "Pollen-center-005-8",
            # "Pollen-center-005-7",
            "Pollen-center-005-6",
            # "Pollen-center-005-5",
            # "Pollen-center-005-4",
            # "Pollen-center-005-3",
            # "Pollen-center-005-2",
            # "Pollen-center-005-1",
            "Pollen-center-005+11",
            # "Pollen-center-005+12",
            # "Pollen-center-005+13",
            # "Pollen-center-005+14",
            # "Pollen-center-005+15",
            "Pollen-center-005+16",
            # "Pollen-center-005+17",
            # "Pollen-center-005+18",
            # "Pollen-center-005+19",
            "Pollen-center-005+20",
        ]
        # 0, 9, 18

        indices = []
        names = []

        for cam in self.train_cameras[resolution_scale]:
            names.append(cam.image_name)

        for name in names_intermidiate:
            index = names.index(name)
            indices.append(index)

        full_cams = []
        for i in range(len(indices)):
            cam = self.train_cameras[resolution_scale][indices[i]]
            full_cams.append(cam)

        cams = [full_cams[0], full_cams[2], full_cams[4]]

        print(cams)
        # breakpoint()

        def interpolate_view_translations(start_translation, end_translation, num_points):
            # Time values for interpolation
            t = np.arange(2)
            t_new = np.linspace(0, 1, num_points)

            t_reversed = t[::-1]

            # Interpolating each dimension separately
            interpolated_translations = []
            for dim in range(3):

                values = [start_translation[dim], end_translation[dim]]
                # values_reversed = [end_translation[dim], start_translation[dim]]

                cs = CubicSpline(t, values)
                # cs = CubicSpline(t_reversed, values_reversed)
                interpolated_translations.append(cs(t_new))

            # Combine results into a (num_points, 3) array
            return np.array(interpolated_translations).T

        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cams])
        # c2ws = np.array([np.asarray((cam.world_view_transform).cpu().numpy()) for cam in cams])
        c2ws_full = np.array(
            [np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in full_cams]
        )
        # c2ws_full = np.array([np.asarray((cam.world_view_transform).cpu().numpy()) for cam in full_cams])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        poses_full = c2ws_full[:, :3, :] @ np.diag([1, -1, -1, 1])

        # c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cams])
        # poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])

        positions = []
        # for i in range(3 - 1):
        #     # could you take inverse?

        #     C2W = poses[i]
        #     C2W_next = poses[i + 1]
        #     start_translation = C2W[:3, 3]
        #     end_translation = C2W_next[:3, 3]

        #     # C2W = np.linalg.inv(poses[i])
        #     # C2W_next = np.linalg.inv(poses[i + 1])
        #     # start_translation = C2W[:3, 3]
        #     # end_translation = C2W_next[:3, 3]

        #     # start_rotation = C2W[:3, :3]
        #     # end_rotation = C2W_next[:3, :3]

        #     # intermidiate_T, _ = interpolate_views(
        #     #     start_translation, end_translation, start_rotation, end_rotation, 30
        #     # )

        #     intermidiate_T = interpolate_view_translations(start_translation, end_translation, 10)
        #     # intermidiate_T = interpolate_view_translations(end_translation, start_translation, 10)
        def normalize(x: np.ndarray) -> np.ndarray:
            """Normalization helper function."""
            return x / np.linalg.norm(x)

        # focal = 11108.022927511847
        center = focus_point_fn(poses_full)

        print(poses[0, :3, 3], poses[2, :3, 3])
        #     positions.extend(intermidiate_T)

        start = poses[1, :3, :4]
        start[:3, 3] = start[:3, 3] - center
        start[1, 3] = start[1, 3] - 500

        # positions = []
        # for theta in np.linspace(0.0, np.pi, 20 + 1)[:-1]:
        #     c = np.dot(start, np.array([600 * np.cos(theta), 20 * -np.sin(theta), 500 * np.sin(theta), 1.0]))
        #     # z = normalize(c - np.dot(start, np.array([0, 0, -focal, 1.0])))
        #     # print(c)
        #     positions.append(c)

        positions = []
        for theta in np.linspace(0.0, np.pi, 20 + 1)[:-1]:
            c = np.dot(start, np.array([600 * np.cos(theta), 20 * -np.sin(theta), 225 * np.sin(theta), 1.0]))
            # z = normalize(c - np.dot(start, np.array([0, 0, -focal, 1.0])))
            # print(c)
            positions.append(c)

        # for theta in np.linspace(0.0, 2.0 * np.pi, 20)[:-1]:
        #     c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5), 1.0]) * rads)
        #     z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        #     render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))

        # positions = []
        # for theta in np.linspace(0.0, 2.0 * np.pi, 20)[:-1]:
        #     # 楕円形のパラメータ (a: x軸半径, b: y軸半径)
        #     a = 100.0  # 横長にしたいので、x軸を大きくする
        #     b = 50  # y軸は元のままの範囲

        #     # 楕円の方程式に基づいて位置を計算
        #     c = np.dot(
        #         poses[0, :3, :4], np.array([a * np.cos(theta), b * -np.sin(theta), -b * np.sin(theta * 0.5), 1.0]) * 0.5
        #     )
        #     positions.append(c[:3])

        # up vector
        avg_up = poses_full[:, :3, 1].mean(0)
        avg_up = avg_up / np.linalg.norm(avg_up)
        ind_up = np.argmax(np.abs(avg_up))
        up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

        # z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        new_poses = np.stack([viewmatrix_(p - center, up, p) for p in positions])

        traj = []
        for c2w in new_poses:
            c2w = c2w @ np.diag([1, -1, -1, 1])
            cam = copy.deepcopy(cams[0])
            cam.image_height = int(cam.image_height / 2) * 2
            cam.image_width = int(cam.image_width / 2) * 2
            cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
            # cam.world_view_transform = torch.from_numpy(c2w.T).float().cuda()
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
            ).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            traj.append(cam)

        return traj

    def generate_intermidiate(self, direction="horizontal", resolution_scale=1, dataset="caterpilar", section=1):
        print(direction)

        names_intermidiate = []
        if dataset == "pollen":
            if direction == "horizontal":
                # sort by this order
                # render poses
                names_intermidiate = [
                    # "Pollen-center-005-10",  # 1
                    # "Pollen-center-005-9",
                    # "Pollen-center-005-8",
                    # "Pollen-center-005-7",
                    # "Pollen-center-005-6",  # 2
                    # "Pollen-center-005-5",
                    # "Pollen-center-005-4",
                    # "Pollen-center-005-3",
                    # "Pollen-center-005-2",
                    "Pollen-center-005+12",  # 3
                    "Pollen-center-005+13",
                    "Pollen-center-005+14",
                    "Pollen-center-005+15",
                    "Pollen-center-005+16",
                    "Pollen-center-005+17",  # 4
                    # "Pollen-center-005+18",
                    # "Pollen-center-005+19",
                    # "Pollen-center-005+20",
                ]
            elif direction == "vertical":
                names_intermidiate = [
                    "Pollen-center-006",
                    # "Pollen-center-007",
                    # "Pollen-center-008",
                    # "Pollen-center-009",
                    "Pollen-center-010",
                    # "Pollen-center-011",
                    # "Pollen
                    # -center-012",
                    # "Pollen-center-013",
                    # "Pollen-center-014",
                    "Pollen-center-015",
                    # "Pollen-center-016",
                    # "Pollen-center-017",
                ]
        elif dataset == "caterpillar":
            names_intermidiate = [
                # "Caterpillar2_100",
                # "Caterpillar2_150",
                "Caterpillar2_200",
                "Caterpillar2_250",
                "Caterpillar2_300",
                "Caterpillar2_350",
                "Caterpillar2_400",
            ]

        indices = []
        names = []

        for cam in self.train_cameras[resolution_scale]:
            names.append(cam.image_name)

        # breakpoint()
        # print(names)
        for name in names_intermidiate:
            index = names.index(name)
            indices.append(index)

        # poses = []
        # for i in range(len(indices)-1):
        #     cam = self.train_cameras[resolution_scale][indices[i]]
        #     W2C = getWorld2View2(cam.R, cam.T)
        #     C2W = np.linalg.inv(W2C)
        #     poses.append(cam)

        #   # up vector
        #   avg_up = poses[:, :3, 1].mean(0)
        #   avg_up = avg_up / np.linalg.norm(avg_up)
        #   ind_up = np.argmax(np.abs(avg_up))
        #   up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

        #   # lookat vector is calculated a position to center.
        #   center = focus_point_fn(poses)

        cams = []
        for i in range(len(indices) - 1):
            cam = self.train_cameras[resolution_scale][indices[i]]
            cam_next = self.train_cameras[resolution_scale][indices[i + 1]]

            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)

            W2C_next = getWorld2View2(cam_next.R, cam_next.T)
            C2W_next = np.linalg.inv(W2C_next)

            start_translation = C2W[:3, 3]
            end_translation = C2W_next[:3, 3]
            start_rotation = C2W[:3, :3]
            end_rotation = C2W_next[:3, :3]

            intermidiate_T, intermidiate_R = interpolate_views(
                start_translation, end_translation, start_rotation, end_rotation, 120
            )

            for t, r in zip(intermidiate_T, intermidiate_R):
                # create cache of gpu
                torch.cuda.empty_cache()

                c2w = np.concatenate([r, t[:, None]], axis=1)
                bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
                c2w = np.concatenate([c2w, bottom], -2)

                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                cam_copied = copy.deepcopy(cam)
                # del cam_new.original_image
                # cam_new.image_height = int(cam_new.image_height / 2) * 2
                # cam_new.image_width = int(cam_new.image_width / 2) * 2
                # cam_new.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
                # # cam.world_view_transform = torch.from_numpy(c2w.T).float().cuda()
                # cam_new.full_proj_transform = (
                #     cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
                # ).squeeze(0)
                # cam_new.camera_center = cam_new.world_view_transform.inverse()[3, :3]

                cam_new = Camera(
                    colmap_id=cam.colmap_id,
                    R=R,
                    T=T,
                    FoVx=cam.FoVx,
                    FoVy=cam.FoVy,
                    image=None,
                    gt_alpha_mask=None,
                    image_name=cam.image_name,
                    uid=cam.uid,
                    data_device=self.args.data_device,
                )
                cam_new.image_height = int(cam_copied.image_height / 2) * 2
                cam_new.image_width = int(cam_copied.image_width / 2) * 2

                del cam_copied

                cams.append(cam_new)
        return cams


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def slerp_custom(quat1, quat2, t):
    dot_product = np.dot(quat1, quat2)

    if dot_product < 0.0:
        quat1 = -quat1
        dot_product = -dot_product

    DOT_THRESHOLD = 0.9995
    if dot_product > DOT_THRESHOLD:
        result = quat1 + t * (quat2 - quat1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t
    quat2_perp = quat2 - quat1 * dot_product
    quat2_perp = quat2_perp / np.linalg.norm(quat2_perp)

    return quat1 * np.cos(theta) + quat2_perp * np.sin(theta)


def interpolate_views(start_translation, end_translation, start_rotation, end_rotation, num_points):
    # Generate interpolation fractions
    t_values = np.linspace(0, 1, num_points)[1:-1]

    # Convert rotation matrices to quaternion
    start_rot_quat = R.from_matrix(start_rotation).as_quat()
    end_rot_quat = R.from_matrix(end_rotation).as_quat()

    # Interpolate translations and rotations
    interpolated_translations = [(1 - t) * start_translation + t * end_translation for t in t_values]
    interpolated_rotations = [slerp_custom(start_rot_quat, end_rot_quat, t) for t in t_values]

    return interpolated_translations, [R.from_quat(rot).as_matrix() for rot in interpolated_rotations]


# def get_spiral_path(
#     steps: int = 30,
#     radius: Optional[float] = None,
#     radiuses: Optional[Tuple[float]] = None,
#     rots: int = 2,
#     zrate: float = 0.5,
# ) -> Cameras:
#     """
#     Returns a list of camera in a spiral trajectory.

#     Args:
#         camera: The camera to start the spiral from.
#         steps: The number of cameras in the generated path.
#         radius: The radius of the spiral for all xyz directions.
#         radiuses: The list of radii for the spiral in xyz directions.
#         rots: The number of rotations to apply to the camera.
#         zrate: How much to change the z position of the camera.

#     Returns:
#         A spiral camera path.
#     """

#     assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."

#     # assert camera.ndim == 1, "We assume only one batch dim here"
#     if radius is not None and radiuses is None:
#         rad = torch.tensor([radius] * 3, device=camera.device)
#     elif radiuses is not None and radius is None:
#         rad = torch.tensor(radiuses, device=camera.device)

#     up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
#     focal = torch.min(camera.fx[0], camera.fy[0])
#     target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

#     c2w = camera.camera_to_worlds[0]
#     c2wh_global = pose_utils.to4x4(c2w)

#     local_c2whs = []
#     for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
#         center = (
#             torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
#         )
#         lookat = center - target
#         c2w = camera_utils.viewmatrix(lookat, up, center)
#         c2wh = pose_utils.to4x4(c2w)
#         local_c2whs.append(c2wh)

#     new_c2ws = []
#     for local_c2wh in local_c2whs:
#         c2wh = torch.matmul(c2wh_global, local_c2wh)
#         new_c2ws.append(c2wh[:3, :4])
#     new_c2ws = torch.stack(new_c2ws, dim=0)

#     times = None
#     if camera.times is not None:
#         times = torch.linspace(0, 1, steps)[:, None]
#     return Cameras(
#         fx=camera.fx[0],
#         fy=camera.fy[0],
#         cx=camera.cx[0],
#         cy=camera.cy[0],
#         camera_to_worlds=new_c2ws,
#         times=times,
#     )
#     )
