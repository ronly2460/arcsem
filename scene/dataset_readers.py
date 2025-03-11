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

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
import open3d as o3d
import torch
from icecream import ic
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB

POSE_SCALE = 1.0


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    mask: np.array = None
    intrinsics: np.array = None
    extrinsics: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    scale_factor = 8
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height * scale_factor
        width = intr.width * scale_factor

        ic(height, width)
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        focal_length_x = intr.params[0] * scale_factor
        focal_length_y = intr.params[0] * scale_factor
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # replace images to images_8
        image_path = image_path.replace("images", "down_1")
        image = Image.open(image_path)
        # make it three channels
        image = image.convert("RGB")

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            mask=None,
            depth=None,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    print(xyz.shape, normals.shape, rgb.shape)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=10):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir)
    )
    cam_infos = []
    style_list = [215, 335, 425]
    for _, c in enumerate(cam_infos_unsorted):
        num = int(c.image_name.split("_")[-1])
        if num < 200:
            continue
        if num % 10 == 0:
            cam_infos.append(c)
        if num in style_list:
            cam_infos.append(c)

    # only keep the images that can be devided by 10.
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 1 or idx in style_list]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 1 and (idx not in style_list)]

    # sort train_cam_infos by image_name
    train_cam_infos = sorted(train_cam_infos, key=lambda x: x.image_name)
    test_cam_infos = sorted(test_cam_infos, key=lambda x: x.image_name)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

    return scene_info


def readRealityCaptureInfo(path, eval, args, is_ply=True):
    if is_ply:
        ply_path = os.path.join(path, "Processing", "RC", "project_rc_undist.ply")
        pcd = fetchPly(ply_path)
    else:
        ply_path = os.path.join(path, "points3d.ply")

        num_pts = 100000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    cam_infos, translation = readCamerasFromRealityCapture(path, args.resolution)
    # cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    # applied_transform[:3, 3] = -translation

    # print(pcd.points)
    # pcd = o3d.io.read_point_cloud(ply_path)
    points3D = np.array(pcd.points)
    points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]
    # points3D = points3D * 0.025
    # points3D = points3D - translation.numpy()
    pcd = BasicPointCloud(points=points3D, colors=pcd.colors, normals=pcd.normals)
    # o3d.io.write_point_cloud("test.ply", pcd)

    # break
    # apply scale to points
    # pcd.points = pcd.points * 0.0025
    storePly("test.ply", pcd.points, pcd.colors)

    if eval:
        # TODO: fixed
        train_cam_infos = [c for idx, c in enumerate(cam_infos)]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx == 10]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    ic(len(train_cam_infos))
    # nerf_normalization = {}
    # nerf_normalization["radius"] = 20
    # nerf_normalization["translate"] = [0, 0, 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ic(nerf_normalization)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromRealityCapture(path, resolution):
    cam_infos = []

    if resolution == 1:
        calib_path = os.path.join(path, "Rdy2Use", "orig", "calib", "rc_undist", "torch-ngp")
    elif resolution == 4:
        calib_path = os.path.join(path, "Rdy2Use", "down4", "calib", "rc_undist", "torch-ngp")
    else:
        raise ValueError(f"Resolution {resolution} not supported")

    with open(os.path.join(calib_path, "transforms.json"), "r") as fp:
        meta = json.load(fp)

    focal_x = meta["fl_x"]
    focal_y = meta["fl_y"]
    height = meta["h"]
    width = meta["w"]

    translation = None

    # load poses
    poses = []
    for idx, frame in enumerate(meta["frames"]):
        poses.append(np.array(frame["transform_matrix"]))
    poses = np.array(poses).astype(np.float32)

    # mean_origin = poses[..., :3, 3].mean(axis=0)
    # translation = focus_of_attention(torch.Tensor(poses), torch.Tensor(mean_origin))

    # transform = np.eye(4)
    # transform[:3, 3] = -translation.cpu()
    # transform = transform[:3, :]
    # poses = transform @ poses

    # poses[:, :3, 3] *= POSE_SCALE

    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    for idx, frame in enumerate(meta["frames"]):

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = poses[idx]

        c2w[:3, 1:3] *= -1
        # c2w[:3, 3] *= 0.0025

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # Load image data
        image_path = os.path.join(calib_path, frame["file_path"])
        mask_path = os.path.join(calib_path, frame["mask_path"])
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        mask = np.array(Image.open(mask_path))
        mask = mask / 255.0

        # process image data
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0

        # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4]
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4]
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        FovY = focal2fov(focal_y, height)
        FovX = focal2fov(focal_x, width)

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                mask=mask,
                depth=None,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos, translation


def focus_of_attention(poses, initial_focus):
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt


sceneLoadTypeCallbacks = {"Colmap": readColmapSceneInfo, "Blender": readNerfSyntheticInfo, "RC": readRealityCaptureInfo}
