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

import os
from argparse import ArgumentParser
from os import makedirs

import open3d as o3d
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh, to_cam_open3d
from utils.render_utils import create_videos, generate_path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_custom", action="store_true")
    parser.add_argument("--skip_fancy", action="store_true")

    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF")
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF")
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF")
    parser.add_argument("--num_cluster", default=50, type=int, help="Mesh: number of connected clusters to export")
    parser.add_argument("--unbounded", action="store_true", help="Mesh: using unbounded mode for meshing")
    parser.add_argument("--mesh_res", default=1024, type=int, help="Mesh: resolution for unbounded mesh extraction")
    # parser.add_argument("--direction", default="horizontal", type=str)
    # parser.add_argument("--train_color", action="store_true", help='Render: use training color')
    parser.add_argument("--start_checkpoint", type=str, help="Path to the checkpoint to start from")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(
        dataset.sh_degree, is_color_decoder=args.affine_color_decoder, is_train_color=args.train_color
    )

    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, optim)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, "train", "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, "test", "ours_{}".format(scene.loaded_iter))
    custom_dir = os.path.join(args.model_path, f"custom_{args.direction}_3", "ours_{}".format(scene.loaded_iter))
    fancy_dir = os.path.join(args.model_path, f"fancy", "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, args=args, bg_color=bg_color)

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)

    if (not args.skip_custom) and (len(scene.getCustomCameras()) > 0):
        print("export custom images ...")
        os.makedirs(custom_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getCustomCameras(), is_test=True)
        gaussExtractor.export_image(custom_dir, is_custom=True)

    if (not args.skip_fancy) and (len(scene.getFancyCameras()) > 0):
        print("export fancy images ...")
        os.makedirs(fancy_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getFancyCameras(), is_test=True)
        gaussExtractor.export_image(fancy_dir)

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, "traj", "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 20
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        # create_videos(base_dir=traj_dir, input_dir=traj_dir, out_name="render_traj", num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = "fuse_unbounded.ply"
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = "fuse.ply"
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(
                voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
            )

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace(".ply", "_post.ply")), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace(".ply", "_post.ply"))))
