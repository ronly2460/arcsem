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
import uuid
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from icecream import ic
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from utils.loss_utils import l1_loss, ssim

sys.path.append("/home/hpc/iwi9/iwi9007h/gaussian-splatting")
from ref_loss import NNFMLoss

# Load Depth Anything Model
sys.path.append("/home/hpc/iwi9/iwi9007h/Ref-NPR/Depth-Anything")
from depth_anything.dpt import DepthAnything

sys.path.append("/home/hpc/iwi9/iwi9007h/gaussian-splatting/utils")
import wandb

from utils.generate_depth import generate_depth


def write_command(args):
    model_params = lp.extract(args)
    command = " ".join(sys.argv)

    file_path = os.path.join(model_params.model_path, "command_log.txt")
    print(file_path)

    with open(file_path, "a") as file:
        # get folder name
        file.write(command + "\n")

    exp_name = os.path.basename(os.path.normpath(model_params.model_path))
    with open("command_log.txt", "a") as file:
        # get folder name
        file.write(exp_name + " " + command + "\n\n")


def get_size(resolution: int):
    # return (3, 256, 384), (3, 256, 384)
    return (3, 2048, 3072), (3, 2048, 3072)

    if resolution == 1:
        return (3, 2048, 3072), (1, 1920, 2800)
    else:
        return (3, 512, 768), (1, 480, 700)


def save_img(img: np.array, exp_name: str, name: str, is_depth: bool = False) -> None:
    print("saving")
    save_path = os.path.join("/home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/outputs/imgs", exp_name, name + ".png")
    # makeir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if is_depth:
        img = (img - img.min()) / (img.max() - img.min())

    img = torch.clamp(torch.tensor(img), 0, 1)
    img = Image.fromarray((img.detach().cpu().numpy() * 255).astype(np.uint8))
    img.save(save_path)
    return


def resize_img(img, mask_size, factor):
    ori_shape = img.shape

    down_size = (int(mask_size[1] / factor), int(mask_size[2] / factor))
    resize_transform = transforms.Resize(down_size)

    if img.shape[0] == 1:
        img = img.squeeze(0)
    if img.shape[2] == 3:
        img = img.permute(2, 0, 1)

    img = resize_transform(img)

    if ori_shape[2] == 3:
        img = img.permute(1, 2, 0)
    if ori_shape[0] == 1:
        img = img.unsqueeze(0)

    return img


def load_style_imgs(config_path, cams, mask_size):
    style_imgs = []
    style_imgs_gray = []
    style_cams_idx = []
    with open(config_path) as fp:
        style_dict = json.load(fp)

    for i in range(len(style_dict["style_img"])):
        path = os.path.join("/home/hpc/iwi9/iwi9007h/Ref-NPR", style_dict["style_img"][i])
        style_image = np.array(Image.open(path).convert("RGB")) / 255.0
        style_image = torch.tensor(style_image, dtype=torch.float64).cuda().permute(2, 0, 1)

        cam_idx = style_dict["tmpl_idx_train"][i]
        style_cam = cams[cam_idx]

        if style_cam.gt_alpha_mask is not None:
            style_image = style_image[:, style_cam.gt_alpha_mask == True].reshape(3, mask_size[1], mask_size[2]).float()

        style_image_gray = style_cam.original_image.cuda()
        ic(style_image_gray.shape)

        if style_cam.gt_alpha_mask is not None:
            style_image_gray = (
                style_image_gray[:, style_cam.gt_alpha_mask == True].reshape(3, mask_size[1], mask_size[2]).float()
            )
        ic(style_image_gray.shape)

        style_image = style_image.permute(1, 2, 0)
        style_image_gray = style_image_gray.unsqueeze(0)

        style_cams_idx.append(cam_idx)
        style_imgs.append(style_image)
        style_imgs_gray.append(style_image_gray)

    return style_imgs, style_imgs_gray, style_cams_idx


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):
    first_iter = 0
    gaussians = GaussianModel(
        dataset.sh_degree, is_color_decoder=args.affine_color_decoder, is_train_color=opt.train_color, n_image=306
    )
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # pollen
    factor = 4
    opt.vgg_blocks = [2, 3, 4]
    opt.content_weight = 5e-3

    nnfm_loss_fn = NNFMLoss(device="cuda", layer=19)

    wandb.init(
        project=args.dataset,
        mode="online",
        name=f"{Path(scene.model_path).name}",
        dir="/home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/outputs/wanb",
    )

    if opt.train_color and "depth_loss" in args.loss_names:
        depth_anything = (
            DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format("vitl")).to("cuda").eval()
        )

    image_size, mask_size = get_size(args.resolution)

    datetime_str = datetime.now().strftime("%m%d_%H%M")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if args.train_color:
        cams = scene.getTrainCameras().copy()
        config_path = f"/home/hpc/iwi9/iwi9007h/Ref-NPR/data/ref_case/{args.ref_case}/data_config.json"
        style_imgs, style_imgs_gray, style_cams_idx = load_style_imgs(config_path, cams, mask_size)
        ic(style_imgs_gray[0].shape)

        if args.resolution == 1:
            if args.dataset == "pollen":
                style_imgs_tmp = [resize_img(img, mask_size, factor) for img in style_imgs]
                style_imgs_gray_tmp = [resize_img(img, mask_size, factor) for img in style_imgs_gray]
                ic(style_imgs_gray_tmp[0].shape, style_imgs_tmp[0].shape)
                nnfm_loss_fn.preload_golden_template(style_imgs_gray_tmp, style_imgs_tmp, blocks=opt.vgg_blocks)
            elif args.dataset == "caterpillar":
                new_size = (3, style_imgs[0].shape[0], style_imgs[0].shape[1])
                style_imgs_tmp = [resize_img(img, new_size, factor) for img in style_imgs]
                style_imgs_gray_tmp = [resize_img(img, new_size, factor) for img in style_imgs_gray]
                style_imgs_tmp = [img.float() for img in style_imgs_tmp]
                style_imgs_gray_tmp = [img.float() for img in style_imgs_gray_tmp]
                nnfm_loss_fn.preload_golden_template(style_imgs_gray_tmp, style_imgs_tmp, blocks=opt.vgg_blocks)
        else:
            if args.dataset == "pollen":
                nnfm_loss_fn.preload_golden_template(style_imgs_gray_tmp, style_imgs_tmp, blocks=opt.vgg_blocks)
            elif args.dataset == "caterpillar":
                new_size = (3, style_imgs[0].shape[0], style_imgs[0].shape[1])
                style_imgs = [resize_img(img, new_size, 1) for img in style_imgs]
                ic(style_imgs[0].shape, style_imgs_gray[0].shape)
                style_imgs = [img.float() for img in style_imgs]
                style_imgs_gray = [img.float() for img in style_imgs_gray]
                nnfm_loss_fn.preload_golden_template(style_imgs_gray, style_imgs, blocks=opt.vgg_blocks)

        # load related_rays
        corr_path = f"/home/hpc/iwi9/iwi9007h/Ref-NPR/exps/refnpr/previous_paper/{args.ref_case}/color_corr.pt"
        corr_path = f"/home/hpc/iwi9/iwi9007h/Ref-NPR/exps/refnpr/{args.ref_case}/color_corr.pt"
        related_rays_gt = torch.load(corr_path).reshape(25, image_size[1], image_size[2], 3).permute(0, 3, 1, 2).cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    online_iter = (opt.iterations - first_iter) / 2 + first_iter
    for iteration in range(first_iter, opt.iterations + 1):

        # release memory
        torch.cuda.empty_cache()

        if opt.train_color:
            gaussians.fix_opacity_params()

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        cam_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack[cam_idx]

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, args=args)
        pred_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        depth = render_pkg["surf_depth"]
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]

        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.gt_alpha_mask
        if viewpoint_cam.gt_alpha_mask is not None:
            mask = mask.cuda()

            gt_image = gt_image[:, mask == True].reshape(3, mask_size[1], mask_size[2])
            pred_image = pred_image[:, mask == True].reshape(3, mask_size[1], mask_size[2])

            depth = depth[:, mask == True].reshape(mask_size)
            rend_dist = rend_dist[:, mask == True].reshape(mask_size)
            rend_normal = rend_normal[:, mask == True].reshape(3, mask_size[1], mask_size[2])
            surf_normal = surf_normal[:, mask == True].reshape(3, mask_size[1], mask_size[2])

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        if iteration % args.save_img_interval == 0:
            exp_name = os.path.basename(os.path.normpath(dataset.model_path))
            save_img(
                pred_image.permute(1, 2, 0), f"{datetime_str}_{exp_name}", f"{iteration}_pred_image", is_depth=False
            )
            save_img(depth[0], f"{datetime_str}_{exp_name}", f"{iteration}_pred_depth", is_depth=True)

        if opt.train_color:
            if cam_idx in style_cams_idx:

                gt_style_img = style_imgs[style_cams_idx.index(cam_idx)].permute(2, 0, 1)
                L1_loss_color_gt = opt.style_gt_weight * l1_loss(pred_image, gt_style_img)
                loss = L1_loss_color_gt
                total_loss = loss

                wandb.log({"L1_loss_color_1": L1_loss_color_gt})
            else:
                depth_pred = None
                depth_gt = None

                pred_image_gray = torchvision.transforms.functional.rgb_to_grayscale(pred_image, num_output_channels=3)
                gt_image_gray = torchvision.transforms.functional.rgb_to_grayscale(gt_image, num_output_channels=3)

                if "depth_loss" in args.loss_names:
                    depth_pred = generate_depth(
                        pred_image_gray.permute(1, 2, 0).detach().cpu().numpy(),
                        "cuda",
                        mask_size[1],
                        mask_size[2],
                        depth_anything,
                    )
                    depth_gt = generate_depth(
                        gt_image_gray.permute(1, 2, 0).detach().cpu().numpy(),
                        "cuda",
                        mask_size[1],
                        mask_size[2],
                        depth_anything,
                    )
                else:
                    depth_pred = None
                    depth_gt = None

                if args.resolution == 1:
                    scale_factor = 1 / factor
                    # ic(scale_factor)
                    # breakpoint()
                    resize_lambda = lambda x: F.interpolate(
                        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
                    )
                    pred_image_outputs = resize_lambda(pred_image.unsqueeze(0))
                    gt_image_outputs = resize_lambda(gt_image.unsqueeze(0))
                else:
                    pred_image_outputs = pred_image.unsqueeze(0)
                    gt_image_outputs = gt_image.unsqueeze(0)

                loss_dict = nnfm_loss_fn(
                    outputs=pred_image_outputs,
                    styles=None,
                    blocks=opt.vgg_blocks,
                    loss_names=args.loss_names,
                    contents=gt_image_outputs,
                    depths_gt=depth_gt,
                    depths_pred=depth_pred,
                )

                # related_rays
                if mask is not None:
                    related_rays = related_rays_gt[cam_idx][:, mask == True].reshape(3, mask_size[1], mask_size[2])
                else:
                    related_rays = related_rays_gt[cam_idx].reshape(3, mask_size[1], mask_size[2])

                pred_image_related = pred_image[related_rays != -1]
                related_rays_ = related_rays[related_rays != -1]

                L1_loss_color = (
                    opt.l1_loss_related_rays * l1_loss(pred_image_related, related_rays_) / len(related_rays_) * 4800000
                )
                loss = L1_loss_color

                # tv_loss
                w_variance = torch.mean(torch.pow(pred_image[:, :, :-1] - pred_image[:, :, 1:], 2))
                h_variance = torch.mean(torch.pow(pred_image[:, :-1, :] - pred_image[:, 1:, :], 2))
                img_tv_loss = (h_variance + w_variance) / 2.0

                # tv loss
                img_tv_loss = opt.tv_weight * img_tv_loss

                loss_dict["color_patch"] = opt.patch_weight * loss_dict["color_patch"]
                loss_dict["tcm_loss"] = opt.tcm_weight * loss_dict["tcm_loss"]
                total_loss = L1_loss_color + loss_dict["tcm_loss"] + loss_dict["color_patch"] + img_tv_loss

                wandb.log({"2_tcm_loss": loss_dict["tcm_loss"]})
                wandb.log({"2_color_patch": loss_dict["color_patch"]})

                if "depth_loss" in args.loss_names:
                    depth_loss = opt.depth_weight * loss_dict["depth_loss"]
                    total_loss += depth_loss

                wandb.log({"2_l1_loss_color": L1_loss_color})
                wandb.log({"2_total_loss": total_loss})
        else:
            Ll1 = l1_loss(pred_image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image))

            # loss
            total_loss = loss + dist_loss + normal_loss
            wandb.log({"dist_loss": dist_loss})
            wandb.log({"normal_loss": normal_loss})
            wandb.log({"total_loss": total_loss})

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.update(10)

            if iteration % 1000 == 0:
                wandb.log({"image_name": wandb.Image(pred_image, caption="This is a caption")})

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # if args.train_color and args.affine_color_decoder:
                #     gaussians.optimizer_affine_decoder.step()
                #     gaussians.optimizer_affine_decoder.zero_grad(set_to_none = True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                print(scene.model_path)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    wandb.finish()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    write_command(args)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/reg_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        try:
                            rend_alpha = render_pkg["rend_alpha"]
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(
                                config["name"] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                rend_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                surf_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config["name"] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                rend_alpha[None],
                                global_step=iteration,
                            )

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(
                                config["name"] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                rend_dist[None],
                                global_step=iteration,
                            )
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="pollen")

    parser.add_argument("--ref_case", type=str, default="pollen_color_artist_multiple")
    parser.add_argument("--save_img_interval", type=int, default=1000)

    # parser.add_argument("--direction", type=str)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args,
    )

    # All done
    print("\nTraining complete.")
