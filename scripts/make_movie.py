import glob
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# /home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/output/c9adbdd9-e
# get custom_horizontal_1, 2, 3, 4, 5
file_path = "/home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/output/c9adbdd9-e"
custom_dirs = [os.path.join(file_path, f"custom_horizontal_{i}", "ours_80000", "renders") for i in range(1, 5)]

mask_path = "/home/hpc/iwi9/iwi9007h/Ref-NPR/data/rc/pollen_artist/Rdy2Use/orig/masks/rc_undist/train/Pollen-center-005-2_mask.png"
mask = torch.tensor(np.array(Image.open(mask_path))).float()
mask = mask / 255.0

# mask = viewpoint_cam.gt_alpha_mask.cuda()


# load images under each custom_dir
images = []
for custom_dir in custom_dirs:
    # load images
    files = sorted(glob.glob(os.path.join(custom_dir, "*.png")))
    images.extend(files)

# take two arguments, video_name and data_path by uisng argparse
save_path = "outputs/test_color_mask.mp4"
factor = 1

with imageio.get_writer(save_path, fps=60) as writer:
    # 画像を順番に動画ファイルに書き込む
    for f in tqdm(images):
        img = Image.open(f)
        # downscale = lambda img: img.resize((img.size[0] // factor, img.size[1] // factor))
        # img = downscale(img)
        img = np.array(img)
        img = img[mask == True, :].reshape(1920, 2800, 3)
        writer.append_data(img)

    for f in tqdm(reversed(images)):
        img = Image.open(f)
        # downscale = lambda img: img.resize((img.size[0] // factor, img.size[1] // factor))
        # img = downscale(img)
        img = np.array(img)
        img = img[mask == True, :].reshape(1920, 2800, 3)
        writer.append_data(img)
