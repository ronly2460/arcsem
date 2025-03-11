import numpy as np
from PIL import Image
import os

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image

idx_to_str = lambda idx: str(0).zfill(5)

input_dir = "/home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/output/8ee5c65c-3/train/ours_60000"
depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
depth_frame = load_img(depth_file)

print(depth_frame.shape)

print(depth_frame[:10, :10])

                names_intermidiate = [
                    #     "Pollen-center-005-10", #1
                    #     "Pollen-center-005-9",
                    #     "Pollen-center-005-8",
                    #     "Pollen-center-005-7",
                    # "Pollen-center-005-6",  # 2
                    # "Pollen-center-005-5",
                    # "Pollen-center-005-4",
                    # "Pollen-center-005-3",
                    # "Pollen-center-005-2",
                    "Pollen-center-005+12",  # 3
                    # "Pollen-center-005+13",
                    # "Pollen-center-005+14",
                    # "Pollen-center-005+15",
                    # "Pollen-center-005+16",
                    # "Pollen-center-005+17",
                    # "Pollen-center-005+18",
                    # "Pollen-center-005+19",
                    # "Pollen-center-005+20",
                ]
