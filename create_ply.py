import os

import numpy as np
from plyfile import PlyData, PlyElement
from typing import NamedTuple


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


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


path = "/home/hpc/iwi9/iwi9007h/2d-gaussian-splatting/data/dataset/gray/"

txt_path = os.path.join(path, "sparse/0/points3D.txt")
ply_path = os.path.join(path, "sparse/0/points3D.ply")
if not os.path.exists(ply_path):
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    xyz, rgb, _ = read_points3D_text(txt_path)

    print("colmap points shape", xyz.shape, rgb.shape)
    print("stats : ", np.min(xyz, axis=0), np.max(xyz, axis=0), np.mean(xyz, axis=0), np.std(xyz, axis=0))
    storePly(ply_path, xyz, rgb)
try:
    pcd = fetchPly(ply_path)
except Exception as e:
    print(f"Error fetching PLY: {e}")
    pcd = None
