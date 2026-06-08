#!/usr/bin/env python3
"""
Render a specific frame with Gaussians inside obj_id 4's bbox shifted by -0.1 in x coordinate.
Uses bbox reading method from object_nerf.
"""

import torch
import numpy as np
from pathlib import Path
import imageio
import tyro
from dataclasses import dataclass
import copy
import sys

sys.path.insert(0, str(Path(__file__).parent / "examples"))

from datasets.colmap import Dataset, Parser
from gsplat.rendering import rasterization


def read_axis_align_matrix(scene_txt_path):
    """Read axis alignment matrix from ScanNet scene .txt file"""
    lines = open(scene_txt_path).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            return np.array(axis_align_matrix).reshape(4, 4)
    raise ValueError("axisAlignment not found in scene file")


def read_bbox_for_instance(bbox_npy_path, instance_id):
    """Read bbox bounds for a specific instance ID from ScanNet bbox file"""
    scene_bbox = np.load(bbox_npy_path)
    for b in scene_bbox:
        if b[6] != instance_id:
            continue
        length = np.array([b[3], b[4], b[5]]) * 0.5
        center = np.array([b[0], b[1], b[2]])
        bbox_bounds = np.array([center - length, center + length])
        bbox_c = center
        return bbox_bounds, bbox_c
    raise ValueError(f"Instance {instance_id} not found in bbox file")


def transform_xyz_to_bbox_coordinates(xyz_normalized, inv_transform, axis_align_mat):
    """Transform xyz from gsplat normalized coordinate to bbox coordinate

    Args:
        xyz_normalized: points in gsplat's normalized space [N, 3]
        inv_transform: inverse of gsplat's normalization transform [4, 4]
        axis_align_mat: ScanNet's axis alignment matrix [4, 4]
    """
    if isinstance(xyz_normalized, torch.Tensor):
        xyz_normalized = xyz_normalized.detach().cpu().numpy()

    # Step 1: Transform from normalized space back to original COLMAP space
    xyz_homogeneous = np.concatenate([xyz_normalized, np.ones((xyz_normalized.shape[0], 1))], axis=1)
    xyz_original = (inv_transform @ xyz_homogeneous.T).T[:, :3]

    # Step 2: Apply ScanNet axis alignment to get bbox coordinates
    xyz_homogeneous = np.concatenate([xyz_original, np.ones((xyz_original.shape[0], 1))], axis=1)
    xyz_bbox = (axis_align_mat @ xyz_homogeneous.T).T[:, :3]

    return xyz_bbox


def check_xyz_in_bounds(xyz_bbox, bbox_bounds, bbox_enlarge=0.0):
    """Check which points are inside the bbox"""
    bbox_bounds = copy.deepcopy(bbox_bounds)
    if bbox_enlarge != 0:
        bbox_bounds[0] -= bbox_enlarge
        bbox_bounds[1] += bbox_enlarge

    x_min, y_min, z_min = bbox_bounds[0]
    x_max, y_max, z_max = bbox_bounds[1]

    in_x = np.logical_and(xyz_bbox[:, 0] >= x_min, xyz_bbox[:, 0] <= x_max)
    in_y = np.logical_and(xyz_bbox[:, 1] >= y_min, xyz_bbox[:, 1] <= y_max)
    in_z = np.logical_and(xyz_bbox[:, 2] >= z_min, xyz_bbox[:, 2] <= z_max)
    in_bounds = np.logical_and(in_x, np.logical_and(in_y, in_z))

    return in_bounds


@dataclass
class Config:
    data_dir: str = "data/scannet_0113_00_train"
    result_dir: str = "results/scannet_0113_00_guard05a"
    checkpoint: str = "ckpt_29999_rank0.pt"
    target_image: str = "160.png"
    target_obj_id: int = 6
    shift_x: float = 0.5
    shift_y: float = 0.5
    shift_z: float = 0.0

    # ScanNet paths
    scene_id: str = "scene0113_00"
    scans_dir: str = "/nas1/home/zhangjunyi/demo_data_object_nerf/scannet_object_nerf_data/scans"
    bbox_dir: str = "/nas1/home/zhangjunyi/demo_data_object_nerf/scannet_object_nerf_data/scannet_train_detection_data"

    # From colmap parser
    inv_transform: np.ndarray = None

    output_path: str = "shifted.png"


def main(cfg: Config):
    device = torch.device("cuda:0")

    # Load dataset to get normalization transform
    print(f"Loading dataset from {cfg.data_dir}...")
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=1,
        normalize=True,
        test_every=8,
    )
    dataset = Dataset(parser, split="train")

    # Get the normalization transform applied by gsplat
    # transform is T2 @ T1 where T1 is similarity and T2 is align_principal_axes
    # To go from normalized space back to original: inv(transform)
    transform = parser.transform  # [4, 4], maps original -> normalized
    inv_transform = np.linalg.inv(transform)
    print(f"Transform matrix shape: {transform.shape}")

    # Extract scale factor from transform (it's embedded in the matrix)
    # The scale is in the rotation part
    scale_factor = np.linalg.norm(transform[:3, 0])
    print(f"Extracted scale factor: {scale_factor}")

    # Find the target image index
    target_idx = int(cfg.target_image.split(".")[0])
    for idx in range(len(dataset)):
        data = dataset[idx]
        if idx == 0:
            print(f"Available keys: {data.keys()}")

        if "image_name" in data:
            img_name = data["image_name"]
        elif "image_id" in data:
            img_name = str(data["image_id"]) + ".png"
        else:
            img_name = f"{idx}.png"

        if img_name == cfg.target_image or str(idx) + ".png" == cfg.target_image:
            target_idx = idx
            print(f"Found {cfg.target_image} at index {idx}")
            break

    if target_idx is None:
        raise ValueError(f"Could not find {cfg.target_image} in dataset")

    # Load the target frame data
    data = dataset[target_idx]
    camtoworlds = data["camtoworld"][None]  # [1, 4, 4]
    Ks = data["K"][None]  # [1, 3, 3]

    # Get image dimensions
    image = data["image"]  # [H, W, C]
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    # Load checkpoint
    ckpt_path = Path(cfg.result_dir) / "ckpts" / cfg.checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract model parameters
    means = ckpt["splats"]["means"].to(device)  # [N, 3]
    quats = ckpt["splats"]["quats"].to(device)  # [N, 4]
    scales = ckpt["splats"]["scales"].to(device)  # [N, 3]
    opacities = ckpt["splats"]["opacities"].to(device)  # [N]

    # Load full spherical harmonics coefficients
    sh0 = ckpt["splats"]["sh0"].to(device)  # [N, 1, 3]
    shN = ckpt["splats"]["shN"].to(device) if "shN" in ckpt["splats"] else None  # [N, 15, 3]

    # Concatenate sh0 and shN to get full SH coefficients [N, 16, 3] for degree 3
    if shN is not None:
        colors = torch.cat([sh0, shN], dim=1)  # [N, 16, 3]
        sh_degree = 3
        print(f"Loaded {means.shape[0]} Gaussians with SH degree {sh_degree}")
    else:
        colors = sh0
        sh_degree = 0
        print(f"Loaded {means.shape[0]} Gaussians with SH degree {sh_degree} (base color only)")

    # Read bbox information
    scene_txt_path = Path(cfg.scans_dir) / cfg.scene_id / f"{cfg.scene_id}.txt"
    bbox_npy_path = Path(cfg.bbox_dir) / f"{cfg.scene_id}_bbox.npy"

    print(f"Reading axis alignment from {scene_txt_path}")
    axis_align_mat = read_axis_align_matrix(scene_txt_path)

    print(f"Reading bbox from {bbox_npy_path} for instance {cfg.target_obj_id}")
    bbox_bounds, bbox_c = read_bbox_for_instance(bbox_npy_path, cfg.target_obj_id)
    print(f"Bbox bounds: {bbox_bounds}")
    print(f"Bbox center: {bbox_c}")

    # Transform Gaussian means to bbox coordinates
    print("Transforming Gaussians to bbox coordinates...")
    xyz_bbox = transform_xyz_to_bbox_coordinates(
        means.cpu(),
        inv_transform,
        axis_align_mat
    )

    # Check which Gaussians are inside the bbox
    in_bbox = check_xyz_in_bounds(xyz_bbox, bbox_bounds, bbox_enlarge=0.0)
    num_in_bbox = in_bbox.sum()
    print(f"Found {num_in_bbox} Gaussians inside bbox of obj_id={cfg.target_obj_id}")

    # Create duplicated object: keep original + add shifted copy
    bbox_mask = torch.from_numpy(in_bbox).to(device)

    # Extract the Gaussians inside bbox
    bbox_means = means[bbox_mask].clone()
    bbox_quats = quats[bbox_mask].clone()
    bbox_scales = scales[bbox_mask].clone()
    bbox_opacities = opacities[bbox_mask].clone()
    bbox_colors = colors[bbox_mask].clone()

    # Shift the duplicated copy
    bbox_means[:, 0] += cfg.shift_x
    bbox_means[:, 1] += cfg.shift_y
    bbox_means[:, 2] += cfg.shift_z

    # Concatenate original + shifted copy
    means_with_dup = torch.cat([means, bbox_means], dim=0)
    quats_with_dup = torch.cat([quats, bbox_quats], dim=0)
    scales_with_dup = torch.cat([scales, bbox_scales], dim=0)
    opacities_with_dup = torch.cat([opacities, bbox_opacities], dim=0)
    colors_with_dup = torch.cat([colors, bbox_colors], dim=0)

    print(f"Duplicated {num_in_bbox} Gaussians with shift (x={cfg.shift_x}, y={cfg.shift_y}, z={cfg.shift_z})")
    print(f"Total Gaussians: {means.shape[0]} -> {means_with_dup.shape[0]}")

    # Prepare rendering parameters
    camtoworlds = camtoworlds.to(device)
    Ks = Ks.to(device)

    print(f"Rendering {height}x{width} image with duplicated object...")

    # Render with original + duplicated object
    renders, alphas, info = rasterization(
        means=means_with_dup,
        quats=quats_with_dup / quats_with_dup.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_with_dup),
        opacities=torch.sigmoid(opacities_with_dup),
        colors=colors_with_dup,
        sh_degree=sh_degree,
        viewmats=torch.linalg.inv(camtoworlds),  # [1, 4, 4]
        Ks=Ks,  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
    )

    # Extract RGB
    rendered_img = renders[0, ..., :3]  # [H, W, 3]
    rendered_img = rendered_img.clamp(0, 1)
    rendered_img = (rendered_img.cpu().numpy() * 255).astype(np.uint8)

    # Save output
    output_path = Path(cfg.result_dir) / cfg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, rendered_img)
    print(f"Saved to {output_path}")

    # Also render original for comparison (if not already exists)
    orig_path = Path(cfg.result_dir) / "original_189.png"
    if True:
        print("Rendering original for comparison...")
        renders_orig, _, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors,
            sh_degree=sh_degree,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
        )

        rendered_orig = renders_orig[0, ..., :3]
        rendered_orig = rendered_orig.clamp(0, 1)
        rendered_orig = (rendered_orig.cpu().numpy() * 255).astype(np.uint8)

        imageio.imwrite(orig_path, rendered_orig)
        print(f"Saved original to {orig_path}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
