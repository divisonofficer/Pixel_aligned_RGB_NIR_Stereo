from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch

import torch.nn.functional as F


def transfrom_points(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: Transformed points
    """
    points = points.reshape(-1, 3)
    points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = transform_mtx @ points.T
    return points[:3].T


def transform_point_inverse(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns
        np.ndarray: Transformed points
    """
    transform_mtx = np.linalg.pinv(transform_mtx)
    return transfrom_points(points, transform_mtx)


def lidar_points_to_disparity_with_cal(
    points: np.ndarray,
    transform_mtx: np.ndarray,
    calibration_dict: dict,
    points_scale=1000,
):
    points = points.reshape(-1, 3) * points_scale
    fx = calibration_dict["mtx_left"][0, 0]
    cx = calibration_dict["mtx_left"][0, 2]
    cx_r = calibration_dict["mtx_right"][0, 2]
    cy = calibration_dict["mtx_left"][1, 2]
    baseline = np.linalg.norm(calibration_dict["T"])

    points = transform_point_inverse(points, transform_mtx)
    points = project_points_on_camera(points, fx, cx, cy, 720, 540)
    points[:, 2] = fx * baseline / points[:, 2] + cx - cx_r
    return points


def depth_points_to_disparity_with_cal(
    points: np.ndarray,
    calibration_dict: dict,
):
    fx = calibration_dict["mtx_left"][0, 0]
    baseline = np.linalg.norm(calibration_dict["T"])
    points[:, 2] = (
        fx * baseline / points[:, 2]
        + calibration_dict["mtx_left"][0, 2]
        - calibration_dict["mtx_right"][0, 2]
    )
    return points


def disparity_points_to_depth_with_cal(
    points: Union[np.ndarray, torch.Tensor],
    calibration_dict: dict,
    width=720,
    height=540,
):
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    fx = calibration_dict["mtx_left"][0, 0]
    baseline = np.linalg.norm(calibration_dict["T"])

    points[:, 2] = (
        fx
        * baseline
        / (
            points[:, 2]
            - (calibration_dict["mtx_left"][0, 2] - calibration_dict["mtx_right"][0, 2])
        )
    )
    return points


def refine_disparity_points(points: torch.Tensor, thresh_dist=0.5, thresh_disp=0.85):
    # u, v, d 좌표 분리
    u = points[:, 0]
    v = points[:, 1]
    d = points[:, 2]

    # 거리 계산 함수
    def calculate_distances(u, v):
        # (N, 1) - (1, N) 으로 브로드캐스팅하여 모든 쌍의 유클리드 거리 계산
        dist_u = u.unsqueeze(1) - u.unsqueeze(0)
        dist_v = v.unsqueeze(1) - v.unsqueeze(0)
        distances = torch.sqrt(dist_u**2 + dist_v**2)
        return distances

    # 거리 행렬 계산
    distances = calculate_distances(u, v)

    # 각 포인트의 거리 d 내에서 다른 포인트 찾기
    mask = distances <= d.unsqueeze(1) * thresh_dist

    # 2배 이상 큰 d 값을 가진 포인트 필터링
    d_ratio = d.unsqueeze(1) / d.unsqueeze(0)
    remove_mask = (d_ratio < thresh_disp) & mask

    # remove_mask를 통해 제거할 포인트를 남기지 않는 새로운 인덱스 계산
    keep_indices = ~(remove_mask.any(dim=1))

    # 최종 남은 포인트들
    filtered_points = points[keep_indices]

    return filtered_points


def project_points_on_camera(
    points: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    image_width: float = 0,
    image_height: float = 0,
):
    """
    Project 3D points to 2D image plane
    Args:
        points (np.ndarray): 3D points to project
        focal_length (float): Focal length of the camera
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        image_width (float): Image width, Optional
        image_height (float): Image height, Optional
    Returns:
        np.ndarray: Projected points
    """
    points[:, 0] = points[:, 0] * focal_length / points[:, 2] + cx
    points[:, 1] = points[:, 1] * focal_length / points[:, 2] + cy

    if image_width > 0 and image_height > 0:
        points = points[
            (points[:, 0] >= 0)
            & (points[:, 0] <= image_width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= image_height - 1)
            & (points[:, 2] > 0)
        ]
    return points


def depth_points_to_depth_map(points: np.ndarray, width=720, height=540):
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    depth_map = np.zeros((height, width), dtype=np.float32)
    u, v, d = points.T
    u = u.astype(int)
    v = v.astype(int)
    depth_map[v, u] = d
    return depth_map


def torch_depth_points_to_depth_map(points: torch.Tensor, width=720, height=540):
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    u, v, d = points.T
    u = u.int()
    v = v.int()
    depth_map[v, u] = d
    return depth_map


def render_depth_map(
    points: np.ndarray, width: int = 0, height: int = 0, max_depth=10000
):
    if width == 0:
        width = int(points[:, 0].max()) + 1
    if height == 0:
        height = int(points[:, 1].max()) + 1
    canvas = np.zeros((height, width), dtype=np.uint8)

    for u, v, depth in points:
        depth = depth / max_depth * 255
        depth = np.clip(depth, 0, 255).asdtype(np.uint8)
        canvas[int(v), int(u)] = depth
    return canvas


def points_sampled_disparity(points: np.ndarray, disparity_map: np.ndarray):
    points = points[
        points[:, 1]
        < disparity_map.shape[0] & points[:, 0]
        < disparity_map.shape[1] & points[:, 1]
        >= 0 & points[:, 0]
        >= 0
    ]
    u, v, d = points.T
    d = disparity_map[v.astype(int), u.astype(int)]
    points[:, 2] = d
    return points


def lidar_points_to_disparity(
    points: np.ndarray,
    transform_mtx: np.ndarray,
    focal_length: float,
    baseline: float,
    cx: float,
    cy: float,
):
    points = transform_point_inverse(points, transform_mtx)
    points = project_points_on_camera(points, focal_length, cx, cy, 720, 540)
    points[:, 2] = focal_length * baseline / points[:, 2] - 1
    return points


def pad_lidar_points(lidar_projected_points, target_size=5000):
    current_size = len(lidar_projected_points)

    if current_size >= target_size:
        return lidar_projected_points[:target_size]

    # 필요한 포인트 수 계산
    needed = target_size - current_size

    # 기존 포인트에서 랜덤하게 샘플링 (복원 추출)
    # 샘플링할 포인트 수가 현재 포인트 수보다 많을 경우, 여러 번 반복할 수 있음
    # NumPy의 random.choice를 사용하여 인덱스를 랜덤하게 선택
    sampled_indices = np.random.choice(current_size, size=needed, replace=True)
    sampled_points = lidar_projected_points[sampled_indices]

    # 기존 포인트와 샘플링된 포인트를 결합
    padded_lidar_projected_points = np.concatenate(
        [lidar_projected_points, sampled_points], axis=0
    )

    return padded_lidar_projected_points


def combine_block(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    combined_disparity: np.ndarray,
    criteria: Callable[
        [np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]], bool
    ],
    blk_w=24,
    blk_h=24,
):
    width = disparity_rgb.shape[-1]
    height = disparity_rgb.shape[-2]
    n_blk_u = (width + blk_w - 1) // blk_w  # Ceiling division
    n_blk_v = (height + blk_h - 1) // blk_h  # Ceiling division
    u, v, z = lidar_points.T
    for blk_v_idx in range(n_blk_v):
        for blk_u_idx in range(n_blk_u):
            # Define the vertical block boundaries
            st_v = blk_v_idx * blk_h
            en_v = min((blk_v_idx + 1) * blk_h, height)
            st_u = blk_u_idx * blk_w
            en_u = min((blk_u_idx + 1) * blk_w, width)

            # Identify LiDAR points within the current vertical block
            in_block = (u >= st_u) & (u < en_u) & (v >= st_v) & (v < en_v)

            if not np.any(in_block):
                # No points in this vertical block; retain the horizontal-based disparity
                continue

            # Get the indices of points in the current block
            bu, bv, bz = lidar_points[in_block].T

            # Ensure u and v are within image bounds
            valid = (bu >= 0) & (bu < width) & (bv >= 0) & (bv < height)
            bu, bv, bz = np.stack([bu, bv, bz], axis=1)[valid].T

            critic = criteria(
                bu.astype(np.int32), bv.astype(np.int32), bz, (st_u, en_u, st_v, en_v)
            )

            # Choose the disparity map with lower loss for this block
            if critic:
                chosen_disparity = disparity_rgb[st_v:en_v, st_u:en_u]
            else:
                chosen_disparity = disparity_nir[st_v:en_v, st_u:en_u]

            # Assign the chosen disparity to the combined map
            combined_disparity[st_v:en_v, st_u:en_u] = chosen_disparity

    return combined_disparity


def ssim_torch(x: torch.Tensor, y: torch.Tensor):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return SSIM


def reproject_disparity(
    disparity_map: torch.Tensor, left_image: torch.Tensor, max_disparity=128
):
    batch_size, channels, height, width = left_image.shape
    # Create a mesh grid for pixel coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, device=left_image.device),
        torch.arange(height, device=left_image.device),
        indexing="xy",
    )

    x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, -1).float()
    y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1).float()

    # Compute the new x coordinates based on disparity
    disparity_map = F.pad(
        disparity_map, (1, 1, 1, 1), mode="constant", value=0
    )  # Pad to handle boundary
    disparity_map = F.interpolate(
        disparity_map, size=(height, width), mode="bilinear", align_corners=False
    )  # Resample disparity map

    # Convert disparity map to float type
    disparity_map = disparity_map.squeeze(1)

    x_new_coords = x_coords - disparity_map
    y_new_coords = y_coords

    # Create grid tensor with shape [N, H, W, 2]
    grid = torch.stack([x_new_coords, y_new_coords], dim=-1)

    # Normalize the grid to the range [-1, 1]
    grid = (
        2.0 * grid / torch.tensor([width - 1, height - 1], device=left_image.device)
        - 1.0
    )

    # Perform bilinear interpolation for the reprojected image
    reprojected_image = F.grid_sample(
        left_image, grid, mode="bilinear", align_corners=False
    )

    return reprojected_image
