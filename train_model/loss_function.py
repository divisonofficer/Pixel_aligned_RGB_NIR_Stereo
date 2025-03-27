from typing import List
import torch
import torch.nn.functional as F
import numpy as np


def ssim(x, y, channel=1):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    loss_ssim = ssim_n / ssim_d

    return loss_ssim


def loss_fn_depth_gt_box(
    flow_preds: list[torch.Tensor],
    target_gt: torch.Tensor,
    box_size=5,
    weight=0.9,
    L2=False,
):
    """
    박스 필터를 사용하여 LiDAR 포인트 주변의 깊이 예측 손실을 계산합니다.

    Parameters:
    - flow_preds: List of flow predictions (each of shape [B,1,H,W]).
    - target_gt: Ground truth tensor of shape [B, N, 3] where last dim is (v, u, depth).
    - box_size: 박스의 크기 (기본값은 5).
    - weight: 각 예측 단계의 가중치 (기본값은 0.9).

    Returns:
    - depth_loss: 전체 박스 손실의 평균.
    - depth_loss_last: 마지막 플로우 예측의 손실.
    """
    B, N = target_gt.shape[:2]
    device = target_gt.device
    dtype = target_gt.dtype

    # 목표 픽셀 위치 (v, u) 추출 및 클램핑
    gt_v = target_gt[:, :, 1].float()  # y 좌표
    gt_u = target_gt[:, :, 0].float()  # x 좌표
    target_depth = target_gt[:, :, 2]  # 목표 깊이 (B, N)

    # 흐름 예측의 크기 가져오기
    B, _, H, W = flow_preds[-1].shape

    # 박스 반지름 계산
    half_box = box_size // 2

    # 박스 오프셋 생성: [box_size^2, 2]
    # torch.meshgrid을 사용하여 박스 내 모든 오프셋을 생성
    offset_range = torch.arange(-half_box, half_box + 1, device=device)
    grid_v, grid_u = torch.meshgrid(
        offset_range, offset_range, indexing="ij"
    )  # grid_v, grid_u: [box_size, box_size]
    delta = torch.stack((grid_u.flatten(), grid_v.flatten()), dim=1)  # [box_size^2, 2]

    # 박스 내 좌표의 수
    num_offsets = box_size**2

    # 초기화
    depth_loss = 0.0
    flows_len = len(flow_preds)
    depth_loss_last = 0

    for idx, flow in enumerate(flow_preds):
        # flow: [B, 1, H, W]
        # target_u, target_v: [B, N]
        # delta: [box_size^2, 2]
        # expanded_gt_u, expanded_gt_v: [B, N, box_size^2]
        expanded_gt_u = gt_u.unsqueeze(-1) + delta[:, 0]  # [B, N, box_size^2]
        expanded_gt_v = gt_v.unsqueeze(-1) + delta[:, 1]  # [B, N, box_size^2]

        expanded_gt_u = expanded_gt_u / (W - 1) * 2 - 1
        expanded_gt_v = expanded_gt_v / (H - 1) * 2 - 1

        grid = torch.stack((expanded_gt_u, expanded_gt_v), dim=-1)
        grid = grid.view(B, N * num_offsets, 1, 2)

        sampled_flow = F.grid_sample(flow, grid, align_corners=True)
        target_pred = (
            sampled_flow.view(B, 1, N, num_offsets, 1).squeeze(1).squeeze(-1)
        )  # (b, n, 25)

        target_pred = target_pred.mean(dim=-1)
        differ = torch.clip(target_pred - target_depth, -100, 100)
        # 손실 계산: 절대 차이
        if L2:
            loss = torch.sqrt((differ**2).mean()).item()
            return loss
        else:
            loss = torch.abs(differ)  # [B, N]

        # 각 예측 단계의 가중치 적용
        loss = loss.mean(dim=1) * (weight ** (flows_len - idx))
        depth_loss += loss
        depth_loss_last = loss  # 마지막 플로우의 손실

    # 전체 평균 박스 손실 계산
    depth_loss_last = depth_loss_last.mean()

    return depth_loss.mean(), depth_loss_last


def warp(disparity_map: torch.Tensor, left_image: torch.Tensor):
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


def warp_reproject_loss(
    flow_preds: List[torch.Tensor],
    img_left: torch.Tensor,
    img_right: torch.Tensor,
    loss_gamma=0.9,
):
    """Loss function defined over sequence of flow predictions"""
    flow_loss = torch.Tensor([0.0]).to(img_left.device)
    preds_cnt = len(flow_preds)
    img_left = img_left / 255.0
    img_right = img_right / 255.0
    # Apply ReLU to ensure disparity is non-negative
    for i, flow_pred in enumerate(flow_preds):
        warp_right = warp(flow_pred, img_left)
        mask = warp(flow_pred, torch.ones_like(img_left).to(img_left.device))
        ssim_loss = ssim(warp_right, img_right)
        l1_loss = torch.abs(warp_right - img_right)
        loss = (ssim_loss * 0.85 + 0.15 * l1_loss.mean(1, True))[mask > 0]
        flow_loss += loss.mean() * (loss_gamma ** (preds_cnt - i - 1))

    return flow_loss, {
        "ssim_loss": ssim_loss[mask > 0].mean(),
        "l1_loss": l1_loss[mask > 0].mean(),
    }


def disparity_smoothness(disp, img):
    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def get_disparity_smoothness(disp, img):
        disp_gradients_x, disp_gradients_y = gradient(disp)
        image_gradients_x, image_gradients_y = gradient(img)

        # Ensure dimensions are consistent
        min_height = min(disp_gradients_x.shape[2], image_gradients_y.shape[2])
        min_width = min(disp_gradients_x.shape[3], image_gradients_y.shape[3])

        disp_gradients_x = disp_gradients_x[:, :, :min_height, :min_width]
        disp_gradients_y = disp_gradients_y[:, :, :min_height, :min_width]
        image_gradients_x = image_gradients_x[:, :, :min_height, :min_width]
        image_gradients_y = image_gradients_y[:, :, :min_height, :min_width]

        # Add epsilon to avoid zero values causing problems
        epsilon = 1e-6

        # Avoid overflow in exp by clamping the gradients
        weights_x = torch.exp(
            -torch.mean(torch.abs(image_gradients_x), 1, keepdim=True).clamp(
                min=-50, max=50
            )
        )
        weights_y = torch.exp(
            -torch.mean(torch.abs(image_gradients_y), 1, keepdim=True).clamp(
                min=-50, max=50
            )
        )

        # Calculate smoothness terms and ensure non-NaN/Inf results
        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        # Ensure no NaN or Inf in smoothness
        smoothness_x = torch.where(
            torch.isnan(smoothness_x), torch.zeros_like(smoothness_x), smoothness_x
        )
        smoothness_y = torch.where(
            torch.isnan(smoothness_y), torch.zeros_like(smoothness_y), smoothness_y
        )
        smoothness_x = torch.where(
            torch.isinf(smoothness_x), torch.zeros_like(smoothness_x), smoothness_x
        )
        smoothness_y = torch.where(
            torch.isinf(smoothness_y), torch.zeros_like(smoothness_y), smoothness_y
        )

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    # Initialize disparity smoothness
    disp_smoothness = torch.Tensor([0.0]).to(disp[-1].device)
    weight = 0.8

    # Loop over disparity scales and calculate smoothness
    dis_len = len(disp)
    for idx, scaled_disp in enumerate(disp):
        mean_disp = scaled_disp.mean(2, True).mean(3, True)
        scaled_disp = scaled_disp / (mean_disp + 1e-7)
        disp_smoothness += (weight ** (dis_len - idx - 1)) * get_disparity_smoothness(
            scaled_disp, img
        ).mean()
        # Reduce weight for each scale

    return disp_smoothness


def self_supervised_loss(input, flow):
    image_viz_left, image_viz_right, image_nir_left, image_nir_right = input

    image_left = torch.concat([image_viz_left, image_nir_left], dim=1)
    image_right = torch.concat([image_viz_right, image_nir_right], dim=1)

    loss, metric = warp_reproject_loss(flow, image_left, image_right)

    disparity_smooth = disparity_smoothness(flow, image_left)

    loss += disparity_smooth
    metric["disp_smooth"] = disparity_smooth
    return loss.mean(), metric


def gt_loss(flow_gt, flow_preds, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    _, _, h, w = flow_preds[-1].shape
    flow_gt = -flow_gt[:, :, :h, :w]
    mask = (
        ~torch.isnan(flow_gt) & ~torch.isinf(flow_gt) & (torch.abs(flow_gt) < max_flow)
    )
    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        if n_predictions > 1:
            i_weight = loss_gamma ** (n_predictions - i - 1)
        else:
            i_weight = 1.0
        i_loss = (
            torch.abs(flow_preds[i] - flow_gt)[mask]
            if mask.any()
            else torch.tensor(0.0, device=flow_gt.device)
        )

        flow_loss += i_weight * i_loss.mean()
    epe = torch.sqrt((flow_preds[-1] - flow_gt) ** 2)
    epe = epe[mask]
    epe = epe.view(-1)

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics
