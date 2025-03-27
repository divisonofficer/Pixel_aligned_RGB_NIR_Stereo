import os
import random
from typing import Tuple, List, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F


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


def cv2toTensor(image: np.ndarray, batch_dim: bool = True):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image).float()
    if batch_dim:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def pixel_graident(img: np.ndarray):
    """
    get normalized pixel gradient of an image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    return gradient_magnitude


def read_image_pair(
    frame_path: str,
):
    ret = []
    for path in ["rgb/left.png", "rgb/right.png", "nir/left.png", "nir/right.png"]:

        if not os.path.exists(f"{frame_path}/{path}"):
            path = path.replace(".png", "_distorted.png")
        img = cv2.imread(
            f"{frame_path}/{path}",
            cv2.IMREAD_GRAYSCALE if "nir" in path else cv2.IMREAD_COLOR,
        )
        if not "nir" in path:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret.append(img)
    return tuple(ret)


def modify_v_channel_numpy_opencv(image_rgb, guide, alpha=0.5, beta=0.5):
    """
    RGB 이미지의 V 채널을 가이드 이미지로 수정하고 다시 RGB로 변환합니다.

    Args:
        image_rgb (numpy.ndarray): 입력 RGB 이미지, shape (h, w, 3), dtype=uint8 또는 float32/float64.
                                   값의 범위는 [0, 255] (uint8) 또는 [0, 1] (float)으로 가정합니다.
        guide (numpy.ndarray): 가이드 이미지, shape (h, w, 1), dtype=uint8 또는 float32/float64.
                               값의 범위는 [0, 255] (uint8) 또는 [0, 1] (float)으로 가정합니다.
        alpha (float, optional): V 채널에 대한 원본 이미지의 가중치. 기본값은 0.5.
        beta (float, optional): V 채널에 대한 가이드 이미지의 가중치. 기본값은 0.5.

    Returns:
        numpy.ndarray: 수정된 RGB 이미지, shape (h, w, 3), dtype=uint8 또는 float32.
    """
    # 입력 이미지가 float일 경우 [0,1] 범위로 정규화되어 있다고 가정
    is_float = False
    if image_rgb.dtype in [np.float32, np.float64]:
        is_float = True
        image = (image_rgb * 255).astype(np.uint8)
    else:
        image = image_rgb.copy()

    # 가이드 이미지도 동일하게 처리
    if guide.dtype in [np.float32, np.float64]:
        guide_normalized = guide * 255
        guide_uint8 = guide_normalized.astype(np.uint8)
    else:
        guide_uint8 = guide.copy()

    # RGB에서 HSV로 변환 (OpenCV는 기본적으로 BGR을 사용하므로 RGB를 BGR로 변환)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # V 채널 수정
    # V 채널은 OpenCV에서 0-255 범위를 가집니다
    V = image_hsv[:, :, 2]

    # 가이드 이미지도 동일한 스케일로 변환
    if is_float:
        I = guide_uint8.astype(np.float32)
    else:
        I = guide_uint8.astype(np.float32)

    # V = alpha * V + beta * I
    V_modified = alpha * V + beta * I.squeeze()  # 가이드 이미지의 채널 차원 제거

    # V 채널 값 클램핑
    V_modified = np.clip(V_modified, 0, 255)

    # 수정된 V 채널을 HSV 이미지에 반영
    image_hsv[:, :, 2] = V_modified

    # HSV에서 BGR로 다시 변환
    image_hsv = image_hsv.astype(np.uint8)
    image_bgr_modified = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # BGR을 다시 RGB로 변환
    image_rgb_modified = cv2.cvtColor(image_bgr_modified, cv2.COLOR_BGR2RGB)

    if is_float:
        # 원래 float 형식으로 반환
        image_rgb_modified = image_rgb_modified.astype(np.float32) / 255.0

    return image_rgb_modified


def guided_filter(I: np.ndarray, p: np.ndarray, radius=15, eps=1e-6):
    # I: guide image (grayscale)
    # p: input image to be filtered (color)
    # radius: window radius
    # eps: regularization parameter

    # Step 1: Mean of I, p, I*p, and I^2

    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))

    # Initialize output
    q = np.zeros_like(p)

    # Process each channel separately
    for c in range(p.shape[2]):
        mean_p = cv2.boxFilter(p[:, :, c], cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(I * p[:, :, c], cv2.CV_64F, (radius, radius))

        # Step 2: Covariance of (I, p) and variance of I
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # Step 3: Calculate a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Step 4: Mean of a and b
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        # Step 5: Output q for channel c
        q[:, :, c] = np.clip(mean_a * I + mean_b, 0, 255)

    return q


def gamma_correction_torch(img, gamma=0.5):
    look_up_table = torch.tensor(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=torch.float32,
        device=img.device,
    )
    img_corrected = torch.take(look_up_table, img.long())
    return img_corrected


def apply_local_gamma_correction_torch(img, gamma=0.5, kernel_size=101, sigma=15):
    """
    Applies gamma correction locally using a Gaussian mask.

    Parameters:
    - img: Input image (torch tensor).
    - gamma: Gamma value for correction.
    - kernel_size: Size of the Gaussian kernel (must be odd).
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - Image with local gamma correction applied.
    """
    # Ensure kernel size is odd
    h, w = img.shape[-2:]
    if kernel_size >= min(h, w):
        kernel_size = min(h, w) - 2

    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a Gaussian mask with values between 0 and 1
    x = (
        torch.arange(kernel_size, dtype=torch.float32, device=img.device)
        - kernel_size // 2
    )
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d / torch.max(gaussian_2d)

    # Resize the mask to match the image size if necessary
    gaussian_mask = F.interpolate(
        gaussian_2d.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Expand mask to have same number of channels as image
    if img.ndim == 4:  # Batch size included
        gaussian_mask = gaussian_mask.unsqueeze(0).expand(
            img.shape[0], img.shape[1], -1, -1
        )
    elif img.ndim == 3:
        gaussian_mask = gaussian_mask.expand_as(img)

    # Apply gamma correction to the entire image
    gamma_corrected = gamma_correction_torch(img, gamma)

    # Blending: result = mask * gamma_corrected + (1 - mask) * original
    blended = gaussian_mask * gamma_corrected + (1.0 - gaussian_mask) * img.float()

    # Clip the values to [0, 255] and convert back to uint8
    blended = torch.clamp(blended, 0, 255).byte()

    return blended


def apply_patch_gamma_correction_torch(img, p=256):
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.shape[1] == 1:
        img = img.expand(-1, 3, -1, -1)
        nir = True
    else:
        nir = False

    # Pad the image
    img = F.pad(img, (p // 2, p // 2, p // 2, p // 2))

    b, c, h, w = img.shape
    for _ in range(20):
        x = random.randint(0, max(w - p, 0))
        y = random.randint(0, max(h - p, 0))
        img[:, :, y : y + p, x : x + p] = apply_local_gamma_correction_torch(
            img[:, :, y : y + p, x : x + p],
            gamma=10 ** (-4 + (random.random() * 8)),
            kernel_size=101,
            sigma=20,
        )

    img = img[:, :, p // 2 : -p // 2, p // 2 : -p // 2]
    if nir:
        return img[:, :1]
    return img


def input_reduce_disparity(inputs: list[torch.Tensor]):
    shift = int(inputs[-1].min() // 16) * 16
    if shift <= 0:
        return inputs
    warp_right = reproject_disparity(
        -inputs[-1].unsqueeze(0) + shift, inputs[0].unsqueeze(0)
    )[0]
    warp_right_nir = reproject_disparity(
        -inputs[-1].unsqueeze(0) + shift, inputs[2].unsqueeze(0)
    )[0]
    rolled_rgb_right = torch.roll(inputs[1], shifts=shift, dims=-1)
    rolled_nir_right = torch.roll(inputs[3], shifts=shift, dims=-1)
    rolled_rgb_right[..., :shift] = warp_right[..., :shift]
    rolled_nir_right[..., :shift] = warp_right_nir[..., :shift]

    inputs[-1] -= shift
    inputs[-2][..., :2] -= shift
    return [
        inputs[0],
        rolled_rgb_right,
        inputs[2],
        rolled_nir_right,
        inputs[-2],
        inputs[-1],
    ]


def crop_and_resize_height(image: torch.Tensor, h_to=360):
    b, c, h, w = image.shape
    cropped = image[:, :, :h_to, :]
    resized = F.interpolate(
        cropped,
        size=(h, w),
        mode="bilinear",
    )
    return resized


def warp_left_to_right(
    left: torch.Tensor, disparity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    왼쪽 이미지를 disparity 맵을 기반으로 오른쪽 이미지로 워핑하고,
    워핑된 영역에 대한 마스크를 생성합니다.

    Args:
        left (torch.Tensor): 왼쪽 이미지 텐서, 형상 (B, C, H, W)
        disparity (torch.Tensor): disparity 맵 텐서, 형상 (B, 1, H, W)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - 워핑된 오른쪽 이미지, 형상 (B, C, H, W)
            - 워핑 마스크, 형상 (B, 1, H, W)
    """
    B, C, H, W = left.size()

    # 그리드 생성
    device = left.device
    dtype = left.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
    )
    grid = torch.stack((grid_x, grid_y), 2)  # Shape: (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, H, W, 2)

    # Disparity 정규화
    disparity_norm = disparity.squeeze(1) * 2.0 / (W - 1)  # Shape: (B, H, W)

    # x 좌표 조정 (오른쪽으로 워핑)
    grid[:, :, :, 0] -= disparity_norm  # Shape: (B, H, W, 2)

    # 오른쪽 이미지 워핑
    warped_right = F.grid_sample(
        left, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # 마스크 생성
    ones = torch.ones_like(disparity)
    warped_mask = F.grid_sample(
        ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # 임계값을 기준으로 이진화 (예: 0.999 이상을 유효 영역으로)
    mask = (warped_mask > 0.999).float()

    return warped_right, mask


def image_disparity_shift(
    image: torch.Tensor,
    disparity: torch.Tensor,
    shift: int = 16,
    is_disparity: bool = False,
    is_reproject: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Disparity 기반으로 이미지를 오른쪽 또는 왼쪽으로 이동시키고,
    이동된 영역에 대한 마스크를 생성합니다.

    Args:
        image (torch.Tensor): 입력 이미지 텐서, 형상 (B, C, H, W)
        disparity (torch.Tensor): 불일치 맵 텐서, 형상 (B, 1, H, W)
        shift (int): 최대 이동 픽셀 수 (양수는 오른쪽, 음수은 왼쪽)
        is_disparity (bool): disparity 맵을 이동하는지 여부

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - 이동된 이미지, 형상 (B, C, H, W)
            - 이동 마스크, 형상 (B, 1, H, W)
    """
    N, C, H, W = image.shape

    # Disparity 정규화 및 shift 맵 생성
    disparity_max = disparity.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    disparity_normalized = disparity / (disparity_max + 1e-8)  # Avoid division by zero
    shift_map = disparity_normalized * shift  # Shape: (B, 1, H, W)

    if is_disparity:
        image = image - shift_map * (
            2 if is_reproject else 1
        )  # Example operation for disparity maps

    # shift_map을 x 축으로 변환 (픽셀 단위 -> [-1, 1] 범위)
    shift_normalized = (shift_map.squeeze(1) / (W - 1)) * 2  # Shape: (B, H, W)

    # 기본 그리드 생성
    theta = torch.zeros(N, 2, 3, device=image.device, dtype=image.dtype)
    theta[:, 0, 0] = 1  # x 스케일
    theta[:, 1, 1] = 1  # y 스케일
    base_grid = F.affine_grid(
        theta, image.size(), align_corners=True
    )  # Shape: (B, H, W, 2)

    # x 좌표에 shift 추가
    grid = base_grid.clone()
    grid_x = grid[..., 0] + shift_normalized  # Shape: (B, H, W)
    grid_y = grid[..., 1]  # y는 변경하지 않음
    grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (B, H, W, 2)

    # grid_sample을 사용하여 이미지 샘플링
    image_shift = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # 마스크 생성
    mask = F.grid_sample(
        torch.ones_like(image[:, :1, :, :]),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return image_shift, mask


def inputs_disparity_shift(
    inputs: List[torch.Tensor], disparities: List[torch.Tensor], shift: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    주어진 입력 이미지와 disparity 맵을 기반으로 이미지를 이동 및 보정하고,
    보정된 이미지와 disparity 맵을 반환합니다.

    Args:
        inputs (List[torch.Tensor]): 입력 이미지 리스트 [left_rgb, right_rgb, left_nir, right_nir]
                                    각 텐서의 형상은 (B, C, H, W)
        disparities (List[torch.Tensor]): disparity 맵 리스트 [disparity_map_left, disparity_map_right]
                                          각 텐서의 형상은 (B, 1, H, W)
        shift (int): 최대 이동 픽셀 수 (양수는 오른쪽, 음수은 왼쪽)

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]:
            - 보정된 이미지 리스트 [left_rgb_corrected, right_rgb_corrected, left_nir_corrected, right_nir_corrected]
            - 보정된 disparity 맵 리스트 [disparity_map_left_corrected, disparity_map_right_corrected]
    """
    assert (
        len(inputs) == 4
    ), "inputs 리스트는 [left_rgb, right_rgb, left_nir, right_nir]의 4개 텐서를 가져야 합니다."
    assert (
        len(disparities) == 2
    ), "disparities 리스트는 [disparity_map_left, disparity_map_right]의 2개 텐서를 가져야 합니다."

    left_rgb, right_rgb, left_nir, right_nir = inputs
    disparity_map_left, disparity_map_right = disparities

    # 왼쪽 이미지와 disparity 맵을 이동
    shift_left, warp_mask_left = image_disparity_shift(
        left_rgb, disparity_map_left, shift=shift
    )
    shift_left_nir, _ = image_disparity_shift(left_nir, disparity_map_left, shift=shift)

    # 왼쪽 disparity 맵 이동 (disparity 맵의 경우 is_disparity=True)
    disparity_shift_left, _ = image_disparity_shift(
        disparity_map_left.clone(),
        disparity_map_left,
        shift=shift,
        is_disparity=True,
        is_reproject=True,
    )

    # 왼쪽 이미지를 오른쪽으로 워핑
    shift_warp_right, _ = warp_left_to_right(shift_left, -disparity_shift_left)

    # 오른쪽 이미지와 disparity 맵을 반대 방향으로 이동
    shift_right, warp_mask_right = image_disparity_shift(
        right_rgb, disparity_map_right, shift=-shift
    )
    shift_right_nir, _ = image_disparity_shift(
        right_nir, disparity_map_right, shift=-shift
    )

    # 오른쪽 disparity 맵 이동
    disparity_shift_right, _ = image_disparity_shift(
        disparity_map_right.clone(),
        disparity_map_right,
        shift=-shift,
        is_disparity=True,
        is_reproject=True,
    )

    # 오른쪽 이미지를 왼쪽으로 워핑
    shift_warp_left, _ = warp_left_to_right(shift_right, disparity_shift_right)

    # NIR 이미지 워핑
    shift_warp_left_nir, _ = warp_left_to_right(shift_right_nir, disparity_shift_right)
    shift_warp_right_nir, _ = warp_left_to_right(shift_left_nir, -disparity_shift_left)

    # 이미지 보정: 마스크가 유효하지 않은 영역(<1)에서는 워핑된 이미지를 사용
    # RGB 이미지
    left_rgb_corrected = shift_left.clone()
    right_rgb_corrected = shift_right.clone()

    # 확장된 마스크를 사용하여 모든 채널에 적용
    warp_mask_left_expanded = warp_mask_left.repeat(1, left_rgb_corrected.size(1), 1, 1)
    warp_mask_right_expanded = warp_mask_right.repeat(
        1, right_rgb_corrected.size(1), 1, 1
    )

    left_rgb_corrected[warp_mask_left_expanded < 1] = shift_warp_left[
        warp_mask_left_expanded < 1
    ]
    right_rgb_corrected[warp_mask_right_expanded < 1] = shift_warp_right[
        warp_mask_right_expanded < 1
    ]

    # NIR 이미지
    left_nir_corrected = shift_left_nir.clone()
    right_nir_corrected = shift_right_nir.clone()

    # 마스크는 채널이 1이므로 repeat 필요 없음
    left_nir_corrected[warp_mask_left < 1] = shift_warp_left_nir[warp_mask_left < 1]
    right_nir_corrected[warp_mask_right < 1] = shift_warp_right_nir[warp_mask_right < 1]

    # disparity 맵 보정
    disparity_map_left_corrected = disparity_shift_left.clone()
    disparity_map_right_corrected = disparity_shift_right.clone()

    disparity_map_left_corrected[warp_mask_left < 1] = disparity_shift_right[
        warp_mask_left < 1
    ]
    disparity_map_right_corrected[warp_mask_right < 1] = disparity_shift_left[
        warp_mask_right < 1
    ]

    # 최종 보정된 이미지와 disparity 맵 리스트 생성
    corrected_inputs = [
        left_rgb_corrected,
        right_rgb_corrected,
        left_nir_corrected,
        right_nir_corrected,
    ]
    corrected_disparities = [
        disparity_map_left_corrected,
        disparity_map_right_corrected,
    ]

    return corrected_inputs, corrected_disparities


def pseudo_nir_np(rgb: np.ndarray):
    if rgb.max() > 10:
        rgb = rgb / 255.0
    interm = np.maximum(rgb, 1 - rgb)

    # Compute the weighted sum and apply the power operation
    nir = (
        interm[..., 0] * 0.229 + interm[..., 1] * 0.587 + interm[..., 2] * 0.114
    ) ** (1 / 0.25)
    return (nir * 255).astype(np.uint8)


def img_pad_np(img: np.ndarray, shape=(540, 720), divd=32, pad_constant=False):
    H = shape[0] + (divd - (shape[0] % divd)) % divd
    W = shape[1] + (divd - (shape[1] % divd)) % divd

    if img.shape[0] > H:
        img = img[(img.shape[0] - H) // 2 : (img.shape[0] + H) // 2]
    if img.shape[1] > W:
        img = img[:, (img.shape[1] - W) // 2 : (img.shape[1] + W) // 2]

    if img.shape[0] < H or img.shape[1] < W:
        img_cp = (
            np.zeros((H, W), dtype=img.dtype)
            if len(img.shape) == 2
            else np.zeros((H, W, 3), dtype=img.dtype)
        )
        if pad_constant:
            img_cp[:] = 1000000

        img_cp[: img.shape[0], : img.shape[1]] = img
        return img_cp
    return img


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
