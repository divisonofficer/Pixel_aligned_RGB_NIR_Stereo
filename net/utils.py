import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy import interpolate


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(
        torch.arange(N).float() - N // 2, torch.arange(N).float() - N // 2
    )
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std**2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1, 1, N, N).to(input)
    output = F.conv2d(input.reshape(B * D, 1, H, W), weights, padding=N // 2)
    return output.view(B, D, H, W)


class GuidedFilter(nn.Module):
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def forward(self, I, p, radius=3, eps=1e-6):
        """
        Perform guided filtering on input images using PyTorch.

        Args:
            I (torch.Tensor): Guide image tensor of shape (batch_size, 1, height, width).
            p (torch.Tensor): Input image tensor to be filtered of shape (batch_size, 3, height, width).
            radius (int, optional): Radius of the window. Default is 15.
            eps (float, optional): Regularization parameter to avoid division by zero. Default is 1e-6.

        Returns:
            torch.Tensor: Filtered image tensor of shape (batch_size, 3, height, width).
        """
        # Ensure the guide image has a single channel
        assert (
            I.dim() == 4 and I.size(1) == 1
        ), "Guide image I must have shape (batch, 1, H, W)"
        # Ensure the input image has three channels
        assert (
            p.dim() == 4 and p.size(1) == 3
        ), "Input image p must have shape (batch, 3, H, W)"

        batch_size, _, height, width = I.size()
        window_size = (2 * radius + 1) ** 2

        # Define a box filter kernel
        # The kernel has shape (channels, 1, kernel_size, kernel_size) and is normalized
        # to compute the mean.
        def box_filter(x):
            # x: (batch, channels, height, width)
            channels = x.size(1)
            kernel = (
                torch.ones(
                    (channels, 1, 2 * radius + 1, 2 * radius + 1),
                    dtype=x.dtype,
                    device=x.device,
                )
                / window_size
            )
            return F.conv2d(
                x, weight=kernel, bias=None, stride=1, padding=radius, groups=channels
            )

        # Step 1: Compute mean of I, p, I*p, and I*I
        mean_I = box_filter(I)  # (batch, 1, H, W)
        mean_p = box_filter(p)  # (batch, 3, H, W)
        mean_Ip = box_filter(I * p)  # (batch, 3, H, W)
        mean_II = box_filter(I * I)  # (batch, 1, H, W)

        # Step 2: Compute covariance of (I, p) and variance of I
        cov_Ip = mean_Ip - mean_I * mean_p  # (batch, 3, H, W)
        var_I = mean_II - mean_I * mean_I  # (batch, 1, H, W)

        # Step 3: Compute a and b
        a = cov_Ip / (var_I + eps)  # (batch, 3, H, W)
        b = mean_p - a * mean_I  # (batch, 3, H, W)

        # Step 4: Compute mean of a and b
        mean_a = box_filter(a)  # (batch, 3, H, W)
        mean_b = box_filter(b)  # (batch, 3, H, W)

        # Step 5: Compute the output image
        q = mean_a * I + mean_b  # Broadcasting I from (batch,1,H,W) to (batch,3,H,W)

        # Optionally, clamp the output to valid image range
        q = torch.clamp(q, 0, 1)

        return q


class RGBHSV(nn.Module):
    def __init__(self):
        super(RGBHSV, self).__init__()

    def forward(self, rgb: torch.Tensor):
        """
        RGB 텐서를 HSV 텐서로 변환합니다.

        입력:
            rgb: Tensor of shape (b, 3, h, w) with values in [0, 1]

        출력:
            hsv: Tensor of shape (b, 3, h, w) with H in [0, 360], S and V in [0, 1]
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        maxc, _ = rgb.max(dim=1)
        minc, _ = rgb.min(dim=1)
        delta = maxc - minc + 1e-10  # 작은 값을 더해 division by zero 방지

        # Hue 계산
        mask = delta > 0
        h = torch.zeros_like(maxc)

        # Red is max
        mask_r = (maxc == r) & mask
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360

        # Green is max
        mask_g = (maxc == g) & mask
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360

        # Blue is max
        mask_b = (maxc == b) & mask
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

        # Saturation 계산
        s = torch.zeros_like(maxc)
        s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]

        # Value 계산
        v = maxc

        hsv = torch.stack([h, s, v], dim=1)
        return hsv


class HSVRGB(nn.Module):
    def __init__(self):
        super(HSVRGB, self).__init__()

    def forward(self, hsv):
        """
        HSV 텐서를 RGB 텐서로 변환합니다.

        입력:
            hsv: Tensor of shape (b, 3, h, w)
                - H in [0, 360]
                - S in [0, 1]
                - V in [0, 1]

        출력:
            rgb: Tensor of shape (b, 3, h, w) with values in [0, 1]
        """
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        c = v * s  # 채도와 명도를 이용해 채도
        h_prime = h / 60.0  # Hue를 60으로 나눠 섹션 결정
        x = c * (1 - torch.abs((h_prime % 2) - 1))

        zero = torch.zeros_like(h)

        # 각 섹션에 따른 RGB 중간값 계산
        cond = (h_prime >= 0) & (h_prime < 1)
        r = torch.where(cond, c, torch.zeros_like(c))
        g = torch.where(cond, x, torch.zeros_like(x))
        b = torch.zeros_like(x)

        cond = (h_prime >= 1) & (h_prime < 2)
        r = torch.where(cond, x, r)
        g = torch.where(cond, c, g)

        cond = (h_prime >= 2) & (h_prime < 3)
        g = torch.where(cond, c, g)
        b = torch.where(cond, x, b)

        cond = (h_prime >= 3) & (h_prime < 4)
        g = torch.where(cond, x, g)
        b = torch.where(cond, c, b)

        cond = (h_prime >= 4) & (h_prime < 5)
        r = torch.where(cond, x, r)
        b = torch.where(cond, c, b)

        cond = (h_prime >= 5) & (h_prime < 6)
        r = torch.where(cond, c, r)
        b = torch.where(cond, x, b)

        m = v - c
        r = r + m
        g = g + m
        b = b + m

        rgb = torch.stack([r, g, b], dim=1)
        return rgb
