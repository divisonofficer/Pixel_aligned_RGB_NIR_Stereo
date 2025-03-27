import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-2):
        super(FixedBatchNorm2d, self).__init__()
        self.eps = eps  # 안정성을 위한 작은 값
        self.num_features = num_features  # 채널 수

    def forward(self, x):
        assert (
            x.size(1) == self.num_features
        ), f"입력의 채널 수가 {self.num_features}와 일치하지 않습니다."
        # 배치와 공간 차원(H, W)을 따라 평균과 분산 계산 (채널별)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, C, 1, 1)

        # sqrt를 사용하지 않고 rsqrt로 정규화
        inv_std = torch.rsqrt(var + self.eps)  # (1, C, 1, 1)

        # 정규화 수행 (weight=1, bias=0)
        x_normalized = (x - mean) * inv_std
        return x_normalized


class LocalAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LocalAttentionModule, self).__init__()
        self.local_conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1
        )
        self.local_bn1 = nn.BatchNorm2d(
            in_channels // reduction, track_running_stats=False
        )
        self.local_relu = nn.ReLU(inplace=False)
        self.local_conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1
        )
        self.local_bn2 = nn.BatchNorm2d(in_channels, track_running_stats=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(
                m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        local_branch = self.local_conv1(x)
        local_branch = self.local_bn1(local_branch)
        local_branch = self.local_relu(local_branch)
        local_branch = self.local_conv2(local_branch)
        local_branch = self.local_bn2(local_branch)
        return local_branch


class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GlobalAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Global average pooling branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # First branch
        self.global_conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1
        )
        self.global_bn1 = nn.BatchNorm2d(
            in_channels // reduction, track_running_stats=False, eps=0.001
        )
        self.global_relu = nn.ReLU(inplace=False)
        self.global_conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, kernel_size=1
        )
        self.global_bn2 = nn.BatchNorm2d(
            in_channels, track_running_stats=False, eps=0.001
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.data *= 0.1
            elif isinstance(
                m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Global average pooling branch
        avg_pool = self.global_avg_pool(x).float()

        # First branch

        global_branch = self.global_conv1(avg_pool)
        global_branch = self.global_bn1(global_branch)

        global_branch = self.global_relu(global_branch)

        global_branch = self.global_conv2(global_branch)

        global_branch = self.global_bn2(global_branch)
        return global_branch


class MultiScaleChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleChannelAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.local_attention = LocalAttentionModule(in_channels, reduction)
        self.global_attention = GlobalAttentionModule(in_channels, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.local_attention(x) + self.global_attention(x)

        out = self.sigmoid(out)

        return out


class AttentionFeatureFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(AttentionFeatureFusion, self).__init__()

        self.attention_rgb = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_nir = MultiScaleChannelAttentionModule(in_channels, reduction)

        self.attention_fusion = MultiScaleChannelAttentionModule(in_channels, reduction)

    def forward(self, rgb, nir, debug_attention=False):
        # Apply the attention modules to the input features
        rgb_att = self.attention_rgb(rgb)
        nir_att = self.attention_nir(nir)

        # Concatenate the attention features
        sum_att = rgb_att + nir_att + 1e-6
        rgb_att = rgb * rgb_att / sum_att * 2
        nir_att = nir * nir_att / sum_att * 2
        att_features = rgb_att + nir_att

        # Apply the attention fusion module
        att_fusion = self.attention_fusion(att_features)

        out = att_fusion * rgb_att + (1 - att_fusion) * nir_att
        if debug_attention:
            return att_fusion, rgb, nir
        return out


class BAttentionFeatureFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(BAttentionFeatureFusion, self).__init__()

        self.attention_rgb = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_nir = MultiScaleChannelAttentionModule(in_channels, reduction)

        self.attention_fusion = MultiScaleChannelAttentionModule(in_channels, reduction)

    def forward(self, rgb, nir, debug_attention=False):
        # Apply the attention modules to the input features
        rgb_att = self.attention_rgb(rgb)
        nir_att = self.attention_nir(nir)

        # Concatenate the attention features
        sum_att = rgb_att + nir_att + 1e-6
        rgb_att = rgb * rgb_att / sum_att * 2
        nir_att = nir * nir_att / sum_att * 2
        att_features = rgb_att + nir_att

        # Apply the attention fusion module
        att_fusion = self.attention_fusion(att_features)

        out = att_fusion * rgb + (1 - att_fusion) * nir
        if debug_attention:
            return att_fusion, rgb, nir
        return out


class IAttentionFeatureFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(IAttentionFeatureFusion, self).__init__()
        self.attention_1 = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_2 = MultiScaleChannelAttentionModule(in_channels, reduction)

    def forward(self, rgb, nir, debug_attention=False):
        # Apply the attention modules to the input features
        att_1 = self.attention_1(rgb + nir)
        rgb_att_1 = att_1 * rgb
        nir_att_1 = (1 - att_1) * nir

        att_2 = self.attention_2(rgb_att_1 + nir_att_1)
        out = att_2 * rgb + (1 - att_2) * nir
        if debug_attention:
            return att_2, rgb, nir
        return out


class ConcatFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=16):
        super(ConcatFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, rgb, nir):
        return self.conv1(torch.cat((rgb, nir), dim=1))


class AdditionFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(AdditionFusion, self).__init__()

    def forward(self, rgb, nir):
        return (rgb + nir) / 2


class WeightsumFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(WeightsumFusion, self).__init__()

    def forward(self, rgb, nir):
        return rgb * 0.25 + nir * 0.75


class MultipleFusion(nn.Module):
    def __init__(self, in_channels=128, reduction=4):
        super(MultipleFusion, self).__init__()

    def forward(self, rgb, nir):
        return rgb * nir
