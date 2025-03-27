import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import Pyramid, BasicEncoder, ResidualBlock
from .fusion import (
    AdditionFusion,
    AttentionFeatureFusion,
    BAttentionFeatureFusion,
    ConcatFusion,
    IAttentionFeatureFusion,
    MultipleFusion,
    WeightsumFusion,
)
from .update import BasicMultiUpdateBlock
from .corr import (
    CorrBlock1D,
    PytorchAlternateCorrBlock1D,
    CorrBlockFast1D,
    AlternateCorrBlock,
)
from .utils import coords_grid, upflow8

autocast = torch.cuda.amp.autocast


class RAFTStereoFusion(nn.Module):
    def define_fusion_layer(self):
        if self.args.fusion == "AFF":
            return AttentionFeatureFusion
        if self.args.fusion == "ConCat":
            return ConcatFusion
        if self.args.fusion == "iAFF":
            return IAttentionFeatureFusion
        if self.args.fusion == "bAFF":
            return BAttentionFeatureFusion
        if self.args.fusion == "add":
            return AdditionFusion
        if self.args.fusion == "mul":
            return MultipleFusion
        if self.args.fusion == "weight":
            return WeightsumFusion

        return AttentionFeatureFusion

    def checknan(self, item, tag="NAN is here"):
        if isinstance(item, list) or isinstance(item, tuple):
            for i in item:
                if self.checknan(i, tag):
                    break

        if torch.isnan(item).any():
            print(f"{tag} shows nan value")
            print(item)
            return True
        if torch.isinf(item).any():
            print(f"{tag} shows inf value")
            return True
        return False

    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims

        self.update_block = BasicMultiUpdateBlock(
            self.args, hidden_dims=args.hidden_dims
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2)
                for i in range(self.args.n_gru_layers)
            ]
        )

        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", downsample=args.n_downsample
        )

        if not self.args.shared_fusion:
            self.fnet_nir = BasicEncoder(
                output_dim=256, norm_fn="instance", downsample=args.n_downsample
            )

        self.fusion = self.define_fusion_layer()(256)
        self.cnet = Pyramid(
            output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm
        )
        self.conv_pyramid = nn.Sequential(
            ResidualBlock(256, 256, "instance", stride=1),
            nn.Conv2d(256, 128, 3, padding=1),
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()
                for _, param in m.named_parameters():
                    param.requires_grad = False

    def freeze_raft(self):
        if "BatchNorm" in self.args.freeze_backbone:
            self.freeze_bn()
        if "Extractor" in self.args.freeze_backbone and self.args.shared_backbone:
            self.cnet.freeze_raft()

        if self.args.shared_backbone:
            if "Volume" in self.args.freeze_backbone:
                self.conv2.eval()
        else:
            if "Extractor" in self.args.freeze_backbone:
                self.fnet.eval()
                for name, param in self.fnet.named_parameters():
                    param.requires_grad = False
        if "Volume" in self.args.freeze_backbone:
            self.context_zqr_convs.eval()
            for name, param in self.context_zqr_convs.named_parameters():
                param.requires_grad = False
            self.cnet.eval()
            for name, param in self.cnet.named_parameters():
                param.requires_grad = False
        if "Updater" in self.args.freeze_backbone:
            self.update_block.eval()
            for name, param in self.update_block.named_parameters():
                param.requires_grad = False
        self.freeze_bn()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, D, H, W = flow.shape
        factor = 2**self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(
        self,
        img_rgb_l,
        img_rgb_r,
        img_nir_l,
        img_nir_r,
        iters=12,
        flow_init=None,
        test_mode=False,
    ):
        """Estimate optical flow between pair of frames"""

        img_rgb_l = (2 * (img_rgb_l / 255.0) - 1.0).contiguous()
        img_rgb_r = (2 * (img_rgb_r / 255.0) - 1.0).contiguous()
        img_nir_l = (2 * (img_nir_l / 255.0) - 1.0).contiguous()
        img_nir_r = (2 * (img_nir_r / 255.0) - 1.0).contiguous()

        if img_rgb_l.shape[1] == 1:
            img_rgb_l = img_rgb_l.repeat(1, 3, 1, 1)
        if img_rgb_r.shape[1] == 1:
            img_rgb_r = img_rgb_r.repeat(1, 3, 1, 1)
        if img_nir_l.shape[1] == 1:
            img_nir_l = img_nir_l.repeat(1, 3, 1, 1)
        if img_nir_r.shape[1] == 1:
            img_nir_r = img_nir_r.repeat(1, 3, 1, 1)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):

            if self.args.shared_backbone:
                *cnet_list, x, x_rgb, x_nir = self.cnet(
                    torch.cat((img_rgb_l, img_rgb_r), dim=0),
                    torch.cat((img_nir_l, img_nir_r), dim=0),
                    dual_inp=True,
                    num_layers=self.args.n_gru_layers,
                    fmap_out=True,
                )
                fmap_fusion_l, fmap_fusion_r = self.conv2(x).split(
                    dim=0, split_size=x.shape[0] // 2
                )
                fmap_rgb_l, fmap_rgb_r = self.conv2(x_rgb).split(
                    dim=0, split_size=x.shape[0] // 2
                )
                fmap_nir_l, fmap_nir_r = self.conv2(x_nir).split(
                    dim=0, split_size=x.shape[0] // 2
                )
            else:

                fmap_rgb_l, fmap_rgb_r = self.fnet([img_rgb_l, img_rgb_r])
                if self.args.shared_fusion:
                    fmap_nir_l, fmap_nir_r = self.fnet([img_nir_l, img_nir_r])
                else:
                    fmap_nir_l, fmap_nir_r = self.fnet_nir([img_nir_l, img_nir_r])

                fmap_fusion = self.fusion(
                    torch.concat([fmap_rgb_l, fmap_rgb_r], dim=0),
                    torch.concat([fmap_nir_l, fmap_nir_r], dim=0),
                )
                fmap_fusion_l, fmap_fusion_r = fmap_fusion.split(
                    dim=0, split_size=fmap_rgb_l.shape[0]
                )
                fmap_fusion_128 = self.conv_pyramid(fmap_fusion_l)
                cnet_list = self.cnet(
                    fmap_fusion_128, num_layers=self.args.n_gru_layers
                )

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap_rgb_l, fmap_rgb_r = fmap_rgb_l.float(), fmap_rgb_r.float()
            fmap_nir_l, fmap_nir_r = fmap_nir_l.float(), fmap_nir_r.float()
            fmap_fusion_l, fmap_fusion_r = fmap_fusion_l.float(), fmap_fusion_r.float()
        elif self.args.corr_implementation == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap_rgb_l, fmap_rgb_r = fmap_rgb_l.float(), fmap_rgb_r.float()
            fmap_nir_l, fmap_nir_r = fmap_nir_l.float(), fmap_nir_r.float()
            fmap_fusion_l, fmap_fusion_r = fmap_fusion_l.float(), fmap_fusion_r.float()
        elif self.args.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn_list = []
        if self.args.alter_option != "Origin":
            corr_fn_fusion = corr_block(
                fmap_fusion_l,
                fmap_fusion_r,
                radius=self.args.corr_radius,
                num_levels=self.args.corr_levels,
            )
            corr_fn_list.append(corr_fn_fusion)
        if self.args.alter_option != "Fusion":

            corr_fn_nir = corr_block(
                fmap_nir_l,
                fmap_nir_r,
                radius=self.args.corr_radius,
                num_levels=self.args.corr_levels,
            )
            corr_fn_list.append(corr_fn_nir)
            if self.args.alter_option != "Nir":
                corr_fn_rgb = corr_block(
                    fmap_rgb_l,
                    fmap_rgb_r,
                    radius=self.args.corr_radius,
                    num_levels=self.args.corr_levels,
                )
                corr_fn_list.append(corr_fn_rgb)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            for corr_fn in corr_fn_list:
                coords1 = coords1.detach()
                corr = corr_fn(coords1)  # index correlation volume
                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if (
                        self.args.n_gru_layers == 3 and self.args.slow_fast_gru
                    ):  # Update low-res GRU
                        net_list = self.update_block(
                            net_list,
                            inp_list,
                            iter32=True,
                            iter16=False,
                            iter08=False,
                            update=False,
                        )
                    if (
                        self.args.n_gru_layers >= 2 and self.args.slow_fast_gru
                    ):  # Update low-res GRU and mid-res GRU
                        net_list = self.update_block(
                            net_list,
                            inp_list,
                            iter32=self.args.n_gru_layers == 3,
                            iter16=True,
                            iter08=False,
                            update=False,
                        )
                    net_list, up_mask, delta_flow = self.update_block(
                        net_list,
                        inp_list,
                        corr,
                        flow,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=self.args.n_gru_layers >= 2,
                    )

                # in stereo mode, project flow onto epipolar
                delta_flow[:, 1] = 0.0

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow

                # We do not need to upsample or output intermediate results in test_mode
                if test_mode and itr < iters - 1:
                    continue

                # upsample predictions
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up = flow_up[:, :1]

                flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
