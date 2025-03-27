from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch
import os
import sys


project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
sys.path.append(project_root + "/..")

from torch.nn.modules import Module
from torch.utils.data import DataLoader, DistributedSampler

try:
    from core.raft_stereo import RAFTStereo
except ImportError:
    import os

    os.chdir("/RAFT-Stereo")
    from core.raft_stereo import RAFTStereo
from fusion_args import FusionArgs
from train_fusion.ddp import DDPTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from hsvfusion.model import HSVNet
from train_fusion.ddp_loss import SelfLoss

from train_fusion.loss_function import (
    DynamicRangeLoss,
    loss_fn_depth_gt_box,
    reproject_disparity,
    self_supervised_loss,
    gt_loss,
    warp_reproject_loss,
    disparity_smoothness,
)
from train_fusion.my_h5_dataloader import MyH5DataSet, MyRefinedH5DataSet
from train_fusion.dataloader import (
    EntityDataSet,
    StereoDataset,
    StereoDatasetArgs,
    MiddleburyDataset,
)
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from tqdm import tqdm


class RaftTrainer(DDPTrainer):
    def __init__(self):

        args = FusionArgs()
        # args.restore_ckpt = "models/raftstereo-eth3d.pth"
        args.restore_ckpt = "checkpoints/latest_HSVFusionResGammaRel.pth"
        args.shared_backbone = False
        args.n_gru_layers = 3
        args.n_downsample = 2
        args.batch_size = 2
        args.valid_steps = 100
        args.lr = 0.000001
        args.real_input_only = False
        # args.corr_implementation = "reg"
        args.log_dir = "runs_hsv"
        args.name = "HSVFusionResGammaRel"
        args.shared_fusion = True
        args.hsv_activation = False
        args.mixed_precision = True
        args.gamma_correction = True
        args.freeze_backbone = ["Extractor", "Updater", "Volume", "BatchNorm"]
        self.args = args
        super().__init__(args)

    def init_models(self) -> Module:
        # raft_model = RAFTStereo(self.args).to(self.device)

        model = HSVNet(self.args, init_raft_stereo=True).to(self.device)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )
        print(model.module.encoder.conv1.state_dict()["weight"][0])
        w_dict = torch.load(self.args.restore_ckpt)
        if "total_steps" in w_dict:
            self.total_steps = w_dict["total_steps"]
            w_dict = w_dict["model_state_dict"]

        # model.module.load_state_dict(w_dict, strict=True)
        # print(model.module.encoder.conv1.state_dict()["weight"][0])
        checkpoint = torch.load("models/raftstereo-middlebury.pth")

        # 새로운 state_dict 생성
        # new_state_dict = OrderedDict()
        for key in list(w_dict.keys()):
            if "raft_stereo." in key:
                del w_dict[key]

        for key, value in checkpoint.items():
            if key.startswith("module."):
                # 'module.' 접두사 제거
                new_key = key[7:]
            else:
                new_key = key
            w_dict[f"raft_stereo.{new_key}"] = value

        # 수정된 state_dict 로드
        model.module.load_state_dict(w_dict, strict=False)
        model.module.raft_stereo.eval()
        model.module.encoder.eval()
        for name, param in model.module.raft_stereo.named_parameters():
            param.require_grad = False
        for name, param in model.module.encoder.named_parameters():
            param.require_grad = False

        return model

    def train_mode(self):
        self.model.train()
        self.model.module.freeze_raft()

        fixed_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        }

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:
        # dataset = MyH5DataSet(
        #     frame_cache=False,
        #     use_right_shift=False,
        #     bpnet_gt=True,
        #     scene_list=[
        #         "10-08-10-39-20",
        #         # "09-08-17-27-33",
        #         # "09-28-17-34-59",
        #         # "09-28-21-15-50",
        #         "10-08-10-26-23",
        #         "10-08-10-34-37",
        #         "10-06-18-27-51",
        #         "10-01-16-20-28",
        #         "10-01-16-03-50",
        #         "09-09-20-04-34",
        #         # "09-09-19-46-45",
        #         # "09-20-13-50-44",
        #     ],
        # )
        # # dataset_refined = MyRefinedH5DataSet(use_right_shift=True)
        dataset_flying = StereoDataset(
            StereoDatasetArgs(
                flying3d_json=True,
                shift_filter=True,
                noised_input=False,
                rgb_rendered=True,
                rgb_gt=True,
                fast_test=True,
                use_rendered_nir=False,
            )
        )

        dataset_drive = StereoDataset(
            StereoDatasetArgs(
                flow3d_driving_json=True,
                shift_filter=True,
                noised_input=False,
                rgb_rendered=True,
                rgb_gt=True,
                fast_test=True,
                use_rendered_nir=False,
            )
        )

        dataset_train = EntityDataSet(
            dataset_flying.input_list
            + dataset_drive.input_list
            # + dataset.input_list[50:]
        )
        # dataset_valid = EntityDataSet(input_list=dataset.input_list[:50])
        # dataset_train = MiddleburyDataset()
        dataset_valid = EntityDataSet(dataset_train.input_list[:30])
        train_sampler = DistributedSampler(dataset_train)
        valid_sampler = DistributedSampler(dataset_valid)
        return (
            train_sampler,
            valid_sampler,
            DataLoader(
                dataset_train,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=1,
            ),
            DataLoader(
                dataset_valid,
                batch_size=2,
                sampler=valid_sampler,
                num_workers=1,
                drop_last=True,
            ),
        )

    def create_image_figure(self, image, cmap=None):
        fig, ax = plt.subplots()
        if isinstance(image, torch.Tensor):
            if image.device != "cpu":
                image = image.cpu()
            if image.ndim > 3:
                image = image[0]
            if image.shape[0] < 100:
                image = image.permute(1, 2, 0).numpy()
        if cmap is not None:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=128)
        else:
            ax.imshow(image.astype(np.uint8))
        return fig

    def log_figures(self, idx: int, batch: List[torch.Tensor]):
        with torch.no_grad():
            fusion, flow = self.model([batch[0], batch[1]], [batch[2], batch[3]])
            flow_rgb = self.model.module.raft_stereo(
                batch[0].cuda(), batch[1].cuda(), test_mode=True
            )[1]
            flow_nir = self.model.module.raft_stereo(
                batch[2].cuda().repeat(1, 3, 1, 1),
                batch[3].cuda().repeat(1, 3, 1, 1),
                test_mode=True,
            )[1]

        val_head = "valid_" if idx < 0 else ""
        if idx < 0:
            idx = -idx

        self.logger.add_figure(
            val_head + "disparity",
            self.create_image_figure(-flow[-1][0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.add_figure(
            val_head + "disparity_rgb",
            self.create_image_figure(-flow_rgb[0, 0].cpu().numpy(), "magma"),
            idx,
        )
        self.logger.add_figure(
            val_head + "disparity_nir",
            self.create_image_figure(-flow_nir[0, 0].cpu().numpy(), "magma"),
            idx,
        )

        self.logger.add_figure(
            val_head + "rgb",
            self.create_image_figure(torch.concat(batch[:2], dim=-2)),
            idx,
        )
        self.logger.add_figure(
            val_head + "nir",
            self.create_image_figure(torch.concat(batch[2:4], dim=-2)),
            idx,
        )
        self.logger.add_figure(
            val_head + "fusion", self.create_image_figure(fusion), idx
        )

        if len(batch) > 5 and not self.args.real_input_only:
            self.logger.add_figure(
                val_head + "rgb_gt",
                self.create_image_figure(torch.concat(batch[4:6], dim=-1)),
                idx,
            )

    def loss_fn_gt(self, flow: List[torch.Tensor], disparity_gt: torch.Tensor):
        return gt_loss(None, disparity_gt, flow)

    def init_loss_function(self) -> Callable[..., Any]:
        self.self_loss = SelfLoss()

        def loss_fn(
            fusion: Tuple[torch.Tensor],
            flow: List[torch.Tensor],
            rgb_gt: Tuple[torch.Tensor],
            flow_gt: torch.Tensor,
            use_color_loss=True,
        ):
            loss, metric = gt_loss(None, flow_gt, flow)
            if use_color_loss:
                rep_loss = self.self_loss.compute_reprojection_loss(
                    fusion[0] / 255, rgb_gt[0] / 255
                )
                +self.self_loss.compute_reprojection_loss(
                    fusion[1] / 255, rgb_gt[1] / 255
                )

                metric["color"] = rep_loss.mean()
                loss = loss + rep_loss.mean() * 150
            for k, v in metric.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=flow[-1].device)
                metric[k] = v

            return loss, metric

        return loss_fn

    def process_batch(self, data_blob):
        inputs = [x.to(self.device) for x in data_blob]
        target_gt = inputs[-2]
        disp_gt = inputs[-1]

        fusion, flow = self.model(
            [inputs[0], inputs[1]], [inputs[2], inputs[3]], raft_stereo=True
        )
        fusion = torch.split(fusion, fusion.shape[0] // 2, dim=0)
        loss, metrics = self.loss_fn(
            fusion,
            flow,
            inputs[4:6],
            disp_gt,
            use_color_loss=not self.args.real_input_only,
        )
        return loss, metrics

    def validate(self, model, valid_loader: DataLoader):
        model.eval()
        metrics: Dict[str, torch.Tensor] = {}
        losses = []
        with torch.no_grad():
            for i_batch, input_valid in tqdm(enumerate(valid_loader)):
                inputs = [x.to(self.device) for x in input_valid]
                target_gt = inputs[-2]
                disp_gt = inputs[-1]
                fusion, flow = self.model(
                    [inputs[0], inputs[1]], [inputs[2], inputs[3]], raft_stereo=True
                )
                loss, metric = self.loss_fn_gt(flow, disp_gt)
                for k, v in metric.items():
                    k = f"valid_{k}"
                    if k not in metrics:
                        metrics[k] = torch.tensor(0.0).to(self.device)
                    metrics[k] += float(v) / len(valid_loader)
                losses.append(loss.item())

                if i_batch % 5 == 0:
                    self.log_figures(-self.total_steps - i_batch, input_valid[:4])

        loss = sum(losses) / len(losses)

        return loss, metrics


if __name__ == "__main__":
    trainer = RaftTrainer()
    trainer.train()
