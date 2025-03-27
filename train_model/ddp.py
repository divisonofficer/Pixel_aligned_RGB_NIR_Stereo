import os
import sys
import signal
import logging
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler

from fusion_args import FusionArgs
from train_stereo import Logger
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class DDPTrainer:
    def __init__(self, args: FusionArgs):
        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.total_steps = 0
        self.should_keep_training = True
        self.global_batch_num = 0

        # DDP 초기화
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl")
        torch.autograd.set_detect_anomaly(True)

        # 신호 처리 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)

        self.model = self.init_models()

        self.train_sampler, self.valid_sampler, self.train_loader, self.valid_loader = (
            self.init_dataloader()
        )
        self.train_mode()
        self.init_optimizers()
        self.loss_fn = self.init_loss_function()
        # rank 0에서만 로거 초기화
        if dist.get_rank() == 0:
            self.logger = Logger(
                self.model.module, self.scheduler, log_dir=self.args.log_dir
            )
            self.logger.total_steps = self.total_steps

    def signal_handler(self, sig, frame):
        print(f"Process {dist.get_rank()} received signal {sig}")
        if dist.get_rank() == 0:
            print("Interrupt received. Saving model and performing validation...")
            torch.save(self.model.module.state_dict(), "interrupted_model.pth")
            self.validate(self.model.module, self.valid_loader)
        dist.barrier()
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        dist.destroy_process_group()

    def init_models(self) -> nn.Module:
        """모델을 초기화합니다."""
        raise NotImplementedError("모델 초기화 메소드를 재정의해야 합니다.")

    def init_dataloader(
        self,
    ) -> Tuple[DistributedSampler, DistributedSampler, DataLoader, DataLoader]:
        """데이터 로더를 초기화합니다."""
        raise NotImplementedError("데이터 로더 초기화 메소드를 재정의해야 합니다.")

    def init_loss_function(
        self,
    ) -> Callable[..., Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """손실 함수를 정의합니다."""
        raise NotImplementedError("손실 함수 초기화 메소드를 재정의해야 합니다.")

    def init_optimizers(self):
        """옵티마이저 및 스케줄러를 초기화합니다."""
        self.train_mode()

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.wdecay,
            eps=1e-8,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            self.args.lr,
            self.args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        self.scaler = GradScaler(enabled=self.args.mixed_precision, init_scale=1024)

    def train_mode(self):
        self.model.train()

    def log_figures(self, idx: int, batch: List[torch.Tensor]):
        """텐서보드에 이미지 기록"""
        raise NotImplementedError("이미지 기록 메소드를 재정의해야 합니다 ")

    def train(self):
        """학습 과정을 정의합니다."""
        self.train_mode()

        while self.should_keep_training:
            self.train_sampler.set_epoch(self.total_steps)
            for i_batch, data_blob in enumerate(tqdm(self.train_loader)):
                try:
                    self.optimizer.zero_grad()
                    loss, metrics = self.process_batch(data_blob)

                    # 손실 및 메트릭을 모든 프로세스에서 합산
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / dist.get_world_size()

                    for k in metrics:
                        dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
                        metrics[k] = metrics[k] / dist.get_world_size()

                    if dist.get_rank() == 0:
                        self.log_metrics(loss, metrics)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()

                    if self.total_steps % 10 == 0 and dist.get_rank() == 0:
                        self.log_figures(i_batch, data_blob)
                    self.total_steps += 1
                    if (
                        dist.get_rank() == 0
                        and self.total_steps % self.args.valid_steps == 0
                    ):
                        self.save_model_checkpoint()
                        self.run_validation()

                    if self.total_steps > self.args.num_steps:
                        self.should_keep_training = False
                        break

                except Exception as e:
                    print(f"Exception occurred in process {dist.get_rank()}: {e}")
                    traceback.print_exc()
                    self.should_keep_training = False
                    break

        if dist.get_rank() == 0:
            self.save_final_model()

    def process_batch(self, data_blob) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """배치 데이터 처리 및 손실 계산"""
        raise NotImplementedError(
            "데이터 처리 및 손실 계산 메소드를 재정의해야 합니다."
        )

    def validate(self, model, valid_loader: DataLoader) -> Tuple[torch.Tensor, Dict]:
        """검증 과정을 정의합니다."""
        raise NotImplementedError("검증 메소드를 재정의해야 합니다.")

    def log_metrics(self, loss, metrics):
        """메트릭을 기록합니다."""
        if self.logger is not None:
            self.logger.write_scalar("live_loss", loss.item(), self.global_batch_num)
            self.logger.write_scalar(
                "learning_rate",
                self.optimizer.param_groups[0]["lr"],
                self.global_batch_num,
            )
            self.global_batch_num += 1
            self.logger.push(metrics)

    def save_model_checkpoint(self):
        """모델 체크포인트 저장"""
        save_path = Path(f"checkpoints/{self.total_steps}_{self.args.name}.pth")
        logging.info(f"Saving file {save_path.absolute()}")
        model_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(model_dict, save_path)
        torch.save(
            model_dict,
            Path(f"checkpoints/latest_{self.args.name}.pth"),
        )

    def run_validation(self):
        """검증을 실행합니다."""
        self.model.eval()
        val_loss, val_metrics = self.validate(self.model.module, self.valid_loader)
        self.logger.write_dict(val_metrics)
        self.logger.write_scalar("valid_loss", val_loss, self.total_steps)
        self.train_mode()

    def save_final_model(self):
        """최종 모델 저장"""
        torch.save(
            self.model.module.state_dict(),
            f"checkpoints/final_{self.args.name}.pth",
        )
