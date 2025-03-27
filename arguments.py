from typing import List, Union, Optional, Literal


class Arguments:
    def __init__(self):
        self._hidden_dims = [128, 128, 128]
        self._corr_levels = 4
        self._corr_radius = 4
        self._n_downsample = 3
        self._context_norm = "batch"
        self._n_gru_layers = 2
        self._shared_backbone = True
        self._mixed_precision = True
        self._corr_implementation: Literal["reg_cuda", "reg", "alt", "alt_cuda"] = (
            "reg_cuda"
        )

        self._input_channel = 3
        self._slow_fast_gru = False
        self._restore_ckpt = "models/raftstereo-realtime.pth"
        self._lr = 0.001
        self._train_iters = 7
        self._valid_iters = 12
        self._wdecay = 0.0001
        self._num_steps = 500000
        self._valid_steps = 1000
        self._name = "StereoFusion"
        self._batch_size = 4
        self._fusion: Literal[
            "AFF", "ConCat", "iAFF", "bAFF", "add", "mul", "weight"
        ] = "bAFF"
        self._shared_fusion = False
        self._freeze_backbone: List[
            Literal["Extractor", "Updater", "Volume", "BatchNorm"]
        ] = ["Extractor"]
        self._both_side_train = False

        self.log_dir = "./train_log"
        self.log_level = "logging.INFO"
        self.n_total_epoch = 10
        self.accumulation_steps = 5
        self.alter_option: Literal["Fusion", "Origin", "Mixed", "Nir"] = "Mixed"
        self.self_supervised_train = False
        self.grad_scale = 512

    @property
    def hidden_dims(self):
        """GRU의 hidden state와 context의 차원을 설정합니다."""
        return self._hidden_dims

    @hidden_dims.setter
    def hidden_dims(self, value):
        self._hidden_dims = value

    @property
    def corr_levels(self):
        """correlation pyramid에서 레벨의 수를 설정합니다."""
        return self._corr_levels

    @corr_levels.setter
    def corr_levels(self, value):
        self._corr_levels = value

    @property
    def corr_radius(self):
        """correlation pyramid의 폭을 설정합니다."""
        return self._corr_radius

    @corr_radius.setter
    def corr_radius(self, value):
        self._corr_radius = value

    @property
    def n_downsample(self):
        """disparity field의 해상도를 설정합니다. (1/2^K)"""
        return self._n_downsample

    @n_downsample.setter
    def n_downsample(self, value):
        self._n_downsample = value

    @property
    def context_norm(self):
        """context encoder의 normalization 방법을 설정합니다."""
        return self._context_norm

    @context_norm.setter
    def context_norm(self, value):
        self._context_norm = value

    @property
    def n_gru_layers(self):
        """hidden GRU 레이어의 수를 설정합니다."""
        return self._n_gru_layers

    @n_gru_layers.setter
    def n_gru_layers(self, value):
        self._n_gru_layers = value

    @property
    def shared_backbone(self):
        """context와 feature encoders에서 단일 백본을 사용할지 여부를 설정합니다."""
        return self._shared_backbone

    @shared_backbone.setter
    def shared_backbone(self, value):
        self._shared_backbone = value

    @property
    def mixed_precision(self):
        """mixed precision을 사용할지 여부를 설정합니다."""
        return self._mixed_precision

    @mixed_precision.setter
    def mixed_precision(self, value):
        self._mixed_precision = value

    @property
    def corr_implementation(self):
        """correlation volume의 구현 방법을 설정합니다."""
        return self._corr_implementation

    @corr_implementation.setter
    def corr_implementation(self, value: Literal["reg_cuda", "reg", "alt", "alt_cuda"]):
        self._corr_implementation = value

    @property
    def slow_fast_gru(self):
        """저해상도 GRU를 더 자주 반복할지 여부를 설정합니다."""
        return self._slow_fast_gru

    @slow_fast_gru.setter
    def slow_fast_gru(self, value: bool):
        self._slow_fast_gru = value

    @property
    def restore_ckpt(self):
        """체크포인트를 복원할 경로를 설정합니다."""
        return self._restore_ckpt

    @restore_ckpt.setter
    def restore_ckpt(self, value: Optional[str]):
        self._restore_ckpt = value

    @property
    def lr(self):
        """학습률을 설정합니다."""
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value

    @property
    def train_iters(self):
        """학습 중 flow-field 업데이트 횟수를 설정합니다."""
        return self._train_iters

    @train_iters.setter
    def train_iters(self, value):
        self._train_iters = value

    @property
    def valid_iters(self):
        """유효성 검사 중 flow-field 업데이트 횟수를 설정합니다."""
        return self._valid_iters

    @valid_iters.setter
    def valid_iters(self, value):
        self._valid_iters = value

    @property
    def wdecay(self):
        """weight decay를 설정합니다."""
        return self._wdecay

    @wdecay.setter
    def wdecay(self, value):
        self._wdecay = value

    @property
    def num_steps(self):
        """총 학습 스텝 수를 설정합니다."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        self._num_steps = value

    @property
    def valid_steps(self):
        """유효성 검사를 수행할 스텝 간격을 설정합니다."""
        return self._valid_steps

    @valid_steps.setter
    def valid_steps(self, value):
        self._valid_steps = value

    @property
    def name(self):
        """실험 또는 모델의 이름을 설정합니다."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def batch_size(self):
        """배치 크기를 설정합니다."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def fusion(self):
        """AFF 또는 ConCat 중 어떤 방법으로 fusion할지 설정합니다."""
        return self._fusion

    @fusion.setter
    def fusion(
        self, value: Literal["AFF", "ConCat", "iAFF", "bAFF", "add", "mul", "weight"]
    ):
        self._fusion = value

    @property
    def shared_fusion(self):
        """ExtractorFusion 모듈에서 fusion module을 공유할지 여부를 설정합니다."""
        return self._shared_fusion

    @shared_fusion.setter
    def shared_fusion(self, value):
        self._shared_fusion = value

    @property
    def freeze_backbone(self):
        """Extractor, Updater, Volume 중 어떤 부분을 freeze할지 설정합니다."""
        return self._freeze_backbone

    @freeze_backbone.setter
    def freeze_backbone(
        self, value: List[Literal["Extractor", "Updater", "Volume", "BatchNorm"]]
    ):
        self._freeze_backbone = value

    @property
    def both_side_train(self):
        """양쪽 이미지를 모두 사용하여 학습할지 여부를 설정합니다."""
        return self._both_side_train

    @both_side_train.setter
    def both_side_train(self, value: bool):
        self._both_side_train = value

    @property
    def input_channel(self):
        return self._input_channel

    @input_channel.setter
    def input_channel(self, value):
        self._input_channel = value
