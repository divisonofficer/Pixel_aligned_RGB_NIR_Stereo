import os
import random
from typing import List, Literal, Optional, Tuple, Union
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
import json

import tqdm
from .image_process import (
    apply_patch_gamma_correction_torch,
    crop_and_resize_height,
    guided_filter,
    inputs_disparity_shift,
    img_pad_np,
    pseudo_nir_np,
)
import pfmread
import cv2


DRIVING_JSON = "flyingthings3d.json"
REAL_DATA_JSON = "real_data.json"
FLYING_JSON = "Flow3dFlyingThings3d.json"


class Entity:
    def get_item(
        self,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("You must implement get_item method")


class EntityFlying3d(Entity):

    cut_resolution = (540, 720)

    def __init__(
        self,
        images: List[str],
        disparity: List[str],
        guided_noise=None,
        gamma_noise=None,
        shift_filter=False,
        vertical_scale=False,
        noise_target: Literal["rgb", "nir"] = "rgb",
        disparity_right=False,
        rgb_gt: Optional[Tuple[str]] = None,
    ):
        self.images = images
        self.disparity = disparity
        self.guided_noise = guided_noise
        self.gamma_noise = gamma_noise
        self.shift_filter = shift_filter
        self.vertical_scale = vertical_scale
        self.noise_target = noise_target
        self.shift_distance = random.randint(8, 16)
        self.disparity_right = disparity_right
        self.rgb_gt = rgb_gt

    def __read_img(self, filename):
        if filename.endswith(".pfm"):
            img = pfmread.read(filename)
        else:
            img = np.array(Image.open(filename)).astype(np.uint8)
        if self.cut_resolution is not None and (
            img.shape[0] != self.cut_resolution[0]
            or img.shape[1] != self.cut_resolution[1]
        ):
            img = img_pad_np(img, pad_constant="disp" in filename)
        return img

    def __to_tensor(self, filename: Union[str, np.ndarray]):
        if isinstance(filename, np.ndarray):
            img = filename
        else:
            img = self.__read_img(filename)

        tensor = torch.from_numpy(img.copy())
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).float()

        return tensor.permute(2, 0, 1).float()

    def get_item(
        self,
    ):
        images = [self.__read_img(img) for img in self.images]

        if self.guided_noise is not None:
            if self.noise_target == "rgb":
                images[0] = guided_filter(
                    images[2], images[0], self.guided_noise * 5 + 2, 1e-6
                )
                images[1] = guided_filter(
                    images[3], images[1], self.guided_noise * 5 + 2, 1e-6
                )
                if random.randint(1, 4) == 1:
                    images[0] = images[0] // 50
                    images[1] = images[1] // 50
            else:
                images[2] = guided_filter(
                    images[0].mean(axis=-1),
                    images[2][..., np.newaxis],
                    self.guided_noise + 3,
                    1e-6,
                )
                images[3] = guided_filter(
                    images[1].mean(axis=-1),
                    images[3][..., np.newaxis],
                    self.guided_noise + 3,
                    1e-6,
                )

        images = [self.__to_tensor(img) for img in images]
        if self.gamma_noise is not None:
            if self.noise_target == "rgb":
                images[0] = apply_patch_gamma_correction_torch(images[0].unsqueeze(0))[
                    0
                ]
                images[1] = apply_patch_gamma_correction_torch(images[1].unsqueeze(0))[
                    0
                ]

            else:
                images[2] = apply_patch_gamma_correction_torch(images[2].unsqueeze(0))[
                    0
                ]
                images[3] = apply_patch_gamma_correction_torch(images[3].unsqueeze(0))[
                    0
                ]

        indices = torch.randperm(self.cut_resolution[1] * self.cut_resolution[0])[:5000]
        u = indices % self.cut_resolution[1]
        v = indices // self.cut_resolution[1]

        disparity = self.__to_tensor(self.disparity[0])

        disparity_right = self.__to_tensor(self.disparity[1])
        if self.rgb_gt is not None:
            image_gt = [self.__to_tensor(self.__read_img(x)) for x in self.rgb_gt]
            images += image_gt
        if self.shift_filter:
            images = [x.unsqueeze(0) for x in images]
            if self.rgb_gt is not None:
                images[0] = torch.concat([images[0], images[4]], dim=1)
                images[1] = torch.concat([images[1], images[5]], dim=1)
                images = images[:4]
            images, (disparity, disparity_right) = inputs_disparity_shift(
                images,
                [disparity.unsqueeze(0), disparity_right.unsqueeze(0)],
                self.shift_distance,
            )
            if self.rgb_gt is not None:
                images.append(images[0][:, 3:6])
                images.append(images[1][:, 3:6])
                images[0] = images[0][:, :3]
                images[1] = images[1][:, :3]

            images = [x[0] for x in images]
            disparity, disparity_right = [x[0] for x in [disparity, disparity_right]]
        if self.vertical_scale:
            images[0], images[1] = crop_and_resize_height(
                torch.stack(images[:2], dim=0)
            )
            images[2], images[3] = crop_and_resize_height(
                torch.stack(images[2:4], dim=0)
            )
            if self.rgb_gt is not None:
                images[4:6] = crop_and_resize_height(torch.stack(images[4:6], dim=0))

            disparity = crop_and_resize_height(disparity.unsqueeze(0))[0]
            if self.disparity_right:
                disparity_right = crop_and_resize_height(disparity_right.unsqueeze(0))[
                    0
                ]
        disparity_sampled = disparity[:, v, u]
        disparity_points = torch.stack((u, v, disparity_sampled[0]), dim=0).T.float()
        if self.disparity_right:
            disparity = torch.concat([disparity, disparity_right], dim=0)
        batch = [
            *images,
            disparity_points,
            disparity,
        ]

        return batch


class EntityDataSet(data.Dataset):
    input_list: List[Entity]

    def __init__(self, input_list: List[Entity]):
        self.input_list = random.sample(input_list, len(input_list))

    def __getitem__(self, index):
        return self.input_list[index].get_item()

    def __len__(self):
        return len(self.input_list)


class EntityMiddlebury(Entity):
    def __init__(self, disparity: str, left: str, right: str):
        self.disparity = disparity
        self.left = left
        self.right = right

    def get_item(self) -> Tuple:
        x = random.randint(0, 800)
        y = random.randint(0, 200)

        left = cv2.imread(self.left)[y : y + 540, x : x + 720].astype(np.float32)
        right = cv2.imread(self.right)[y : y + 540, x : x + 720].astype(np.float32)
        left_nir = pseudo_nir_np(left.copy()).astype(np.float32)
        right_nir = pseudo_nir_np(right.copy()).astype(np.float32)
        left_rgb = torch.from_numpy(img_pad_np(left)).permute(2, 0, 1)
        right_rgb = torch.from_numpy(img_pad_np(right)).permute(2, 0, 1)
        left_nir = torch.from_numpy(img_pad_np(left_nir)).unsqueeze(0)
        right_nir = torch.from_numpy(img_pad_np(right_nir)).unsqueeze(0)
        left_rgb_original = left_rgb.clone()
        right_rgb_original = right_rgb.clone()
        disparity_left = pfmread.read(self.disparity).copy()[y : y + 540, x : x + 720]
        disparity_left[np.isinf(disparity_left)] = 100000
        disparity_left[np.isnan(disparity_left)] = 100000
        disparity_left = img_pad_np(disparity_left, pad_constant=True)
        disparity_left = torch.from_numpy(disparity_left).unsqueeze(0)

        nsmt = random.randint(1, 100)
        if nsmt > 70:
            left_rgb = apply_patch_gamma_correction_torch(left_rgb.unsqueeze(0))[0]
            right_rgb = apply_patch_gamma_correction_torch(right_rgb.unsqueeze(0))[0]
        elif nsmt > 20:
            left_rgb = torch.clip(left_rgb - random.randint(64, 224), 0, 255)
            right_rgb = torch.clip(right_rgb - random.randint(64, 224), 0, 255)
        else:
            left_nir = apply_patch_gamma_correction_torch(left_nir.unsqueeze(0))[0]
            right_nir = apply_patch_gamma_correction_torch(right_nir.unsqueeze(0))[0]
        return (
            left_rgb,
            right_rgb,
            left_nir,
            right_nir,
            left_rgb_original,
            right_rgb_original,
            disparity_left,
            disparity_left,
        )


class MiddleburyDataset(EntityDataSet):
    input_list: List[Entity]

    def __init__(self, folder: str = "data/middlebury"):
        self.input_list = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith("disp0.pfm"):
                    for i in range(100):
                        self.input_list.append(
                            EntityMiddlebury(
                                os.path.join(root, file),
                                os.path.join(
                                    root,
                                    file.replace("disp0", "im0").replace("pfm", "png"),
                                ),
                                os.path.join(
                                    root,
                                    file.replace("disp0", "im1").replace("pfm", "png"),
                                ),
                            )
                        )


class StereoDatasetArgs:
    def __init__(
        self,
        folder: str = "/bean/depth",
        flow3d_driving_json=False,
        flying3d_json=False,
        fast_test=False,
        synth_no_filter=False,
        synth_no_rgb=False,
        validate_json=False,
        noised_input=False,
        shift_filter=False,
        vertical_scale=False,
        rgb_rendered=False,
        disparity_right=False,
        rgb_gt=False,
        use_rendered_nir=True,
    ):
        self.folder = folder
        self.flow3d_driving_json = flow3d_driving_json
        self.flying3d_json = flying3d_json
        self.synth_no_filter = synth_no_filter
        self.validate_json = validate_json
        self.synth_no_rgb = synth_no_rgb
        self.fast_test = fast_test
        self.noised_input = noised_input
        self.shift_filter = shift_filter
        self.vertical_scale = vertical_scale
        self.rgb_rendered = rgb_rendered
        self.disparity_right = disparity_right
        self.rgb_gt = rgb_gt
        self.use_rendered_nir = use_rendered_nir


class Ethe3dEntity(Entity):
    def __init__(self, path: str):
        self.path = path

    def get_item(self):
        img_left = cv2.imread(os.path.join(self.path, "im0.png"))
        img_right = cv2.imread(os.path.join(self.path, "im1.png"))
        img_left_nir = pseudo_nir_np(img_left.copy())
        img_right_nir = pseudo_nir_np(img_right.copy())
        H, W = img_left.shape[:2]
        H = min(H, 540)
        W = min(W, 720)
        occ_mask = cv2.imread(
            os.path.join(self.path, "mask0nocc.png"), cv2.IMREAD_GRAYSCALE
        )

        disparity = pfmread.read(os.path.join(self.path, "disp0GT.pfm"))

        images = [
            torch.from_numpy(img_pad_np(x))
            for x in [
                img_left,
                img_right,
                img_left_nir,
                img_right_nir,
                disparity,
                occ_mask,
            ]
        ]
        disparity = images[4]
        occ_mask = images[5]
        disparity[occ_mask < 200] = torch.inf
        indices = torch.randperm(H * W)[:5000]
        u = indices % W
        v = indices // W
        disparity_sampled = disparity[v, u]
        disparity = disparity.unsqueeze(0)
        images[0] = images[0].permute(2, 0, 1)
        images[1] = images[1].permute(2, 0, 1)
        images[2] = images[2].unsqueeze(0)
        images[3] = images[3].unsqueeze(0)

        disparity_points = torch.stack((u, v, disparity_sampled), dim=0).T.float()
        return (*images[:4], disparity_points, disparity)


class Ethe3dStereo(EntityDataSet):
    input_list: List[EntityFlying3d]

    def __init__(self):
        frames = []
        for frame in os.listdir("/bean/eth3d/stereo"):
            path_left = os.path.join("/bean/eth3d/stereo", frame, "im0.png")
            if os.path.exists(path_left):
                frames.append(Ethe3dEntity(os.path.dirname(path_left)))
        self.input_list = random.sample(frames, len(frames))


class StereoDataset(EntityDataSet):
    input_list: List[EntityFlying3d]

    def __init__(
        self,
        args: StereoDatasetArgs,
        copy_of_self=False,
        cut_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.args = args
        self.cut_resolution = cut_resolution
        if copy_of_self:
            return
        self.input_list = []
        if args.flow3d_driving_json:
            self.input_list += self.flow3d_driving_json(
                DRIVING_JSON, args.validate_json
            )
        if args.flying3d_json:
            self.input_list += self.flow3d_driving_json(FLYING_JSON, args.validate_json)

    def flow3d_driving_json(self, filename: str, validate=False):
        with open(filename, "r") as file:
            entries = json.load(file)
        self.entries: List[EntityFlying3d] = []
        validate_entries = []
        error_cnt = 0
        for idx, entry in enumerate(tqdm.tqdm(entries)):

            if self.args.synth_no_filter and "frame_burnt_filtered" in entry:
                continue
            if (
                self.args.fast_test
                and len(self.entries) > 100000
                or (len(self.entries) > 20000 and idx > 1500)
            ):
                break

            if "nir" in entry:
                nir = entry["nir"]
            else:
                nir = (
                    entry["rgb"][0].replace("frames_cleanpass", "nir_rendered"),
                    entry["rgb"][1].replace("frames_cleanpass", "nir_rendered"),
                )
                entry["nir"] = nir
            nir_ambient = [x.replace("nir_rendered", "nir_ambient") for x in nir]
            if not self.args.synth_no_rgb:
                self.entries.append(
                    EntityFlying3d(
                        [*entry["rgb"], *nir_ambient],
                        entry["disparity"],
                        disparity_right=self.args.disparity_right,
                        shift_filter=self.args.shift_filter,
                        rgb_gt=entry["rgb"] if self.args.rgb_gt else None,
                    )
                )
            if self.args.rgb_rendered:
                ir_gb = -1
                ir_nir = -1
                for ir in range(10):
                    i = 9 - ir
                    render_left = (
                        entry["rgb"][0]
                        .replace("frames_cleanpass", "frame_shaded")
                        .replace(".png", f"_{i}.png")
                    )
                    render_right = render_left.replace("left", "right")
                    if os.path.exists(render_left) and os.path.exists(render_right):
                        ir_gb = i
                        break
                if self.args.use_rendered_nir:
                    for ir in range(10):
                        i = 9 - ir
                        render_left = (
                            entry["rgb"][0]
                            .replace("frames_cleanpass", "frame_shaded_nir")
                            .replace(".png", f"_{i}.png")
                        )
                        render_right = render_left.replace("left", "right")
                        if os.path.exists(render_left) and os.path.exists(render_right):
                            ir_nir = i
                            break
                for ir in range(ir_gb):
                    render = (
                        entry["rgb"][0]
                        .replace("frames_cleanpass", "frame_shaded")
                        .replace(".png", f"_{ir}.png")
                    )
                    if self.args.use_rendered_nir:
                        for inr in range(ir_nir):
                            render_nir = (
                                entry["rgb"][0]
                                .replace("frames_cleanpass", "frame_shaded_nir")
                                .replace(".png", f"_{inr}.png")
                            )
                            self.entries.append(
                                EntityFlying3d(
                                    [
                                        render,
                                        render.replace("left", "right"),
                                        render_nir,
                                        render_nir.replace("left", "right"),
                                    ],
                                    entry["disparity"],
                                    shift_filter=self.args.shift_filter,
                                    disparity_right=self.args.disparity_right,
                                    vertical_scale=self.args.vertical_scale,
                                    guided_noise=(
                                        random.randint(3, 8)
                                        if self.args.noised_input
                                        else None
                                    ),
                                    rgb_gt=entry["rgb"] if self.args.rgb_gt else None,
                                )
                            )
                    else:
                        self.entries.append(
                            EntityFlying3d(
                                [
                                    render,
                                    render.replace("left", "right"),
                                    nir[0],
                                    nir[1],
                                ],
                                entry["disparity"],
                                shift_filter=self.args.shift_filter,
                                disparity_right=self.args.disparity_right,
                                vertical_scale=self.args.vertical_scale,
                                guided_noise=(
                                    random.randint(3, 8)
                                    if self.args.noised_input
                                    else None
                                ),
                                rgb_gt=entry["rgb"] if self.args.rgb_gt else None,
                            )
                        )

            if self.args.noised_input:
                for _ in range(5):
                    for t in ["rgb", "nir"]:
                        self.entries.append(
                            EntityFlying3d(
                                [*entry["rgb"], *nir_ambient],
                                entry["disparity"],
                                guided_noise=int((random.random() * 100) % 20),
                                gamma_noise=(random.random() * 2),
                                shift_filter=self.args.shift_filter,
                                vertical_scale=self.args.vertical_scale,
                                # noise_target=t,
                                disparity_right=self.args.disparity_right,
                                rgb_gt=entry["rgb"] if self.args.rgb_gt else None,
                            )
                        )
            if validate:
                validated = True
                for key, value in entry.items():
                    if (isinstance(value, str) and not os.path.exists(value)) or (
                        isinstance(value, tuple)
                        and (
                            not os.path.exists(value[0]) or not os.path.exists(value[1])
                        )
                    ):
                        validated = False
                        break
                try:
                    disparity = pfmread.readPFM(entry["disparity"][0])
                except Exception as e:
                    validated = False

                if validated:
                    validate_entries.append(entry)

        if validate:
            print("Error Cnt ", error_cnt)
            with open(filename, "w") as file:
                json.dump(validate_entries, file)
        return self.entries

    def __getitem__(self, index):
        """
        return: file_name_list, (img_viz_left, img_viz_right, img_nir_left, img_nir_right)
        """
        return self.input_list[index].get_item()

    def __len__(self):
        return len(self.input_list)
