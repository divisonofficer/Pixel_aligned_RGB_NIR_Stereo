import os
from typing import List, Optional, Tuple
import numpy as np
import tqdm
import pfmread
import cv2
import torch
import torch.nn.functional as F
import random
from IPython import get_ipython


class ImageItem:
    image: torch.Tensor
    depth_map: torch.Tensor
    material: torch.Tensor
    albedo: Optional[torch.Tensor]

    def __init__(self, image, depth_map, material, albedo=None):
        self.image = image
        self.depth_map = depth_map
        self.material = material
        self.albedo = albedo


class AugmentRender:
    def depth_to_normal_map(self, depth_map: torch.Tensor, fx: float, fy: float):
        # Calculate depth map gradients in x and y directions using Sobel filter
        dz_dx = F.conv2d(
            depth_map.unsqueeze(0).unsqueeze(0),
            torch.tensor(
                [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                device=depth_map.device,
                dtype=depth_map.dtype,
            )
            / 8,
            padding=1,
        ).squeeze()
        dz_dy = F.conv2d(
            depth_map.unsqueeze(0).unsqueeze(0),
            torch.tensor(
                [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                device=depth_map.device,
                dtype=depth_map.dtype,
            )
            / 8,
            padding=1,
        ).squeeze()

        # Calculate normal vectors
        normal_x = -dz_dx  # / fx
        normal_y = -dz_dy  # / fy
        normal_z = torch.ones_like(depth_map)

        normals = torch.stack((normal_x, normal_y, normal_z), dim=-1)
        norm = torch.norm(normals, dim=-1, keepdim=True)
        normals = normals / (norm + 1e-8)  # Normalize and prevent division by zero
        return normals

    def render_diffuse(self, normal_map: torch.Tensor, light_direction: torch.Tensor):
        # Calculate the dot product between the normal map and the light direction
        diffuse = torch.clamp(torch.sum(normal_map * -light_direction, dim=-1), 0, 1)
        return diffuse

    def render_specular(
        self,
        normal_map: torch.Tensor,
        light_direction: torch.Tensor,
        view_direction: torch.Tensor,
        shininess: float,
    ):
        # Calculate the reflection direction
        reflection = (
            2
            * torch.sum(normal_map * -light_direction, dim=-1, keepdim=True)
            * normal_map
            - light_direction
        )

        # Calculate the dot product between the reflection direction and the view direction
        specular = (
            torch.clamp(torch.sum(reflection * view_direction, dim=-1), 0, 1)
            ** shininess
        )
        return specular

    def compute_pointcloud(self, K: torch.Tensor, depth_map: torch.Tensor):
        height, width = depth_map.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device),
            indexing="ij",
        )

        # Use camera intrinsic parameters fx, fy
        fx, fy = K[0, 0], K[1, 1]
        p_x = (x - width / 2) / fx * depth_map
        p_y = (y - height / 2) / fy * depth_map
        p_z = depth_map
        points = torch.stack((p_x, p_y, p_z), dim=-1)
        return points

    def render_attenuation(
        self,
        K: torch.Tensor,
        depth_map: torch.Tensor,
        light_position: torch.Tensor,
        attenuation_factor: float = 1.0,
    ):
        points = self.compute_pointcloud(K, depth_map)
        # Calculate distance to the light source
        light_distance = torch.norm(points - light_position, dim=-1)

        # Calculate attenuation based on distance to the light source
        attenuation = 1 / (1 + attenuation_factor * light_distance**2)

        return attenuation, light_distance

    def compute_spot_factor(
        self,
        light_direction: torch.Tensor,
        spot_direction: torch.Tensor,
        spot_exponent: float,
        cutoff_angle: float,
    ):
        # Calculate the angle between the light direction and spotlight direction
        cos_angle = light_direction * spot_direction.unsqueeze(0).unsqueeze(0)

        # Calculate the spotlight attenuation factor based on the cutoff angle
        spot_factor = torch.clamp(
            (cos_angle - torch.cos(torch.tensor(cutoff_angle)))
            / (
                torch.cos(torch.tensor(spot_exponent))
                - torch.cos(torch.tensor(cutoff_angle))
            ),
            0,
            1,
        )
        return spot_factor

    def render(
        self,
        K: torch.Tensor,
        depth_map: torch.Tensor,
        light_position: torch.Tensor,
        view_direction: torch.Tensor,
        shininess: float,
        texture: torch.Tensor,
        intensity: float = 1.0,
        spot_exp=torch.pi / 4,
        spot_cutoff=torch.pi / 16,
    ):
        fx, fy = K[0, 0], K[1, 1]
        normal_map = self.depth_to_normal_map(depth_map.clone(), fx, fy)
        points = self.compute_pointcloud(K, depth_map)
        light_direction = light_position - points

        light_direction = light_direction / torch.norm(
            light_direction, dim=-1, keepdim=True
        )

        diffuse = self.render_diffuse(normal_map, light_direction.clone())

        specular = self.render_specular(
            normal_map, light_direction.clone(), view_direction, shininess
        )
        spot_factor = self.compute_spot_factor(
            -light_direction.clone(), view_direction, spot_exp, spot_cutoff
        )
        attenuation, light_distance = self.render_attenuation(
            K, depth_map, light_position
        )
        return (
            intensity * attenuation * (diffuse + specular) * (1 - spot_factor[..., -1])
        )[..., None] * texture, [
            depth_map,
            diffuse,
            specular,
            spot_factor,
            attenuation,
            light_distance,
        ]

    def peusdo_nir(self, rgb):
        # Reverse the channels and use torch.maximum
        interm = torch.maximum(rgb, 1 - rgb)

        # Compute the weighted sum and apply the power operation
        nir = (
            interm[..., 0] * 0.229 + interm[..., 1] * 0.587 + interm[..., 2] * 0.114
        ) ** (1 / 0.25)

        return nir

    def material_albedo(self, image, material):
        image_np = image.float().cpu().numpy()
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        v = image_hsv[..., 2] / 512
        v = v**0.3
        image_hsv[..., 2] = v * 255
        image_np = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        image = torch.from_numpy(image_np).cuda().float()

        values = torch.linspace(0, 1080, 1080).cuda()
        values = torch.zeros_like(values)
        values = torch.stack([values, values, values], dim=-1)
        values[material.long().squeeze(-1)] = image
        texture = values[material.long().squeeze(-1)] * 0.9 + 0.1 * image
        return texture


class RandomAugmentRender:
    def __init__(self):
        self.core = AugmentRender()

    def random_light_render(
        self,
        K_tensor,
        baseline,
        item_left: ImageItem,
        item_right: ImageItem,
        depth_range=[3, 30],
        intensity_range=[2, 8],
        spot_exp=np.pi / 4,
        ambient_cnt=20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = item_left.image.device
        image_left, depth_map_left, _ = (
            item_left.image,
            item_left.depth_map,
            item_left.material,
        )
        image_right, depth_map_right, _ = (
            item_right.image,
            item_right.depth_map,
            item_right.material,
        )
        shaded_left = torch.zeros_like(image_left, dtype=torch.float32).to(device)
        shaded_right = torch.zeros_like(image_right, dtype=torch.float32).to(device)
        if item_left.albedo is not None:
            albedo_left = item_left.albedo
            albedo_right = item_right.albedo
        else:
            albedo_left = self.core.material_albedo(item_left.image, item_left.material)
            albedo_right = self.core.material_albedo(
                item_left.image, item_left.material
            )
        for i in range(ambient_cnt):
            light_z = random.randint(*depth_range)
            intensity = 2 ** random.randint(*intensity_range)
            shininess = 64
            light_position = torch.tensor(
                [
                    random.randrange(-1, 1) * (light_z),
                    random.randrange(-2, 1) * light_z,
                    light_z,
                ],
                dtype=torch.float32,
            ).to(device)
            shaded, _ = self.core.render(
                K_tensor,
                depth_map_left,
                light_position=light_position
                + torch.tensor([baseline / 2, 0, 0], dtype=torch.float32).to(device),
                view_direction=torch.tensor([0, 0, 1], dtype=torch.float32).to(device),
                shininess=shininess,
                texture=albedo_left / 255,
                intensity=intensity,
                spot_exp=spot_exp,
            )
            shaded_left += shaded
            shaded, _ = self.core.render(
                K_tensor,
                depth_map_right,
                light_position=light_position
                - torch.tensor([baseline / 2, 0, 0], dtype=torch.float32).to(device),
                view_direction=torch.tensor([0, 0, 1], dtype=torch.float32).to(device),
                shininess=shininess,
                texture=albedo_right / 255,
                intensity=intensity,
                spot_exp=spot_exp,
            )
            shaded_right += shaded
        shaded_left = torch.clip(shaded_left - 0.1, 0, 1)
        shaded_right = torch.clip(shaded_right - 0.1, 0, 1)
        return shaded_left, shaded_right

    def random_nir_render(
        self,
        item_left: ImageItem,
        item_right: ImageItem,
        K_tensor: torch.Tensor,
        baseline=1,
        shiniess=32,
        spot_exp_range=[2, 5],
        spot_cutoff=torch.pi / 16,
        intensity_range=[10, 100],
    ):
        device = item_left.image.device
        light_position = torch.tensor([0, random.randint(-3, 3), -2]).to(device)
        intensity = random.randint(*intensity_range)
        spot_exp = torch.pi / (random.randint(*spot_exp_range))
        albedo_left = self.core.peusdo_nir(item_left.albedo / 255)
        albedo_right = self.core.peusdo_nir(item_right.albedo / 255)
        left, _ = self.core.render(
            K_tensor,
            item_left.depth_map,
            light_position=light_position
            + torch.tensor([baseline / 2, 0, 0], dtype=torch.float32).to(device),
            view_direction=torch.tensor([0, 1, 1], dtype=torch.float32).to(device),
            shininess=shiniess,
            texture=albedo_left[..., None] * 2,
            intensity=intensity,
            spot_exp=spot_exp,
            spot_cutoff=spot_cutoff,
        )
        right, _ = self.core.render(
            K_tensor,
            item_right.depth_map,
            light_position=light_position
            - torch.tensor([baseline / 2, 0, 0], dtype=torch.float32).to(device),
            view_direction=torch.tensor([0, 0, 1], dtype=torch.float32).to(device),
            shininess=shiniess,
            texture=albedo_right[..., None] * 2,
            intensity=intensity,
            spot_exp=spot_exp,
            spot_cutoff=spot_cutoff,
        )
        return torch.clip(
            left.squeeze(-1) - 0.1 + (albedo_left * 0.1 - 0.05), 0, 1
        ), torch.clip(right.squeeze(-1) - 0.1 + (albedo_right * 0.1 - 0.05), 0, 1)


class SceneFlowConfig:
    def __init__(
        self,
        K_tensor=None,
        baseline=1,
        augment_cnt=10,
        ambient_cnt=20,
        term="normal_map",
        rgb_intensity_range=[2, 9],
        rgb_depth_range=[-10, 30],
        rgb_spot_exp=np.pi / 6,
        nir_shiniess=32,
        nir_spot_exp_range=[2, 5],
        nir_spot_cutoff=torch.pi / 16,
        nir_intensity_range=[10, 100],
    ):
        # 기본 카메라 매트릭스 설정
        self.K_tensor = (
            K_tensor
            if K_tensor is not None
            else np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])
        )
        self.baseline = baseline  # 기본 기준 거리
        self.augment_cnt = augment_cnt  # 증강 횟수
        self.ambient_cnt = ambient_cnt  # 주변광 개수
        self.term = term  # 데이터 타입(term)
        self.rgb_intensity_range = rgb_intensity_range
        self.rgb_depth_range = rgb_depth_range
        self.rgb_spot_exp = rgb_spot_exp
        self.nir_shiniess = nir_shiniess
        self.nir_spot_exp_range = nir_spot_exp_range
        self.nir_spot_cutoff = nir_spot_cutoff
        self.nir_intensity_range = nir_intensity_range

    def update(self, **kwargs):
        """Config 값 업데이트를 위한 메서드"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config attribute: {key}")

    def __repr__(self):
        """Config 정보 출력용"""
        return (
            f"Flying3DConfig(K_tensor={self.K_tensor}, baseline={self.baseline}, "
            f"augment_cnt={self.augment_cnt}, ambient_cnt={self.ambient_cnt}, term='{self.term}')"
        )


class SceneFlowAugmentRender(RandomAugmentRender):
    def __init__(self):
        super().__init__()

    def warp_map(self, single_channel_map, disparity_map):
        H, W = single_channel_map.shape
        # X, Y 좌표 생성
        y_coords, x_coords = torch.meshgrid(
            torch.arange(0, H, device=disparity_map.device),
            torch.arange(0, W, device=disparity_map.device),
            indexing="ij",
        )
        # 오른쪽 뷰의 좌표 생성
        x_coords_right = x_coords - disparity_map
        y_coords_right = y_coords.float()
        # 정규화된 좌표로 변환 ([-1, 1] 범위)
        x_coords_norm = 2.0 * x_coords_right / (W - 1) - 1.0
        y_coords_norm = 2.0 * y_coords_right / (H - 1) - 1.0
        # Grid 생성 (H, W, 2)
        grid = torch.stack((x_coords_norm, y_coords_norm), dim=-1)
        # Grid sample을 사용해 warping
        single_channel_map = single_channel_map.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, H, W)
        warped_map = F.grid_sample(
            single_channel_map,
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped_map.squeeze(0).squeeze(0)  # (H, W)

    def read_sceneflow_frame(
        self,
        frame_path: str,
        term="normal_map",
        K=np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]]),
        baseline=1,
        right_label=True,
    ):
        left_path = frame_path.replace(term, "frames_cleanpass").replace("pfm", "png")
        if not os.path.exists(left_path):
            left_path = left_path.replace("frames_cleanpass", "RGB_cleanpass")
        image_left = cv2.imread(left_path)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        image_right = cv2.imread(left_path.replace("/left/", "/right/"))
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

        if right_label:
            disparity_map = pfmread.read(
                frame_path.replace(term, "disparity").replace(".png", ".pfm")
            )
            material_index = pfmread.read(
                frame_path.replace(term, "material_index").replace(".png", ".pfm")
            )
            disparity_map_right = pfmread.read(
                frame_path.replace(term, "disparity")
                .replace("left", "right")
                .replace(".png", ".pfm")
            )
            material_index_right = pfmread.read(
                frame_path.replace(term, "material_index")
                .replace("left", "right")
                .replace(".png", ".pfm")
            )
        else:
            disparity_map = pfmread.read(
                frame_path.replace(term, "disparity")
                .replace("/left/", "/")
                .replace(".png", ".pfm")
            )
            material_index = pfmread.read(
                frame_path.replace(term, "material_index")
                .replace("/left/", "/")
                .replace(".png", ".pfm")
            )

        material_index = torch.from_numpy(material_index.copy())
        disparity_map = torch.from_numpy(disparity_map.copy())

        if right_label:
            disparity_map_right = torch.from_numpy(disparity_map_right.copy())
            material_index_right = torch.from_numpy(material_index_right.copy())
        else:
            disparity_map_right = self.warp_map(disparity_map.clone(), disparity_map)
            material_index_right = self.warp_map(material_index.clone(), disparity_map)

        material_index = material_index.unsqueeze(-1)
        material_index_right = material_index_right.unsqueeze(-1)

        image_left = torch.from_numpy(image_left).float()
        image_right = torch.from_numpy(image_right).float()

        depth_map = K[0, 0] * baseline / disparity_map
        depth_map_right = K[0, 0] * baseline / disparity_map_right

        albedo = self.core.material_albedo(
            torch.concat([image_left, image_right], dim=0),
            torch.concat([material_index, material_index_right], dim=0),
        )

        return ImageItem(
            image_left.cuda(),
            depth_map.cuda(),
            material_index.cuda(),
            albedo[: image_left.shape[0]].cuda(),
        ), ImageItem(
            image_right.cuda(),
            depth_map_right.cuda(),
            material_index_right.cuda(),
            albedo[image_left.shape[0] :].cuda(),
        )

    def sceneflow_search_frames(self, dataset_root: str):
        paths = []
        for root, dirs, files in os.walk(
            dataset_root
        ):  # os.walk를 사용하여 재귀적으로 탐색
            if os.path.basename(root) == "left":  # 현재 디렉토리가 'left'인지 확인
                for file in files:
                    if file.endswith(".png"):  # 확장자가 .png인 파일만 선택
                        paths.append(os.path.join(root, file))
        paths.sort()
        return paths

    def sceneflow_frame_augment(self, frame_path: str, config: SceneFlowConfig):
        item_left, item_right = self.read_sceneflow_frame(frame_path, config.term)
        for i in range(config.augment_cnt):
            shaded_left, shaded_right = self.random_light_render(
                config.K_tensor,
                config.baseline,
                item_left,
                item_right,
                intensity_range=config.rgb_intensity_range,
                depth_range=config.rgb_depth_range,
                spot_exp=config.rgb_spot_exp,
                ambient_cnt=config.ambient_cnt,
            )
            store_path_left = (
                frame_path.replace(config.term, "frame_shaded")
                .replace(".pfm", ".png")
                .replace(".png", f"_{i}.png")
            )
            store_path_right = store_path_left.replace("left", "right")
            os.makedirs(os.path.dirname(store_path_left), exist_ok=True)
            os.makedirs(os.path.dirname(store_path_right), exist_ok=True)
            cv2.imwrite(
                store_path_left, (shaded_left.cpu().numpy() * 255).astype(np.uint8)
            )
            cv2.imwrite(
                store_path_right, (shaded_right.cpu().numpy() * 255).astype(np.uint8)
            )

            nir_left, nir_right = self.random_nir_render(
                item_left,
                item_right,
                config.K_tensor,
                config.baseline,
                shiniess=config.nir_shiniess,
                spot_exp_range=config.nir_spot_exp_range,
                spot_cutoff=config.nir_spot_cutoff,
                intensity_range=config.nir_intensity_range,
            )
            store_path_left = store_path_left.replace(
                "frame_shaded", "frame_shaded_nir"
            )
            store_path_right = store_path_left.replace("left", "right")
            os.makedirs(os.path.dirname(store_path_left), exist_ok=True)
            os.makedirs(os.path.dirname(store_path_right), exist_ok=True)
            cv2.imwrite(
                store_path_left,
                (np.clip(nir_left.cpu().numpy(), 0, 1) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                store_path_right,
                (np.clip(nir_right.cpu().numpy(), 0, 1) * 255).astype(np.uint8),
            )

    def sceneflow_dataset_augment(
        self, paths: List[str], config=SceneFlowConfig(), _tqdm=None
    ):
        # Jupyter 환경 감지
        if _tqdm is None:
            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                # Jupyter Notebook 환경
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            _tqdm = tqdm

        for path in _tqdm(paths):
            self.sceneflow_frame_augment(path, config)
