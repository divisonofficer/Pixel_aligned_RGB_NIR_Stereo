import random
from typing import List, Optional, Tuple
import cv2
import h5py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .hy5py import calibration_property, read_lidar
from .image_process import (
    img_pad_np,
    pad_lidar_points,
    project_points_on_camera,
    refine_disparity_points,
    transform_point_inverse,
)

from train_fusion.dataloader import Entity, EntityDataSet


class MyH5Entity(Entity):
    def __init__(
        self,
        h5_path,
        frame_path,
        shift: Optional[int] = None,
        is_refined_gt=False,
        bpnet_gt=False,
    ):
        self.h5_path = h5_path
        self.frame_path = frame_path
        self.shift = shift
        self.is_refined_gt = is_refined_gt
        self.bpnet_gt = bpnet_gt

    def imread(self, path: str, gray=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_ANYCOLOR)
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_pad_np(img)
        tensor = torch.from_numpy(img).float()
        if gray:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1)

        if tensor.shape[-2] < 540:
            padding = (
                (720 - tensor.shape[-1]) // 2,
                (720 - tensor.shape[-1]) // 2,
                (540 - tensor.shape[-2]) // 2,
                (540 - tensor.shape[-2]) // 2,
            )
            tensor = F.pad(tensor, padding)

        return tensor

    def get_item(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h5_path = self.h5_path
        frame_path = self.frame_path
        if type(h5_path) == bytes:
            h5_path = h5_path.decode("utf-8")
            frame_path = frame_path.decode("utf-8")
        transform_mtx = np.load("jai_transform.npy")
        resolution = cv2.imread(os.path.join(frame_path, "rgb", "left.png")).shape
        with h5py.File(h5_path, "r", swmr=True) as f:
            frame = f.require_group(f"frame/{frame_path.split('/')[-1]}")
            focal_length, baseline, cx, cy = calibration_property(
                f["calibration"].attrs
            )

            lidar_points = read_lidar(frame)
            lidar_points = transform_point_inverse(lidar_points, transform_mtx)
            lidar_projected_points = project_points_on_camera(
                lidar_points, focal_length, cx, cy, resolution[1], resolution[0]
            )
            cx = 351.19399470967926
            cx2 = 351.89804045
            lidar_projected_points[:, 2] = (
                focal_length * baseline / lidar_projected_points[:, 2] - 1
            )
            lidar_projected_points = refine_disparity_points(
                torch.from_numpy(lidar_projected_points),
            ).numpy()

            lidar_projected_points = pad_lidar_points(lidar_projected_points, 5000)
            # disparity = frame["depth_bpnet/near"][:][:540, :720]
            # disparity = torch.from_numpy(disparity).unsqueeze(0)
            # disparity[disparity > 64] = 0
            # monodepth_rgb = frame["depth_mono/rgb"][:].squeeze()
            # monodepth_nir = frame["depth_mono/nir"][:].squeeze()
            if self.is_refined_gt:
                disparity = frame["disparity/bpnet"][:] + (
                    self.shift if self.shift is not None else 0
                )

            elif self.bpnet_gt:
                disparity = frame["disparity/bpnet"][:]
                disparity[disparity < 12] = 10000
                disparity = disparity + (self.shift if self.shift is not None else 0)
                mask = frame["disparity/bpnet_occ_map"][:]
                disparity[np.isinf(disparity)] = 10000
                disparity[np.isnan(disparity)] = 10000
            else:
                disparity = np.zeros((540, 720), np.float32) - 1
            # disparity = refine_disparity_with_monodepth(disparity, monodepth_nir)

            # disparity_rgb = frame["disparity/rgb"][:].squeeze()[:540, :720]
            # disparity_nir = frame["disparity/nir"][:].squeeze()[:540, :720]
            #
            # disparity_rgb = refine_disparity_with_monodepth(
            #     disparity_rgb, monodepth_rgb
            # )
            # rgb_left = cv2.imread(
            #     os.path.join(frame_path, "rgb", "left.png"), cv2.IMREAD_COLOR
            # )
            # nir_left = cv2.imread(
            #     os.path.join(frame_path, "nir", "left.png"), cv2.IMREAD_GRAYSCALE
            # )
            # disparity_nir = refine_disparity_with_monodepth(
            #     disparity_nir, monodepth_nir
            # )
            # disparity = combine_disparity_by_edge(
            #     lidar_projected_points, disparity_rgb, disparity_nir, rgb_left, nir_left
            # )
            disparity = img_pad_np(disparity)
            disparity = torch.from_numpy(disparity).unsqueeze(-1).permute(2, 0, 1)

        lidar_projected_points = torch.from_numpy(lidar_projected_points).float()

        if resolution[0] < 540:
            lidar_projected_points[:, 1] += (540 - resolution[0]) // 2
            lidar_projected_points[:, 0] += (720 - resolution[1]) // 2
        rgb_left = self.imread(os.path.join(frame_path, "rgb", "left.png"))
        rgb_right = self.imread(os.path.join(frame_path, "rgb", "right.png"))
        nir_left = self.imread(os.path.join(frame_path, "nir", "left.png"), gray=True)
        nir_right = self.imread(os.path.join(frame_path, "nir", "right.png"), gray=True)
        if self.shift is not None:
            rgb_right = torch.roll(rgb_right, shifts=-self.shift, dims=-1)
            nir_right = torch.roll(nir_right, shifts=-self.shift, dims=-1)
            lidar_projected_points[:, 2] += self.shift

        return (
            rgb_left,
            rgb_right,
            nir_left,
            nir_right,
            lidar_projected_points,
            disparity,
        )


class MyRefinedH5DataSet(EntityDataSet):
    def __init__(self, root="/bean/depth/refined", use_right_shift=False):
        self.transform_mtx = np.load("jai_transform.npy")
        self.use_right_shift = use_right_shift
        frame_id_ret: List[MyH5Entity] = []
        h5_file = os.path.join(root, "0.hdf5")
        with h5py.File(h5_file, "r", swmr=True) as f:
            frame_ids = list(f["frame"].keys())
            for frame_id in frame_ids:
                frame = f.require_group(f"frame/{frame_id}")
                if "disparity" in frame:
                    frame_id_ret.append(
                        MyH5Entity(
                            h5_file,
                            os.path.join(os.path.dirname(h5_file), frame_id),
                            (random.randint(0, 36) if self.use_right_shift else None),
                            is_refined_gt=True,
                        )
                    )
        self.input_list = frame_id_ret


class MyH5DataSet(EntityDataSet):
    def __init__(
        self,
        root="/bean/depth",
        fast_test=False,
        frame_cache=False,
        update_cache=False,
        use_right_shift=False,
        scene_list: List[str] = None,
        bpnet_gt=False,
    ):
        self.transform_mtx = np.load("jai_transform.npy")
        self.frame_cache = frame_cache
        self.update_cache = update_cache
        self.use_right_shift = use_right_shift
        ## find h5 files
        self.bpnet_gt = bpnet_gt
        if scene_list is not None:
            scene_list = [os.path.join("/bean/depth", x) for x in scene_list]
            h5files = [
                os.path.join(x, y)
                for x in scene_list
                for y in os.listdir(x)
                if y.endswith(".hdf5")
            ]
        else:
            h5files = self.find_h5_files(root)
        h5files.sort()
        ## read h5 files and get frame_id list
        frame_id_list: List[MyH5Entity] = []
        for h5file in tqdm(h5files):
            try:
                frame_id_list += list(self.read_h5_file(h5file))
            except OSError:
                print(f"{h5file} is not prepared yet")
                break

            if fast_test and len(frame_id_list) > 100:
                break

        self.input_list = []
        for entity in tqdm(frame_id_list):
            self.input_list.append(entity)
            if self.use_right_shift:
                self.input_list.append(
                    MyH5Entity(
                        entity.h5_path, entity.frame_path, random.randint(24, 36)
                    )
                )

        self.input_list = random.sample(self.input_list, len(self.input_list))

    def __len__(self):
        return len(self.input_list)

    def find_h5_files(self, root):
        h5_files = []
        for folder in os.listdir(root):
            folder = os.path.join(root, folder)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith(".hdf5"):
                    h5_files.append(os.path.join(folder, file))
        return h5_files

    def read_h5_file(self, h5_file):
        frame_id_ret: List[MyH5Entity] = []
        id_cache: List[Tuple[str, str]] = []
        with h5py.File(h5_file, "a" if self.update_cache else "r", swmr=True) as f:
            if self.frame_cache and "frame_ids" in f:
                frame_ids = f["frame_ids"][:]
                f.close()
                return [MyH5Entity(*x) for x in frame_ids[:]]
            frame_ids = list(f["frame"].keys())
            for frame_id in frame_ids:
                frame = f.require_group(f"frame/{frame_id}")
                if "disparity" in frame:
                    if "align_error" in frame.attrs and frame.attrs["align_error"]:

                        continue
                    if (
                        "exposure_error" in frame.attrs
                        and frame.attrs["exposure_error"]
                    ):

                        continue
                    # if (
                    #     "lidar_align_error" in frame.attrs
                    #     and frame.attrs["lidar_align_error"]
                    # ):
                    #     print("lidar align error")
                    #     continue
                    if self.bpnet_gt:
                        if not "disparity/bpnet_occ_map" in frame:

                            continue
                        frame_id_ret.append(
                            MyH5Entity(
                                h5_file,
                                os.path.join(os.path.dirname(h5_file), frame_id),
                                bpnet_gt=True,
                            )
                        )
                        continue
                    frame_id_ret.append(
                        MyH5Entity(
                            h5_file, os.path.join(os.path.dirname(h5_file), frame_id)
                        )
                    )
                    id_cache.append(
                        (
                            h5_file,
                            os.path.join(os.path.dirname(h5_file), frame_id),
                        )
                    )
            if self.update_cache:
                if "frame_ids" in f:
                    del f["frame_ids"]
                f.create_dataset("frame_ids", data=id_cache)
            f.close()
        return frame_id_ret

    def __getitem__(self, index):
        return self.input_list[index].get_item()
