import os
import cv2
import numpy as np
import torch
from .hy5py import get_frame_by_path, read_calibration
from .image_process import cv2toTensor, read_image_pair
from .points import project_points_on_camera, transform_point_inverse


def get_lidar_consistency(path: str):
    cal = read_calibration(os.path.dirname(path) + "/0.hdf5")
    mtx = np.load("meta/jai_transform_left.npy")
    mtx_right = np.load("meta/jai_transform_right.npy")

    with get_frame_by_path(path) as f:
        points_left = transform_point_inverse(f["lidar/points"][:] * 1000, mtx)
        points_right = transform_point_inverse(f["lidar/points"][:] * 1000, mtx_right)
        points_left = project_points_on_camera(
            points_left,
            cal["mtx_left"][0, 0],
            cal["mtx_left"][0, 2],
            cal["mtx_left"][1, 2],
        )
        points_right = project_points_on_camera(
            points_right,
            cal["mtx_right"][0, 0],
            cal["mtx_right"][0, 2],
            cal["mtx_right"][1, 2],
        )
        baseline = np.linalg.norm(cal["T"])
        points_left[..., 2] = cal["mtx_left"][0, 0] * baseline / points_left[..., 2]
        points_right[..., 2] = cal["mtx_left"][0, 0] * baseline / points_right[..., 2]
    point_merged = []
    for i in range(points_left.shape[0]):
        u, v, d = points_left[i]
        u2, v2, d2 = points_right[i]
        if (
            d > 0
            and d2 > 0
            and u > 0
            and u2 > 0
            and u < 720
            and u2 < 720
            and v > 0
            and v2 > 0
            and v < 540
            and v2 < 540
        ):
            point_merged.append([u, v, (u - u2)])
    point_merged = np.asarray(point_merged)
    return point_merged


def get_valid_input_from_path(frame_path: str):
    images = read_image_pair(frame_path)
    calibration = read_calibration("/bean/depth/09-08-17-27-33/0.hdf5")
    if os.path.exists(f"{frame_path}/rgb/left_distorted.png"):
        cal = calibration
        K1 = np.array(
            [
                [885.0251763218129, 0, 365.99494891917567],
                [0, 885.6902347857052, 280.8226273769081],
                [0, 0, 1],
            ]
        )
        dist1 = np.array(
            [
                [
                    -0.12932803350642905,
                    0.14596345071537353,
                    0.00022378334058335,
                    0.0007839950632442408,
                    -0.13988730265018726,
                ]
            ]
        )
        K2 = np.array(
            [
                [885.1874714356616, 0, 365.29017354396206],
                [0, 885.4923166475501, 273.8090452919349],
                [0, 0, 1],
            ]
        )
        dist2 = np.array(
            [
                [
                    -0.1391030383143629,
                    0.2077821794527266,
                    0.00028205889437537515,
                    0.0005763560759438774,
                    -0.23620181699687776,
                ]
            ]
        )

        R = cal["R"]
        T = cal["T"]
        T = -T
        R1, R2, P1, P2, Q, do1, do2 = cv2.stereoRectify(
            K1, dist1, K2, dist2, (720, 540), R, T
        )
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, dist1, R1, P1, (720, 540), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, dist2, R2, P2, (720, 540), cv2.CV_32FC1
        )
        images = list(images)
        images[0] = cv2.remap(
            images[0].astype(np.float32),
            map1x,
            map1y,
            cv2.INTER_LINEAR,
        )
        images[1] = cv2.remap(
            images[1].astype(np.float32),
            map2x,
            map2y,
            cv2.INTER_LINEAR,
        )
        images[2] = cv2.remap(
            images[2].astype(np.float32),
            map1x,
            map1y,
            cv2.INTER_LINEAR,
        )
        images[3] = cv2.remap(
            images[3].astype(np.float32),
            map2x,
            map2y,
            cv2.INTER_LINEAR,
        )

    images = [cv2toTensor(x) for x in images]

    lidar_points = get_lidar_consistency(frame_path)
    with get_frame_by_path(frame_path) as frame:

        if "rgb_exposure_left" in frame["image"].attrs:
            print(
                frame["image"].attrs["rgb_exposure_left"],
                frame["image"].attrs["rgb_exposure_right"],
                frame["image"].attrs["nir_exposure_left"],
                frame["image"].attrs["nir_exposure_right"],
            )
        if "disparity/bpnet_occ_map" in frame:
            disparity_gt = frame["disparity/bpnet"][:]
        else:
            disparity_gt = None
    images.append(torch.from_numpy(lidar_points).unsqueeze(0))
    if disparity_gt is not None:
        images.append(torch.from_numpy(disparity_gt).unsqueeze(0).unsqueeze(0))
    return images
