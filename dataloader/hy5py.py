import os
from typing import Callable, Tuple, Union
import h5py
import numpy as np


def read_calibration(h5path: Union[str, h5py.File]):
    """
    Read calibration attributes from an HDF5 file.
    args:
        h5path: str or h5py.File
            Path to the HDF5 file or h5py.File object.
    returns:
        dict
            Dictionary of calibration attributes.
    """
    if isinstance(h5path, h5py.File):
        return dict(h5path["calibration"].attrs)
    with h5py.File(h5path, "r") as f:
        return dict(f["calibration"].attrs)


def read_lidar(frame: h5py.Group, scale=1000) -> np.ndarray:
    """
    Read LiDAR data from a frame.
    args:
        frame: h5py.Group
            HDF5 group of a frame.
    returns:
        np.ndarray
            LiDAR data.
    """
    return frame["lidar/points"][:] * scale


def calibration_property(calibration: dict) -> Tuple[float, float, float, float]:
    """
    Get calibration properties from a calibration dictionary.
    args:
        calibration: dict
            Dictionary of calibration attributes.
    returns:
        focal_length: float
        baseline: float
        cx: float
        cy: float
    """
    focal_length = calibration["mtx_left"][0, 0]
    cx = calibration["mtx_left"][0, 2]
    cy = calibration["mtx_left"][1, 2]
    baseline = np.linalg.norm(calibration["T"][:])
    return (
        float(focal_length),
        float(baseline),
        float(cx),
        float(cy),
    )


class FrameContext:
    def __init__(self, h5file: str, frame_id: Union[int, str]):
        self.h5file = h5file
        self.frame_id = frame_id
        self.file = None
        self.group = None

    def __enter__(self) -> h5py.Group:
        self.file = h5py.File(self.h5file, "r")
        if isinstance(self.frame_id, int):
            frame_ids = list(self.file["frame"])
            self.frame_id = frame_ids[self.frame_id]
        print(f"Opening frame {self.frame_id} from {self.h5file}")
        self.group = self.file.require_group(f"frame/{self.frame_id}")
        return self.group

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if exc_type:
            print(f"An exception occurred: {exc_val}")
        # 예외를 처리하지 않고 다시 발생시키려면 False를 반환
        return False


def get_frame_in_h5(h5file: str, frame_id: Union[int, str]) -> FrameContext:
    """
    Get a frame from an HDF5 file.
    args:
        h5file: str
            Path to the HDF5 file.
        frame_id: int
            Frame ID.
    returns:
        h5py.Group
            HDF5 group of the frame.
    """
    return FrameContext(h5file, frame_id)


def get_frame_by_path(frame_path: str) -> FrameContext:
    """
    Get a frame from an HDF5 file.
    args:
        frame_path: str
            Path to the HDF5 file.
    returns:
        h5py.Group
            HDF5 group of the frame.
    """
    frame_dir = os.path.dirname(frame_path)
    frame_id = frame_path.split("/")[-1].split(".")[0]
    frame_list = os.listdir(frame_dir)
    frame_list = [f for f in frame_list if f.split("_")[-1].isdigit()]
    frame_list.sort()
    frame_idx = frame_list.index(frame_id)
    scene_idx = frame_idx // 100
    return get_frame_in_h5(f"{frame_dir}/{scene_idx}.hdf5", frame_id)


def process_frames(
    h5file: str, frame_callback: Callable[[h5py.Group], None], update=False
):
    """
    Process frames in an HDF5 file.
    args:
        h5file: str
            Path to the HDF5 file.
        frame_callback: Callable[[h5py.Group], None]
            Function to call for each frame.
    """
    with h5py.File(h5file, "a" if update else "r") as f:
        for frame_id in f["frame"]:
            frame_callback(f.require_group(f"frame/{frame_id}"))
