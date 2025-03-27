import os
from typing import Callable
import cv2
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt


class FrameExplorer:
    def __init__(
        self, frame_callback: Callable[[str], None], scene_root: str = "./data"
    ):
        self.scene_root = scene_root
        self.frame_callback = frame_callback

        # Scene path 설정 및 초기화
        self.scene_list = [
            x
            for x in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, x))
        ]
        self.scene_list.sort()
        self.scene_selector = widgets.Dropdown(
            options=self.scene_list,
            description="Scene:",
            value=self.scene_list[0],  # 기본값 첫 번째 scene으로 설정
        )

        # 현재 scene, frame 설정
        self.scene_path = os.path.join(self.scene_root, self.scene_selector.value)
        self.frame_list = self.load_frame_list(self.scene_path)
        self.total_frames = len(self.frame_list)

        # 현재 인덱스
        self.current_index = widgets.BoundedIntText(
            value=0,
            min=0,
            max=self.total_frames - 1,
            description=f"Index (0-{self.total_frames-1}):",
        )

        # 출력 및 UI 위젯 설정
        self.image_display = widgets.Output()
        self.graph_display = widgets.Output()

        # 버튼 설정
        self.prev_button = widgets.Button(description="<< Previous")
        self.next_button = widgets.Button(description="Next >>")
        self.confirm_button = widgets.Button(description="Confirm")

        # 버튼 이벤트 연결
        self.prev_button.on_click(self.on_prev_clicked)
        self.next_button.on_click(self.on_next_clicked)
        self.confirm_button.on_click(self.on_confirm_clicked)

        # 이벤트 연결
        self.current_index.observe(self.update_image, names="value")
        self.scene_selector.observe(self.update_frame_list, names="value")

        # 초기 UI 디스플레이
        self.display_images()
        self.render_ui()

    def load_frame_list(self, scene_path: str):
        """scene_path에 있는 유효한 프레임 리스트를 로드"""
        frame_list = os.listdir(scene_path)
        frame_list = [x for x in frame_list if x.split("_")[-1].isdigit()]
        frame_list.sort()
        return frame_list

    def get_frame_image(self, frame_path: str, src="rgb", side="left"):
        img_path = os.path.join(frame_path, src, f"{side}.png")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".png", "_distorted.png")
        return cv2.imread(img_path)

    def resize_image(self, image_array, scale=0.5):
        """이미지 크기를 scale 배율만큼 줄임"""
        if image_array is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)  # None일 경우 빈 이미지 반환
        width = int(image_array.shape[1] * scale)
        height = int(image_array.shape[0] * scale)
        return cv2.resize(image_array, (width, height), interpolation=cv2.INTER_AREA)

    def show_images_side_by_side(self, images):
        """여러 이미지를 한 줄에 나란히 보여줌"""
        combined_image = np.hstack(images)
        image = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        with BytesIO() as f:
            image.save(f, format="PNG")
            display(widgets.Image(value=f.getvalue(), format="png"))

    def display_images(self):
        with self.image_display:
            self.image_display.clear_output(wait=True)
            frame_path = os.path.join(
                self.scene_path, self.frame_list[self.current_index.value]
            )
            imgs = [
                self.resize_image(
                    self.get_frame_image(frame_path, src="rgb", side="left")
                ),
                self.resize_image(
                    self.get_frame_image(frame_path, src="rgb", side="right")
                ),
                self.resize_image(
                    self.get_frame_image(frame_path, src="nir", side="left")
                ),
                self.resize_image(
                    self.get_frame_image(frame_path, src="nir", side="right")
                ),
            ]
            self.show_images_side_by_side(imgs)

    def update_frame_list(self, change):
        """Scene이 변경될 때 프레임 리스트를 갱신"""
        self.scene_path = os.path.join(self.scene_root, self.scene_selector.value)
        self.frame_list = self.load_frame_list(self.scene_path)
        self.total_frames = len(self.frame_list)
        self.current_index.max = self.total_frames - 1
        self.current_index.description = f"Index (0-{self.total_frames-1}):"
        self.current_index.value = 0  # 첫 프레임으로 초기화
        self.display_images()

    def update_image(self, change):
        """인덱스 변경 시 이미지 업데이트"""
        self.display_images()

    def on_prev_clicked(self, b):
        """이전 프레임으로 이동"""
        if self.current_index.value > 0:
            self.current_index.value -= 1

    def on_next_clicked(self, b):
        """다음 프레임으로 이동"""
        if self.current_index.value < self.total_frames - 1:
            self.current_index.value += 1

    def on_confirm_clicked(self, b):
        """Confirm 클릭 시 그래프 업데이트"""
        selected_frame = os.path.join(
            self.scene_path, self.frame_list[self.current_index.value]
        )
        with self.graph_display:
            self.graph_display.clear_output(wait=True)  # 그래프 영역만 지우기
            self.frame_callback(selected_frame)

    def render_ui(self):
        """전체 UI 구성 및 디스플레이"""
        control_buttons = widgets.HBox(
            [self.prev_button, self.next_button, self.confirm_button]
        )
        display(
            widgets.VBox(
                [
                    self.scene_selector,
                    self.current_index,
                    self.image_display,
                    control_buttons,
                    self.graph_display,
                ]
            )
        )
