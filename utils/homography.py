import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

# преобразует координаты изображения в координаты плоскости пола/карты
class HomographyTransformer:

    def __init__(self):
        self.homography_matrix = None
        self.calibrated = False
        self.has_real_scale = False
        self.real_width_m = 1.0
        self.real_height_m = 1.5

    # калибрует гомографию по соответствиям точек
    # аргументы: список координат в изображении и карты пола, возвращает true если калибровка успешна
    def calibrate(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        real_width_m: Optional[float] = None,
        real_height_m: Optional[float] = None,
        knows_real_dimensions: bool = False
    )-> bool:

        if len(image_points) != len(world_points) or len(image_points) < 4:
            print('Ошибка: нужно минимум 4 соответствия точек')
            return False

        self.has_real_scale = knows_real_dimensions
        if knows_real_dimensions and real_width_m and real_height_m:
            self.real_width_m = real_width_m
            self.real_height_m = real_height_m
            print(f"Реальный масштаб: {real_width_m}м × {real_height_m}м")
        else:
            self.real_width_m = 1.0
            self.real_height_m = 1.5
            print("Используются относительные единицы (пиксели)")

        try:
            # приводим к numpy массивам
            src_pts = np.array(image_points, dtype=np.float32)
            dst_pts = np.array(world_points, dtype=np.float32)

            # находим матрицу гомографии
            self.homography_matrix, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                5.0
            )

            if self.homography_matrix is not None:
                self.calibrated = True
                return True
            return False
        except Exception as e:
            print(f"Ошибка калибровки гомографии: {e}")
            return False

    def get_scale_info(self):
        return {
            'has_real_scale': self.has_real_scale,
            'real_width_m': self.real_width_m,
            'real_height_m': self.real_height_m,
            'units': 'meters' if self.has_real_scale else 'pixels'
        }

    # преобразует одну точку из координат изображения в координаты карты
    def transform_point(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:

        if not self.calibrated:
            return None

        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        pt = np.array([pt])

        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        world_point = transformed[0][0]

        return float(world_point[0]), float(world_point[1])

    # преобразует траекторию bbox (по кадрам) в траекторию точек на кадре
    def transform_trajectory(
        self,
        trajectory: List[Tuple[int, List[float]]]
    ) -> List[Tuple[int, Tuple[float, float]]]:

        if not self.calibrated:
            return []

        world_trajectory = []
        for frame_id, bbox in trajectory:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = y2

            world_point = self.transform_point((center_x, center_y))
            if world_point:
                world_trajectory.append((frame_id, world_point))

        return world_trajectory

    def save(self, filepath: Path):
        if not self.calibrated:
            raise ValueError('Гомография еще не откалибрована')

        data = {
            "homography_matrix": self.homography_matrix.tolist(),
            "calibrated": self.calibrated,
            "has_real_scale": self.has_real_scale,  # ДОБАВИЛИ
            "real_width_m": self.real_width_m,  # ДОБАВИЛИ
            "real_height_m": self.real_height_m,  # ДОБАВИЛИ
            "version": "1.1"  # Версия для обратной совместимости
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path):
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)

        self.homography_matrix = np.array(data["homography_matrix"])
        self.calibrated = data.get("calibrated", True)

        self.has_real_scale = data.get("has_real_scale", False)
        self.real_width_m = data.get("real_width_m", 1.0)
        self.real_height_m = data.get("real_height_m", 1.5)

        if "has_real_scale" not in data:
            print("Загружена старая версия калибровки, используется относительный масштаб")
            self.has_real_scale = False