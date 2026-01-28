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

    # калибрует гомографию по соответствиям точек
    # аргументы: список координат в изображении и карты пола, возвращает true если калибровка успешна
    def calibrate(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    )-> bool:

        if len(image_points) != len(world_points) or len(image_points) < 4:
            print('Ошибка: нужно минимум 4 соответствия точек')
            return False

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
            'homography_matrix': self.homography_matrix.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path):
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.homography_matrix = np.array(data['homography_matrix'])
        self.calibrated = True